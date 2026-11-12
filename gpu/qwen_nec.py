#!/usr/bin/env python3
"""Bounded Qwen experiment using task-aware sparse ternary weight programming.

This is an experimental GPU runner, not a proven 4% solution. It saves each
operator as an atomic gzip checkpoint and reconstructs dense weights for eval.
"""
import argparse, gzip, hashlib, io, json, os, random, shutil, signal, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from weightspace import compile_matrix, reconstruct_matrix
from my_tools import make
from feedback import Feedback, tool_key

DEVICE='cuda'
MODEL='Qwen/Qwen2.5-7B-Instruct'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
DATA_REV='public-C2-content-split-v1'
LETTERS='ABCDE'

def atomic_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)

def make_data(root,split,n,seed):
    root.mkdir(parents=True,exist_ok=True); path=root/(split+'.jsonl')
    rows=make(split,n,seed); data=''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows).encode()
    if path.exists() and path.read_bytes()!=data: raise ValueError('dataset conflict: '+str(path))
    path.write_bytes(data)
    print('DATA',split,'rows',len(rows),'sha256',hashlib.sha256(data).hexdigest(),flush=True)
    return rows

def read_frozen_data(root,split):
    root=Path(root); manifest=json.loads((root/'manifest.json').read_text()); name=split+'.jsonl'; data=(root/name).read_bytes()
    expected=manifest['files'][name]['sha256']
    if hashlib.sha256(data).hexdigest()!=expected: raise ValueError('split hash mismatch: '+name)
    return [json.loads(line) for line in data.splitlines() if line.strip()]

def prompt(row):
    return row['inst']+'\n\nText: '+row['text']+'\n\nOptions:\n'+''.join(f'{l}. {c}\n' for l,c in zip(LETTERS,row['names']))+'Answer:'

def write_gz(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wb',compresslevel=4) as f: torch.save(obj,f)
    os.replace(tmp,path)

def read_gz(path):
    with gzip.open(path,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)

def to_cpu(x):
    if isinstance(x,torch.Tensor): return x.detach().cpu()
    if isinstance(x,dict): return {k:to_cpu(v) for k,v in x.items()}
    if isinstance(x,list): return [to_cpu(v) for v in x]
    if isinstance(x,tuple): return tuple(to_cpu(v) for v in x)
    return x

def select_vocab(tok,rows,n):
    if n<=0: return torch.arange(len(tok),dtype=torch.long)
    counts=torch.zeros(len(tok),dtype=torch.int64)
    for row in rows:
        ids=tok.encode(prompt(row),add_special_tokens=True)
        counts.index_add_(0,torch.tensor(ids),torch.ones(len(ids),dtype=torch.int64))
    special=set(tok.all_special_ids)
    for c in LETTERS: special.update(tok.encode(' '+c,add_special_tokens=False))
    ids=set(counts.topk(min(n,len(counts))).indices.tolist())|special
    return torch.tensor(sorted(ids),dtype=torch.long)

def remapper(vocab_size,keep,fallback,device):
    mask=torch.zeros(vocab_size,dtype=torch.bool,device=device); mask[keep.to(device)]=True
    mapping=torch.arange(vocab_size,device=device); mapping[~mask]=fallback
    return mapping,mask

def score(model,tok,rows,head,mapping=None,max_length=512,batch_size=4,feedback=None,return_predictions=False):
    correct=total=oov=tokens=0; predictions=[]
    with torch.inference_mode():
        for offset in range(0,len(rows),batch_size):
            group=rows[offset:offset+batch_size]
            enc=tok([prompt(r) for r in group],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
            ids=enc.input_ids.to(DEVICE); attention=enc.attention_mask.to(DEVICE)
            if mapping is not None:
                mapped=mapping[ids]; oov+=int(((mapped!=ids)*attention.bool()).sum()); tokens+=int(attention.sum()); ids=mapped
            hidden=model.model(input_ids=ids,attention_mask=attention,use_cache=False).last_hidden_state[:,-1,:].float()
            logits=hidden@head.T
            if feedback is not None:
                choices=[int(feedback.adjust(logits[i],hidden[i],row['names']).argmax()) for i,row in enumerate(group)]
            else: choices=logits.argmax(-1).tolist()
            correct+=sum(a==int(r['answer']) for a,r in zip(choices,group)); total+=len(group); predictions.extend(choices)
    result={'correct':correct,'total':total,'accuracy':correct/total if total else None,'oov_tokens':oov,'input_tokens':tokens}
    if return_predictions: result['predictions']=predictions
    return result

def capture(model,tok,rows,mapping,npos,max_length,batch_size=4,name_prefix=None):
    """Stream per-channel second moments on device, in padded batches."""
    stats={}; counts={}; hooks=[]; positions=None
    for name,mod in model.named_modules():
        if name.startswith('model.layers.') and isinstance(mod,torch.nn.Linear) and (name_prefix is None or name.startswith(name_prefix)):
            stats[name]=torch.zeros(mod.in_features,device=mod.weight.device,dtype=torch.float32); counts[name]=0
            def hook(m,args,name=name):
                x=args[0].detach(); idx=positions.unsqueeze(-1).expand(-1,-1,x.shape[-1])
                selected=x.gather(1,idx).float(); stats[name].add_(selected.square().sum(dim=(0,1))); counts[name]+=selected.shape[0]*selected.shape[1]
            hooks.append(mod.register_forward_pre_hook(hook))
    try:
        with torch.inference_mode():
            for off in range(0,len(rows),batch_size):
                group=rows[off:off+batch_size]
                enc=tok([prompt(row) for row in group],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
                ids=enc.input_ids.to(DEVICE); attention=enc.attention_mask.to(DEVICE)
                lengths=attention.sum(-1); start=ids.shape[1]-lengths
                fractions=torch.linspace(0,1,npos,device=ids.device)
                positions=start[:,None]+((lengths[:,None]-1)*fractions).long()
                model.model(input_ids=mapping[ids],attention_mask=attention,use_cache=False)
                if (off//batch_size)%8==0: print('CALIBRATION',min(off+batch_size,len(rows)),'/',len(rows),flush=True)
    finally:
        for hook in hooks: hook.remove()
    return {k:(v/counts[k]).sqrt() for k,v in stats.items()}

def capture_task_gradient(model,tok,rows,mapping,head,max_length,batch_size=1):
    """Collect activation RMS and output-gradient RMS for task-loss saliency.

    The labels only determine pruning scores; no parameter is updated. Frozen
    model parameters keep autograd's memory use lower while gradients still
    flow through intermediate activations.
    """
    import torch.nn.functional as F
    modules={n:m for n,m in model.named_modules() if n.startswith('model.layers.') and isinstance(m,torch.nn.Linear)}
    x2={n:torch.zeros(m.in_features,device=m.weight.device,dtype=torch.float32) for n,m in modules.items()}
    d2={n:torch.zeros(m.out_features,device=m.weight.device,dtype=torch.float32) for n,m in modules.items()}
    token_count=0; targets=[]; target_names=[]; mask_ref=[None]
    prehooks=[]; hooks=[]
    for p in model.parameters(): p.requires_grad_(False)
    def require_embedding_grad(mod,args,out):
        out.requires_grad_(True)
    emb_hook=model.model.embed_tokens.register_forward_hook(require_embedding_grad)
    for name,mod in modules.items():
        def pre(m,args,name=name):
            x=args[0].detach().float(); mask=mask_ref[0].unsqueeze(-1).float()
            x2[name].add_((x.square()*mask).sum(dim=(0,1)))
        def post(m,args,out,name=name):
            targets.append(out); target_names.append(name)
        prehooks.append(mod.register_forward_pre_hook(pre)); hooks.append(mod.register_forward_hook(post))
    try:
        for off in range(0,len(rows),batch_size):
            group=rows[off:off+batch_size]
            enc=tok([prompt(r) for r in group],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
            ids=enc.input_ids.to(DEVICE); attention=enc.attention_mask.to(DEVICE); mask_ref[0]=attention.bool()
            token_count+=int(attention.sum())
            labels=torch.tensor([int(r['answer']) for r in group],device=DEVICE)
            with torch.enable_grad():
                hidden=model.model(input_ids=mapping[ids],attention_mask=attention,use_cache=False).last_hidden_state[:,-1,:].float()
                loss=F.cross_entropy(hidden@head.T,labels)
                grads=torch.autograd.grad(loss,targets,allow_unused=False,retain_graph=False)
            for name,grad in zip(target_names,grads):
                d2[name].add_((grad.detach().float().square()*mask_ref[0].unsqueeze(-1)).sum(dim=(0,1)))
            targets.clear(); target_names.clear()
            del grads,loss,hidden,ids,attention,labels,enc
            if (off//batch_size)%32==0: print('TASK_GRAD_CALIBRATION',min(off+batch_size,len(rows)),'/',len(rows),flush=True)
    finally:
        emb_hook.remove()
        for h in prehooks+hooks: h.remove()
    acts={n:(v/max(token_count,1)).sqrt() for n,v in x2.items()}
    grads={n:(v/max(token_count,1)).sqrt() for n,v in d2.items()}
    return acts,grads

def pack_embeddings(model,keep,head):
    emb=model.model.embed_tokens.weight; vals=[]; scales=[]
    for start in range(0,len(keep),512):
        w=emb.index_select(0,keep[start:start+512].to(emb.device)).float()
        scale=w.abs().amax(dim=1).clamp_min(1e-10)/7
        q=(w/scale[:,None]).round().clamp(-7,7).to(torch.int8).add(8).to(torch.uint8)
        if q.shape[1]%2:q=torch.nn.functional.pad(q,(0,1),value=8)
        vals.append((q[:,::2] | (q[:,1::2]<<4)).cpu()); scales.append(scale.half().cpu())
    return {'ids':keep.cpu(),'values':torch.cat(vals),'scales':torch.cat(scales),'hidden_size':emb.shape[1], 'head':head.cpu()}

def apply_embedding(model,packed):
    ids=packed['ids']; p=packed['values']; s=packed['scales']
    with torch.no_grad():
        for i in range(0,len(ids),512):
            q=p[i:i+512].to(DEVICE); a=torch.empty((len(q),q.shape[1]*2),device=DEVICE,dtype=torch.int16); a[:,::2]=q&15; a[:,1::2]=q>>4
            w=(a[:,:packed['hidden_size']].float()-8)*s[i:i+512].to(DEVICE).float()[:,None]
            model.model.embed_tokens.weight.index_copy_(0,ids[i:i+512].to(DEVICE),w.to(model.model.embed_tokens.weight.dtype))

def feature_cache(model,tok,rows,head,mapping,max_length,batch_size=4):
    cached=[]; stats={'oov_tokens':0,'input_tokens':0}
    with torch.inference_mode():
        for offset in range(0,len(rows),batch_size):
            group=rows[offset:offset+batch_size]
            enc=tok([prompt(r) for r in group],return_tensors='pt',padding=True,truncation=True,max_length=max_length)
            ids=enc.input_ids.to(DEVICE); attention=enc.attention_mask.to(DEVICE)
            mapped=mapping[ids]; stats['oov_tokens']+=int(((mapped!=ids)*attention.bool()).sum()); stats['input_tokens']+=int(attention.sum())
            h=model.model(input_ids=mapped,attention_mask=attention,use_cache=False).last_hidden_state[:,-1,:].float()
            logits=h@head.T
            for i,row in enumerate(group): cached.append((row,h[i].cpu(),logits[i].cpu()))
    return cached,stats

def cache_accuracy(cached,feedback=None):
    correct=0
    for row,h,logits in cached:
        adjusted=feedback.adjust(logits,h,row['names']) if feedback is not None else logits
        correct+=int(int(adjusted.argmax())==int(row['answer']))
    return {'correct':correct,'total':len(cached),'accuracy':correct/len(cached) if cached else None}

def choice_diagnostics(cached,feedback=None):
    """Report gold/predicted option counts and a gold-by-prediction matrix."""
    confusion=[[0]*5 for _ in range(5)]
    for row,h,logits in cached:
        scores=feedback.adjust(logits,h,row['names']) if feedback is not None else logits
        gold=int(row['answer']); predicted=int(scores.argmax())
        if not 0<=gold<5 or not 0<=predicted<5: raise ValueError('choice index outside 0..4')
        confusion[gold][predicted]+=1
    return {'gold_counts':[sum(r) for r in confusion],
            'prediction_counts':[sum(r[j] for r in confusion) for j in range(5)],
            'confusion_gold_by_prediction':confusion}

def evaluate_feedback(model,tok,rows,old_cache,head,mapping,max_length):
    groups={}
    for row in rows:
        answer=int(row['answer'])
        if answer>=len(row['names'])-1: continue
        groups.setdefault(tool_key(row['names'][answer]),[]).append(row)
    chosen=[(k,v) for k,v in sorted(groups.items()) if len(v)>=3][:16]
    feedback=Feedback(strength=0); annotations=[]; tune=[]; held=[]
    with torch.inference_mode():
        for key,items in chosen:
            row=items[0]; ids=tok(prompt(row),return_tensors='pt',truncation=True,max_length=max_length).input_ids.to(DEVICE)
            hidden=model.model(input_ids=mapping[ids],use_cache=False).last_hidden_state[0,-1,:]
            feedback.ingest(hidden,row['names'][int(row['answer'])]); annotations.append(row.get('id',key))
            tune.append(items[1]); held.append(items[2])
    tune_cache,_=feature_cache(model,tok,tune,head,mapping,max_length)
    held_cache,_=feature_cache(model,tok,held,head,mapping,max_length)
    old_before=cache_accuracy(old_cache); new_before=cache_accuracy(held_cache)
    trials=[]
    for strength in (0.,.1,.25,.5,1.,2.):
        feedback.strength=strength
        t=cache_accuracy(tune_cache,feedback); old=cache_accuracy(old_cache,feedback)
        trials.append({'strength':strength,'tune_correct':t['correct'],'old_correct':old['correct']})
    eligible=[t for t in trials if t['old_correct']>=old_before['correct']]
    selected=max(eligible,key=lambda t:(t['tune_correct'],-t['strength']))
    feedback.strength=selected['strength']
    return feedback,{'annotations':len(annotations),'annotation_ids':annotations,'tune_examples':len(tune),'trials':trials,'unseen_new_before':new_before,'unseen_new_after':cache_accuracy(held_cache,feedback),'old_dev_before':old_before,'old_dev_after':cache_accuracy(old_cache,feedback),'strength':feedback.strength,'method':'one annotation per tool; running hidden-state means; tune strength on disjoint examples and preserve old dev; no example text stored'}

def main():
    global MODEL, REVISION, DEVICE
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='/workspace/qwen-direct'); ap.add_argument('--model',default=MODEL); ap.add_argument('--revision',default=REVISION); ap.add_argument('--device',choices=('cuda','cpu'),default='cuda'); ap.add_argument('--data-dir',default='gpu/prepared'); ap.add_argument('--calibration',type=int,default=256); ap.add_argument('--calibration-positions',type=int,default=32); ap.add_argument('--calibration-mode',choices=('activation','task-gradient'),default='activation'); ap.add_argument('--vocab',type=int,default=0); ap.add_argument('--vocab-source',type=int,default=512); ap.add_argument('--dev',type=int,default=256); ap.add_argument('--max-length',type=int,default=512); ap.add_argument('--density',type=float,default=.02); ap.add_argument('--group-size',type=int,default=128); ap.add_argument('--allocation',choices=('rowwise','groupwise'),default='rowwise'); ap.add_argument('--limit-seconds',type=int,default=6600); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    MODEL, REVISION, DEVICE = args.model, args.revision, args.device
    if DEVICE=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA GPU required; no job started.')
    if DEVICE=='cpu': torch.set_num_threads(4)
    if args.calibration*args.calibration_positions < 16 or args.calibration*args.calibration_positions%8: raise SystemExit('calibration positions must total a multiple of 8, at least 16')
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True); package=out/'package'; package.mkdir(exist_ok=True)
    start=time.monotonic(); deadline=start+args.limit_seconds; torch.manual_seed(args.seed); torch.set_grad_enabled(False)
    print('DEVICE',DEVICE,'budget seconds',args.limit_seconds,flush=True)
    if DEVICE=='cuda':
        free,total=torch.cuda.mem_get_info(); print('GPU',torch.cuda.get_device_name(0),'VRAM free/total',free,total,flush=True)
        if MODEL=='Qwen/Qwen2.5-7B-Instruct' and free < 25*(1<<30): raise SystemExit('Qwen preflight requires at least 25 GiB free VRAM; no model was loaded.')
    cfg=vars(args).copy(); cfg.pop('limit_seconds'); cfg.pop('output'); cfg.update(model=MODEL,revision=REVISION,data_revision=DATA_REV, data_manifest_sha256=hashlib.sha256((Path(args.data_dir)/'manifest.json').read_bytes()).hexdigest(), runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), compiler_sha256=hashlib.sha256((Path(__file__).parent/'weightspace.py').read_bytes()).hexdigest(), feedback_sha256=hashlib.sha256((Path(__file__).parent/'feedback.py').read_bytes()).hexdigest(), loader_sha256=hashlib.sha256((Path(__file__).parent/'reload_qwen.py').read_bytes()).hexdigest(), dataset_sha256=hashlib.sha256((Path(__file__).parent/'my_tools.py').read_bytes()).hexdigest())
    manifest=out/'run_config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=cfg: raise SystemExit('existing checkpoints use a different config; choose a new output directory')
    atomic_json(manifest,cfg)
    cal=read_frozen_data(args.data_dir,'calib'); dev_pool=read_frozen_data(args.data_dir,'dev'); feedback_rows=read_frozen_data(args.data_dir,'feedback')
    rng=random.Random(args.seed); chosen=rng.sample(cal,args.calibration); dev=rng.sample(dev_pool,min(args.dev,len(dev_pool)))
    atomic_json(out/'selected_examples.json',{'calibration_ids':[r.get('id') for r in chosen],'development_ids':[r.get('id') for r in dev]})
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'; letter_ids=[tok.encode(' '+c,add_special_tokens=False) for c in LETTERS]
    if any(len(x)!=1 for x in letter_ids): raise RuntimeError('answer tokens must be single tokens')
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16 if DEVICE=='cuda' else torch.float32,device_map=DEVICE if DEVICE=='cuda' else None,attn_implementation='sdpa').eval(); model.config.use_cache=False
    fp16_bytes=sum(p.numel()*2 for p in model.parameters()); head=model.lm_head.weight.detach()[[x[0] for x in letter_ids]].float().clone()
    # The task reads only the saved five rows; free the full output head.
    model.lm_head=torch.nn.Identity()
    fallback=tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
    if fallback is None: raise ValueError('tokenizer lacks unknown/eos fallback')
    keep=select_vocab(tok,cal[:args.vocab_source],args.vocab)
    if fallback not in keep.tolist(): keep=torch.unique(torch.cat((keep,torch.tensor([fallback]))),sorted=True)
    mapping,mask=remapper(model.model.embed_tokens.num_embeddings,keep,fallback,DEVICE)
    print('VOCAB',len(keep),'FP16 bytes',fp16_bytes,flush=True)
    baseline=score(model,tok,dev,head,max_length=args.max_length); print('BASELINE',json.dumps(baseline),flush=True)
    retained=pack_embeddings(model,keep,head); apply_embedding(model,retained)
    # Save config, tokenizer, packed embedding, task head, all model norms/biases.
    retained['other']={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('model.embed_tokens.') and not n.startswith('lm_head.') and (p.ndim==1 or not n.startswith('model.layers.'))}
    write_gz(retained,package/'retained.pt.gz'); tok.save_pretrained(package/'tokenizer'); model.config.save_pretrained(package/'config')
    source=Path(__file__).resolve().parent
    for filename in ('weightspace.py','reload_qwen.py','feedback.py'): shutil.copy2(source/filename,package/filename)
    if args.calibration_mode=='task-gradient': acts,output_importance=capture_task_gradient(model,tok,chosen,mapping,head,args.max_length)
    else: acts,output_importance=capture(model,tok,chosen,mapping,args.calibration_positions,args.max_length),None
    modules=dict(model.named_modules()); names=list(acts); results=[]
    writer=ThreadPoolExecutor(max_workers=1); pending=[]
    operator_start=time.monotonic(); operator_elements=sum(modules[n].weight.numel() for n in names); processed_elements=0
    for i,name in enumerate(names):
        if time.monotonic()>deadline:
            for future in pending: future.result()
            writer.shutdown()
            atomic_json(out/'status.json',{'status':'time_budget_reached','completed':i,'total':len(names),'elapsed':time.monotonic()-start,'baseline':baseline}); print('TIME BUDGET REACHED; PARTIAL ONLY',flush=True); return
        path=package/(name.replace('.','_')+'.pt.gz'); mod=modules[name]
        if path.exists():
            artifact=read_gz(path); decoded=reconstruct_matrix(artifact,device=DEVICE); mode='resumed'
        else:
            w=mod.weight.detach(); importance=acts[name]
            artifact=compile_matrix(w,importance,density=args.density,group_size=args.group_size,allocation=args.allocation,output_importance=None if output_importance is None else output_importance[name])
            decoded=reconstruct_matrix(artifact,device=DEVICE)
            pending.append(writer.submit(write_gz,artifact,path))
            if len(pending)>=2: pending.pop(0).result()
            del w,importance; mode='compiled'
        mod.weight.copy_(decoded.to(mod.weight.dtype)); del decoded,artifact; acts.pop(name,None)
        processed_elements+=mod.weight.numel(); elapsed_operators=time.monotonic()-operator_start
        estimated_remaining=elapsed_operators*(operator_elements-processed_elements)/max(processed_elements,1)
        row={'name':name,'mode':mode,'estimated_operator_seconds_remaining':round(estimated_remaining,1)}; results.append(row)
        print('PROGRESS',i+1,'/',len(names),'elapsed',round(time.monotonic()-start,1),json.dumps(row),flush=True)
        atomic_json(out/'status.json',{'status':'compiling','completed':i+1,'total':len(names),'elapsed':time.monotonic()-start,'baseline':baseline})
    for future in pending: future.result()
    writer.shutdown()
    if time.monotonic()>deadline:
        atomic_json(out/'status.json',{'status':'compiled_evaluation_pending','completed':len(names),'total':len(names),'elapsed':time.monotonic()-start,'baseline':baseline})
        print('TIME BUDGET REACHED BEFORE EVALUATION',flush=True); return
    old_cache,token_stats=feature_cache(model,tok,dev,head,mapping,args.max_length)
    compressed=cache_accuracy(old_cache); compressed.update(token_stats)
    feedback, feedback_report = evaluate_feedback(model,tok,feedback_rows,old_cache,head,mapping,args.max_length)
    feedback.save(package/'feedback.pt')
    files=[p for p in package.rglob('*') if p.is_file()]; size=sum(p.stat().st_size for p in files)
    report={'status':'complete','model':MODEL,'revision':REVISION,'data_revision':DATA_REV,'baseline_dev':baseline,'compressed_dev':compressed,'feedback':feedback_report,'unique_fp16_bytes':fp16_bytes,'artifact_bytes':size,'artifact_ratio':size/fp16_bytes,'vocab_rows':len(keep),'calibration_examples':len(chosen),'activation_positions':args.calibration*args.calibration_positions,'operators':len(results),'elapsed_seconds':time.monotonic()-start,'limitations':['Public C2-derived data may overlap model pretraining; development results are not an independent held-out exam.','Artifact includes compiler support and loader but excludes runtime libraries; standalone reload must be verified.','Low input-vocabulary coverage maps omitted tokens to fallback; OOV count is reported.','Feedback prototype statistics may need organizer interpretation; evaluation is preliminary.','Per-layer calibration is collected from the original dense trajectory, not sequentially updated.','Timing limit is checked between operators and does not shut down the GPU.']}
    atomic_json(out/'report.json',report); atomic_json(out/'status.json',report); print('RESULT',json.dumps(report),flush=True)
if __name__=='__main__': main()
