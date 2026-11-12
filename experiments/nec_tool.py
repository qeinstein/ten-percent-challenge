#!/usr/bin/env python3
"""Task-calibrated, whole-transformer NEC sparse-Schmidt experiment (CPU)."""
import argparse, gzip, json, os, shutil, sys, time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '/home/fluxx/Workspace/research')
from nox.sparse_schmidt import compile_sparse_schmidt
from nox.packed_schmidt import reconstruct_packed_schmidt
from tool_prune import examples, batched

MODEL='HuggingFaceTB/SmolLM2-135M-Instruct'
REV='12fd25f77366fa6b3b4b768ec3050bf629380bac'

def load_model():
    return AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, local_files_only=True, dtype=torch.float32).eval()

def evaluate(model, tok, samples, batch, head):
    pred=[]; gold=[]
    with torch.inference_mode():
        for enc,y in batched(tok,samples,batch):
            h=model.model(**enc).last_hidden_state[:,-1,:].float()
            pred.extend((h @ head.T).argmax(-1).tolist()); gold.extend(y.tolist())
    return {'correct':sum(a==b for a,b in zip(pred,gold)), 'total':len(gold), 'accuracy':sum(a==b for a,b in zip(pred,gold))/len(gold)}

def capture(model,tok,samples,batch,tokens_per_example):
    cache={}; handles=[]; active_mask=None
    for n,m in model.named_modules():
        if n.startswith('model.layers.') and isinstance(m,torch.nn.Linear):
            cache[n]=[]
            def hook(mod,args,n=n):
                # Sample across every non-padding prompt, including request and tools.
                states=args[0].detach(); pieces=[]
                for row in range(states.shape[0]):
                    positions=active_mask[row].nonzero().flatten()
                    take=torch.linspace(0, len(positions)-1, tokens_per_example).long()
                    pieces.append(states[row,positions[take],:])
                x=torch.cat(pieces,dim=0).T.contiguous()
                cache[n].append(x.cpu())
            handles.append(m.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for enc,_ in batched(tok,samples,batch):
            active_mask=enc['attention_mask']; model.model(**enc)
    for h in handles: h.remove()
    return {k:torch.cat(v,dim=1) for k,v in cache.items()}

def pack_embedding(w):
    a=w.float(); scale=a.abs().amax(dim=1).clamp_min(1e-10)/7
    q=(a/scale[:,None]).round().clamp(-7,7).to(torch.int8).add(8).to(torch.uint8)
    if q.shape[1]%2: q=torch.nn.functional.pad(q,(0,1),value=8)
    return {'shape':tuple(w.shape),'scale':scale.half(),'values':q[:,::2] | (q[:,1::2]<<4)}
def unpack_embedding(p):
    q=p['values']; a=torch.empty((q.shape[0],q.shape[1]*2),dtype=torch.int16); a[:,::2]=q&15; a[:,1::2]=q>>4
    return (a[:,:p['shape'][1]].float()-8)*p['scale'].float()[:,None]

def save_gz(obj,path):
    with gzip.open(path,'wb',compresslevel=6) as f: torch.save(obj,f)
def load_gz(path):
    with gzip.open(path,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)

def main():
    global MODEL, REV
    p=argparse.ArgumentParser(); p.add_argument('--model',default=MODEL); p.add_argument('--revision',default=REV); p.add_argument('--calibration',type=int,default=32); p.add_argument('--test',type=int,default=64); p.add_argument('--batch',type=int,default=4); p.add_argument('--tokens-per-example',type=int,default=8); p.add_argument('--matrix-ratio',type=float,default=.20); p.add_argument('--max-layers',type=int,default=30); p.add_argument('--out',default='results/nec_smollm'); p.add_argument('--threads',type=int,default=4); args=p.parse_args()
    MODEL, REV = args.model, args.revision
    torch.set_num_threads(args.threads); torch.manual_seed(1234); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'
    model=load_model(); ids=[tok.encode(' '+c,add_special_tokens=False) for c in 'ABCD']; assert all(len(x)==1 for x in ids),ids
    head=model.lm_head.weight.detach()[[x[0] for x in ids]].float().clone()
    base_bytes=sum(p.numel()*2 for p in model.parameters()); print('unique FP16 weight bytes',base_bytes,flush=True)
    cal=examples(10,args.calibration); test=examples(9001,args.test)
    baseline=evaluate(model,tok,test,args.batch,head); print('baseline',baseline,flush=True)
    activations=capture(model,tok,cal,args.batch,args.tokens_per_example); print('captured',len(activations),'operators',flush=True)
    modules=dict(model.named_modules()); results=[]; begin=time.time()
    names=[n for n in activations if int(n.split('.')[2]) < args.max_layers]
    for idx,name in enumerate(names):
        mod=modules[name]; w=mod.weight.detach().float(); x=activations.pop(name)
        cap=int(w.numel()*2*args.matrix_ratio)
        start=time.time()
        artifact, decoded, diag=compile_sparse_schmidt(w,byte_budget=cap,coefficient_budget=w.numel(),calibration_inputs=x,backbone_fraction=.25,amplitude_bits=3,excitation_cg_iterations=2,excitation_sequence_length=8,bits_candidates=(3,),group_size=32,als_sweeps=0,layout_policy='packed_efficiency',projection_name=name.split('.')[-1],power_iterations=0,oversample=2)
        f=out/(name.replace('.','_')+'.pt.gz'); save_gz(artifact,f)
        loaded=load_gz(f); restored=reconstruct_packed_schmidt(loaded)
        if not torch.equal(decoded,restored): raise RuntimeError('serialized artifact failed parity '+name)
        with torch.no_grad(): mod.weight.copy_(restored)
        row={'name':name,'payload_bytes':artifact['payload_bytes'],'serialized_bytes':f.stat().st_size,'seconds':round(time.time()-start,2),'density':diag['sparse_density']}; results.append(row)
        print(f'{idx+1}/{len(names)} '+json.dumps(row),flush=True)
        del artifact,decoded,loaded,restored,diag,x
    # Store retained tensors once; output rows deliberately specialize this task.
    emb=pack_embedding(model.model.embed_tokens.weight.detach()); save_gz({'embedding':emb,'head':head.half(),'norms':{n:p.detach().half() for n,p in model.named_parameters() if 'norm' in n}},out/'retained.pt.gz')
    with torch.no_grad(): model.model.embed_tokens.weight.copy_(unpack_embedding(emb))
    tok.save_pretrained(out/'tokenizer'); model.config.save_pretrained(out/'config')
    result=evaluate(model,tok,test,args.batch,head)
    total=sum(f.stat().st_size for f in out.rglob('*') if f.is_file() and f.suffix!='.json')
    report={'model':MODEL,'revision':REV,'calibration_examples':len(cal),'calibration_activation_tokens':len(cal)*args.tokens_per_example,'test_examples':len(test),'baseline':baseline,'compressed':result,'compiled_operators':len(results),'expected_transformer_operators':210,'layers':args.max_layers,'weight_fp16_unique_bytes':base_bytes,'package_bytes_incomplete':total,'incomplete_package_ratio':total/base_bytes,'operator_results':results,'elapsed_seconds':round(time.time()-begin,1),'limitations':['Synthetic template diagnostic; examples may overlap in wording.','Package excludes config JSON, inference loader and NEC dependencies; partial-layer runs also omit untouched weights.','Calibration samples positions across prompts rather than every token.','No feedback-learning test.']}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n'); print('RESULT '+json.dumps({k:v for k,v in report.items() if k!='operator_results'}),flush=True)
if __name__=='__main__': main()
