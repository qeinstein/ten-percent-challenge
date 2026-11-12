#!/usr/bin/env python3
"""A single-load, task-calibrated density sweep for Qwen tool selection.

Pruning scores use labeled calibration losses only; model weights are frozen.
Candidates are selected on public development data. This script never reads
the locked split or invokes the final-exam runner.
"""
import argparse, hashlib, json, random, shutil, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import qwen_nec as qn
from weightspace import compile_matrix, reconstruct_matrix, add_task_weighted_residual

def atomic_json(path,data):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n'); tmp.replace(path)

def package_size(root):
    return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

def replace_package(source,dest):
    dest=Path(dest)
    if dest.exists(): shutil.rmtree(dest)
    shutil.copytree(source,dest)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',default='/workspace/qwen-search')
    ap.add_argument('--data-dir',default='gpu/prepared')
    ap.add_argument('--densities',default='0.02,0.05,0.10,0.15,0.20,0.25,0.30')
    ap.add_argument('--calibration',type=int,default=512)
    ap.add_argument('--dev',type=int,default=963)
    ap.add_argument('--gradient-batch-size',type=int,default=1)
    ap.add_argument('--sequential-calibration',action='store_true',help='Refresh each layer input statistics through the compressed prefix')
    ap.add_argument('--sequential-examples',type=int,default=128)
    ap.add_argument('--sequential-positions',type=int,default=32)
    ap.add_argument('--quantization',choices=('ternary','int4'),default='ternary')
    ap.add_argument('--residual-rank',type=int,default=0,help='Optional task-weighted closed-form residual rank per operator')
    ap.add_argument('--max-length',type=int,default=512)
    ap.add_argument('--group-size',type=int,default=128)
    ap.add_argument('--budgets',default='0.04,0.10')
    ap.add_argument('--limit-seconds',type=int,default=7200)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    qn.DEVICE='cuda'
    if not torch.cuda.is_available(): raise SystemExit('CUDA GPU required')
    densities=sorted(set(float(x) for x in a.densities.split(',')))
    budgets=sorted(set(float(x) for x in a.budgets.split(',')))
    if not densities or any(not 0<d<=1 for d in densities): raise SystemExit('invalid density list')
    out=Path(a.output)
    if out.exists() and any(out.iterdir()): raise SystemExit('output directory is not empty')
    out.mkdir(parents=True,exist_ok=True); started=time.monotonic(); deadline=started+a.limit_seconds
    source=Path(qn.__file__).resolve().parent
    files=('search_qwen.py','qwen_nec.py','weightspace.py','reload_qwen.py','feedback.py')
    atomic_json(out/'run_config.json',{'arguments':vars(a),'model':qn.MODEL,'revision':qn.REVISION,
        'manifest_sha256':hashlib.sha256((Path(a.data_dir)/'manifest.json').read_bytes()).hexdigest(),
        'source_sha256':{f:hashlib.sha256((source/f).read_bytes()).hexdigest() for f in files}})
    torch.manual_seed(a.seed); torch.set_grad_enabled(False)
    free,total=torch.cuda.mem_get_info()
    print('DEVICE',torch.cuda.get_device_name(0),'VRAM free/total',free,total,'budget',a.limit_seconds,flush=True)
    data=Path(a.data_dir); cal=qn.read_frozen_data(data,'calib'); dev_pool=qn.read_frozen_data(data,'dev'); feedback_rows=qn.read_frozen_data(data,'feedback')
    rng=random.Random(a.seed); chosen=rng.sample(cal,min(a.calibration,len(cal))); dev=rng.sample(dev_pool,min(a.dev,len(dev_pool)))
    qn.atomic_json(out/'selected_examples.json',{'calibration_ids':[r.get('id') for r in chosen],'development_ids':[r.get('id') for r in dev]})
    dev_path=out/'development.jsonl'; dev_path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in dev))
    tok=AutoTokenizer.from_pretrained(qn.MODEL,revision=qn.REVISION); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    letter_ids=[tok.encode(' '+c,add_special_tokens=False) for c in qn.LETTERS]
    if any(len(x)!=1 for x in letter_ids): raise RuntimeError('answer tokens must be single tokens')
    model=AutoModelForCausalLM.from_pretrained(qn.MODEL,revision=qn.REVISION,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval(); model.config.use_cache=False
    fp16_bytes=sum(p.numel()*2 for p in model.parameters()); head=model.lm_head.weight.detach()[[x[0] for x in letter_ids]].float().clone(); model.lm_head=torch.nn.Identity()
    fallback=tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
    keep=qn.select_vocab(tok,cal[:min(512,len(cal))],0)
    if fallback not in keep.tolist(): keep=torch.unique(torch.cat((keep,torch.tensor([fallback]))),sorted=True)
    mapping,_=qn.remapper(model.model.embed_tokens.num_embeddings,keep,fallback,qn.DEVICE)
    baseline=qn.score(model,tok,dev,head,max_length=a.max_length)
    retained=qn.pack_embeddings(model,keep,head); qn.apply_embedding(model,retained)
    embedding_only=qn.score(model,tok,dev,head,max_length=a.max_length)
    retained['other']={n:p.detach().cpu().clone() for n,p in model.named_parameters() if not n.startswith('model.embed_tokens.') and not n.startswith('lm_head.') and (p.ndim==1 or not n.startswith('model.layers.'))}
    common=out/'common_package'; common.mkdir()
    qn.write_gz(retained,common/'retained.pt.gz'); tok.save_pretrained(common/'tokenizer'); model.config.save_pretrained(common/'config')
    for f in ('weightspace.py','reload_qwen.py','feedback.py'): shutil.copy2(source/f,common/f)
    print('BASELINE',json.dumps(baseline),'EMBEDDING_ONLY',json.dumps(embedding_only),flush=True)
    print('TASK_GRAD_CALIBRATION_START',len(chosen),flush=True)
    acts,out_importance=qn.capture_task_gradient(model,tok,chosen,mapping,head,a.max_length,batch_size=a.gradient_batch_size)
    modules=dict(model.named_modules()); names=list(acts)
    originals={n:modules[n].weight.detach().cpu().clone() for n in names}
    print('TASK_GRAD_CALIBRATION_DONE',len(names),'operators',flush=True)
    trials=[]; best={b:None for b in budgets}; tmp=out/'candidate_tmp'
    for di,density in enumerate(densities):
        if time.monotonic()+180>deadline:
            print('TIME_BUDGET_STOP before_density',density,flush=True); break
        if tmp.exists(): shutil.rmtree(tmp)
        package=tmp/'package'; shutil.copytree(common,package)
        writer=ThreadPoolExecutor(max_workers=1); pending=[]
        compile_start=time.monotonic(); print('CANDIDATE_START',density,di+1,'/',len(densities),flush=True)
        current_group=None
        for i,name in enumerate(names):
            layer='.'.join(name.split('.')[:3])+'.'
            suffix=name.rsplit('.',1)[-1]
            stage=('qkv' if suffix in ('q_proj','k_proj','v_proj') else 'attention_out' if suffix=='o_proj' else 'mlp_in' if suffix in ('gate_proj','up_proj') else 'mlp_out')
            group=(layer,stage)
            if a.sequential_calibration and group!=current_group:
                # Refresh after attention and SwiGLU groups as well as after
                # each layer, so their inputs reflect the compressed prefix.
                refreshed=qn.capture(model,tok,chosen[:a.sequential_examples],mapping,a.sequential_positions,a.max_length,name_prefix=layer)
                if not refreshed: raise RuntimeError('no operators captured for '+layer)
                acts.update(refreshed); current_group=group
                print('SEQUENTIAL_CALIBRATION',density,layer,stage,len(chosen[:a.sequential_examples]),flush=True)
            mod=modules[name]; w=originals[name].to(qn.DEVICE)
            artifact=compile_matrix(w,acts[name],density=density,group_size=a.group_size,allocation='rowwise',output_importance=out_importance[name],quantization=a.quantization)
            if a.residual_rank:
                artifact=add_task_weighted_residual(artifact,w,acts[name],out_importance[name],rank=a.residual_rank)
            decoded=reconstruct_matrix(artifact,device=qn.DEVICE)
            mod.weight.copy_(decoded.to(mod.weight.dtype))
            path=package/(name.replace('.','_')+'.pt.gz'); pending.append(writer.submit(qn.write_gz,artifact,path))
            if len(pending)>=2: pending.pop(0).result()
            del w,artifact,decoded
            if (i+1)%16==0: print('CANDIDATE_PROGRESS',density,i+1,'/',len(names),flush=True)
        for f in pending: f.result()
        writer.shutdown()
        old_cache,token_stats=qn.feature_cache(model,tok,dev,head,mapping,a.max_length)
        compressed=qn.cache_accuracy(old_cache); compressed.update(token_stats)
        feedback,fb_report=qn.evaluate_feedback(model,tok,feedback_rows,old_cache,head,mapping,a.max_length)
        feedback.save(package/'feedback.pt')
        size=package_size(package); ratio=size/fp16_bytes
        trial={'density':density,'allocation':'rowwise','quantization':a.quantization,'residual_rank':a.residual_rank,'calibration_mode':'task_gradient_plus_'+('sequential_prefix_activation_rms' if a.sequential_calibration else 'dense_activation_rms'),'dev_before_feedback':compressed,'choice_diagnostics':qn.choice_diagnostics(old_cache),'feedback':fb_report,'artifact_bytes':size,'artifact_ratio':ratio,'compile_seconds':time.monotonic()-compile_start}
        trials.append(trial)
        for budget in budgets:
            if ratio<=budget:
                score=fb_report['old_dev_after']['accuracy']; key=(score,compressed['accuracy'],-ratio)
                if best[budget] is None or key>best[budget]['key']:
                    dest=out/('best_4pct' if budget<=.040001 else f'best_{int(budget*100)}pct')/'package'
                    replace_package(package,dest)
                    best[budget]={'key':key,'density':density,'artifact_ratio':ratio,'artifact_bytes':size,'dev_before_feedback':compressed,'dev_after_feedback':fb_report['old_dev_after'],'feedback':fb_report}
        atomic_json(out/'sweep_progress.json',{'trials':trials,'best':{str(k):v for k,v in best.items()},'elapsed_seconds':time.monotonic()-started})
        print('CANDIDATE_RESULT',json.dumps(trial),flush=True)
        shutil.rmtree(tmp)
        if torch.cuda.is_available(): torch.cuda.synchronize()
    result={'status':'complete' if len(trials)==len(densities) else 'partial_time_budget','model':qn.MODEL,'revision':qn.REVISION,'data_revision':qn.DATA_REV,'dense_dev':baseline,'embedding_only_dev':embedding_only,'fp16_bytes':fp16_bytes,'quantization':a.quantization,'calibration_examples':len(chosen),'development_examples':len(dev),'activation_and_task_gradient_tokens':'all non-padding calibration positions','trials':trials,'best_by_size_budget':{str(k):({x:y for x,y in v.items() if x!='key'} if v else None) for k,v in best.items()},'elapsed_seconds':time.monotonic()-started,'locked_exam_used':False,'limitations':['Calibration and development source is public C2 data; results are exploratory and may overlap pretraining.','The selected split is development data and is not an unseen challenge exam.','Task gradients are used only to rank/prune weights; no optimizer or parameter update is performed.','Feedback labels are simulated from public labeled examples, not human annotations.','The task scores tool choice from candidate names; it does not measure argument generation or API execution.']}
    if common.exists(): shutil.rmtree(common)
    atomic_json(out/'summary.json',result); print('SWEEP_RESULT',json.dumps(result),flush=True)
    if tmp.exists(): shutil.rmtree(tmp)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    # Verify the highest-accuracy package within the largest requested budget
    # by independently rebuilding it and checking pre/post-feedback scores.
    largest=max(budgets); selected=best.get(largest)
    if selected is not None and time.monotonic()+180<deadline:
        package=out/(('best_4pct' if largest<=.040001 else f'best_{int(largest*100)}pct'))/'package'
        if package.exists():
            del model,originals,modules; torch.cuda.empty_cache()
            import subprocess
            check=subprocess.run([sys.executable,str(package/'reload_qwen.py'),str(package),'--dev-jsonl',str(dev_path),'--limit',str(len(dev)),'--compare-feedback','--feedback-file',str(package/'feedback.pt')],check=True,text=True,capture_output=True,timeout=max(60,int(deadline-time.monotonic())))
            atomic_json(out/'reload_verification.json',json.loads(check.stdout.strip().splitlines()[-1]))
            print('RELOAD_VERIFICATION',check.stdout.strip().splitlines()[-1],flush=True)

if __name__=='__main__': main()
