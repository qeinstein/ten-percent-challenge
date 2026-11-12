#!/usr/bin/env python3
"""CPU task-calibrated pruning pilot. No gradient training; artifacts are reloadable."""
import argparse, json, os, random, time, zlib
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.expanduser('~/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M/snapshots/93efa2f097d58c2a74874c7e644dbc9b0cee75a2')
TOOLS = [('weather', 'look up weather forecasts', ['rain','temperature','wind','humidity']), ('calendar', 'manage calendar events', ['meeting','appointment','schedule','event']), ('email', 'send and search email messages', ['email','inbox','message','mail']), ('maps', 'find routes and locations', ['directions','route','address','location']), ('calculator', 'perform mathematical calculations', ['sum','multiply','divide','percentage']), ('web', 'search the web for information', ['research','article','news','website']), ('files', 'find and read local documents', ['document','file','folder','pdf']), ('timer', 'set timers and alarms', ['timer','alarm','countdown','reminder'])]
TEMPLATES = ['I need help with {x} for {y}.', 'Please handle the {x} related to {y}.', 'Can you check {x} concerning {y}?', 'Help me with {x} about {y}.', 'Find the {x} for {y}.', 'I want to deal with {x} for {y}.']
SUBJECTS = ['tomorrow','my work','this afternoon','the next week','my trip','a project','today','the weekend','our team','the morning','my family','a client']
def examples(seed, n):
    rng = random.Random(seed); out=[]
    for _ in range(n):
        target = rng.randrange(len(TOOLS)); others = rng.sample([i for i in range(len(TOOLS)) if i != target], 3)
        opts = others + [target]; rng.shuffle(opts); answer=opts.index(target)
        request=rng.choice(TEMPLATES).format(x=rng.choice(TOOLS[target][2]), y=rng.choice(SUBJECTS))
        prompt='Choose the best tool for the request. Answer with one letter.\nRequest: '+request+'\n'+''.join(f'{chr(65+j)}. {TOOLS[i][0]}: {TOOLS[i][1]}\n' for j,i in enumerate(opts))+'Answer:'
        out.append((prompt,answer))
    return out

def batched(tokenizer, data, batch):
    for i in range(0,len(data),batch):
        part=data[i:i+batch]; enc=tokenizer([x[0] for x in part],return_tensors='pt',padding=True)
        yield enc, torch.tensor([x[1] for x in part])

def evaluate(model,tok,data,batch):
    ids=[tok.encode(' '+c,add_special_tokens=False)[-1] for c in 'ABCD']; pred=[]; labels=[]
    with torch.inference_mode():
        for enc,y in batched(tok,data,batch):
            logits=model(**enc, logits_to_keep=1).logits[:,-1,ids]; pred.extend(logits.argmax(-1).tolist()); labels.extend(y.tolist())
    return pred, labels

def calibrate(model,tok,data,batch):
    stats={}; handles=[]
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.Linear):
            def hook(mod, args, name=name):
                x=args[0].detach().float().reshape(-1,args[0].shape[-1]); val=x.square().sum(0)
                if name not in stats: stats[name]=[val,x.shape[0]]
                else: stats[name][0].add_(val); stats[name][1]+=x.shape[0]
            handles.append(m.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for enc,_ in batched(tok,data,batch): model(**enc, logits_to_keep=1)
    for h in handles: h.remove()
    return {k:(v/n).sqrt() for k,(v,n) in stats.items()}

def encode_tensor(t, importance, keep):
    
    if t.ndim == 2 and t.shape[0] > 512:
        return {'kind':'chunks','parts':[encode_tensor(t[i:i+512], importance, keep) for i in range(0,t.shape[0],512)]}
    a=t.detach().float().numpy(); shape=a.shape
    if a.ndim != 2: return {'kind':'raw','shape':shape,'data':zlib.compress(t.detach().to(torch.float16).numpy().tobytes(),9)}
    # Per-row symmetric ternary: zero, +scale, -scale. Activation-aware mask.
    score=np.abs(a) * importance[None,:] if importance is not None else np.abs(a)
    threshold=np.quantile(score,1-keep) if keep<1 else -1
    mask=score>threshold
    count=np.maximum(mask.sum(axis=1),1); scale=(np.abs(a)*mask).sum(axis=1)/count
    codes=np.where(mask,np.where(a>=0,1,2),0).astype(np.uint8).reshape(-1)
    pad=(-len(codes))%4
    if pad: codes=np.pad(codes,(0,pad))
    packed=codes[0::4] | (codes[1::4]<<2) | (codes[2::4]<<4) | (codes[3::4]<<6)
    return {'kind':'ternary','shape':shape,'scale':scale.astype(np.float16).tobytes(),'data':zlib.compress(packed.tobytes(),9)}
def decode_tensor(obj):
    if obj['kind']=='chunks': return torch.cat([decode_tensor(part) for part in obj['parts']],dim=0)
    if obj['kind']=='raw': return torch.from_numpy(np.frombuffer(zlib.decompress(obj['data']),dtype=np.float16).copy().reshape(obj['shape']))
    p=np.frombuffer(zlib.decompress(obj['data']),dtype=np.uint8); c=np.empty(p.size*4,dtype=np.uint8)
    for j in range(4): c[j::4]=(p>>(2*j))&3
    c=c[:np.prod(obj['shape'])].reshape(obj['shape']); s=np.frombuffer(obj['scale'],dtype=np.float16).astype(np.float32)[:,None]
    return torch.from_numpy((np.where(c==1,1,np.where(c==2,-1,0))*s).astype(np.float32))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--calibration',type=int,default=128); p.add_argument('--test',type=int,default=128); p.add_argument('--batch',type=int,default=4); p.add_argument('--keeps',default='0.5,0.2,0.1,0.05,0.02'); p.add_argument('--out',default='results/smollm'); args=p.parse_args()
    torch.set_num_threads(4); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval()
    cal=examples(10,args.calibration); test=examples(9001,args.test)
    print('baseline',flush=True); base,labels=evaluate(model,tok,test,args.batch); stats=calibrate(model,tok,cal,args.batch)
    baseline=sum(x==y for x,y in zip(base,labels))/len(labels); print('baseline accuracy',baseline,flush=True)
    state=model.state_dict(); denom=sum(t.numel()*2 for t in state.values()); rows=[]
    for keep in map(float,args.keeps.split(',')):
        start=time.time(); path=out/f'keep_{keep:g}.pt'; packed={}
        for name,t in state.items():
            module=name.rsplit('.',1)[0]; imp=stats.get(module); packed[name]=encode_tensor(t,imp.numpy() if imp is not None and t.ndim==2 else None,keep)
        torch.save(packed,path); del packed
        compressed=torch.load(path,weights_only=False,map_location='cpu')
        with torch.no_grad():
            for name,obj in compressed.items(): state[name].copy_(decode_tensor(obj))
        del compressed
        pred,_=evaluate(model,tok,test,args.batch); accuracy=sum(x==y for x,y in zip(pred,labels))/len(labels)
        row={'keep':keep,'artifact_bytes':path.stat().st_size,'fp16_state_bytes':denom,'ratio':path.stat().st_size/denom,'accuracy':accuracy,'seconds':round(time.time()-start,1)}; rows.append(row); print(json.dumps(row),flush=True)
        # Restore original model from disk, not the mutated state_dict references.
        del state,model
        model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.float32).eval(); state=model.state_dict()
    result={'model':MODEL,'calibration_examples':args.calibration,'test_examples':args.test,'baseline_accuracy':baseline,'results':rows,'caveat':'Synthetic diagnostic; archive excludes tokenizer, config, loader, libraries and feedback. Full vocab logits; no output-head specialization. Dense reconstruction at runtime.'}
    (out/'report.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
