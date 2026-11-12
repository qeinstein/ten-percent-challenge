#!/usr/bin/env python3
"""Reload and validate a complete NEC package without the base checkpoint."""
import argparse, gzip, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from nox.packed_schmidt import reconstruct_packed_schmidt
from tool_prune import examples, batched
from tool_holdout import holdout

def read(path):
    with gzip.open(path,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)

def main():
    p=argparse.ArgumentParser(); p.add_argument('package',type=Path); p.add_argument('--test',type=int,default=64); p.add_argument('--holdout',action='store_true'); args=p.parse_args()
    torch.set_num_threads(4); root=args.package
    cfg=AutoConfig.from_pretrained(root/'config',local_files_only=True)
    model=AutoModelForCausalLM.from_config(cfg).eval()
    retained=read(root/'retained.pt.gz'); emb=retained['embedding']; q=emb['values']; a=torch.empty((q.shape[0],q.shape[1]*2),dtype=torch.int16); a[:,::2]=q&15; a[:,1::2]=q>>4
    weights=dict(model.named_parameters()); handled=set()
    with torch.no_grad():
        weights['model.embed_tokens.weight'].copy_((a[:,:emb['shape'][1]].float()-8)*emb['scale'].float()[:,None]); handled.add('model.embed_tokens.weight')
        for name,value in retained['norms'].items(): weights[name].copy_(value); handled.add(name)
        for name,mod in model.named_modules():
            if name.startswith('model.layers.') and isinstance(mod,torch.nn.Linear):
                key=name+'.weight'; f=root/(name.replace('.','_')+'.pt.gz')
                if not f.exists(): raise ValueError('missing projection '+key)
                weights[key].copy_(reconstruct_packed_schmidt(read(f))); handled.add(key)
    missing=set(weights)-handled
    if missing: raise ValueError('package missing weights: '+str(sorted(missing)))
    tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'; head=retained['head'].float(); correct=0
    samples=holdout() if args.holdout else examples(9001,args.test)
    with torch.inference_mode():
        for enc,y in batched(tok,samples,4):
            h=model.model(**enc).last_hidden_state[:,-1,:]; correct+=int(((h.float()@head.T).argmax(-1)==y).sum())
    print(json.dumps({'correct':correct,'total':len(samples),'accuracy':correct/len(samples),'parameters_loaded':len(handled)}))
if __name__=='__main__': main()
