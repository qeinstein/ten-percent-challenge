#!/usr/bin/env python3
"""Reconstruct a complete compressed Qwen package without the original model weights."""
import argparse, gzip, json, sys
from pathlib import Path
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from weightspace import reconstruct_matrix
from feedback import Feedback

def load(path):
    with gzip.open(path,'rb') as f: return torch.load(f,map_location='cpu',weights_only=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('package',type=Path); ap.add_argument('--dev-jsonl',type=Path,required=True); ap.add_argument('--limit',type=int,default=128); ap.add_argument('--predictions',action='store_true'); ap.add_argument('--compare-feedback',action='store_true'); ap.add_argument('--feedback-file',type=Path); ap.add_argument('--max-length',type=int,default=512); ap.add_argument('--device',choices=('cpu','cuda'),default='cuda'); args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA is required')
    if args.device=='cpu': torch.set_num_threads(4)
    device=args.device; dtype=torch.float32 if device=='cpu' else torch.bfloat16
    if args.compare_feedback and not args.feedback_file: raise SystemExit('--compare-feedback requires --feedback-file')
    root=args.package; cfg=AutoConfig.from_pretrained(root/'config',local_files_only=True)
    with torch.device('meta'): model=AutoModelForCausalLM.from_config(cfg,dtype=dtype)
    # Inference uses only the five stored head rows, never the full output head.
    model.lm_head=torch.nn.Identity(); model.to_empty(device=device); model.eval(); model.config.use_cache=False
    # Rebuild nonpersistent rotary frequencies explicitly after to_empty.
    rotary=Qwen2RotaryEmbedding if cfg.model_type=='qwen2' else LlamaRotaryEmbedding if cfg.model_type=='llama' else None
    if rotary is None: raise ValueError('unsupported model architecture '+cfg.model_type)
    model.model.rotary_emb=rotary(config=cfg,device=device)
    saved=load(root/'retained.pt.gz'); params=dict(model.named_parameters()); seen=set()
    with torch.no_grad():
        emb=params['model.embed_tokens.weight']; emb.zero_(); ids=saved['ids']; q=saved['values']; scales=saved['scales']
        for i in range(0,len(ids),512):
            x=q[i:i+512].to(device); unpacked=torch.empty((len(x),x.shape[1]*2),dtype=torch.int16,device=device); unpacked[:,::2]=x&15; unpacked[:,1::2]=x>>4
            row=(unpacked[:,:saved['hidden_size']].float()-8)*scales[i:i+512].to(device).float()[:,None]; emb.index_copy_(0,ids[i:i+512].to(device),row.to(emb.dtype))
        seen.add('model.embed_tokens.weight')
        for name,val in saved['other'].items():
            if name not in params: raise ValueError('unknown parameter '+name)
            params[name].copy_(val.to(device)); seen.add(name)
        for name,mod in model.named_modules():
            if name.startswith('model.layers.') and isinstance(mod,torch.nn.Linear):
                key=name+'.weight'; path=root/(name.replace('.','_')+'.pt.gz')
                if not path.exists(): raise ValueError('missing operator '+name)
                mod.weight.copy_(reconstruct_matrix(load(path),device=device).to(mod.weight.dtype)); seen.add(key)
        missing=set(params)-seen
        if missing: raise ValueError('uninitialized model parameters: '+str(sorted(missing)))
    tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
    fallback=tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
    mask=torch.zeros(cfg.vocab_size,dtype=torch.bool,device=device); mask[ids.to(device)]=True
    mapping=torch.arange(cfg.vocab_size,device=device); mapping[~mask]=fallback
    head=saved['head'].float().to(device); feedback=Feedback.load(args.feedback_file) if args.feedback_file else None; rows=[json.loads(x) for x in args.dev_jsonl.read_text().splitlines() if x.strip()][:args.limit]; correct=oov=tokens=0; predictions=[]; before_correct=0; before_predictions=[]
    with torch.inference_mode():
        for offset in range(0,len(rows),4):
            group=rows[offset:offset+4]
            prompts=[r['inst']+'\n\nText: '+r['text']+'\n\nOptions:\n'+''.join(f'{l}. {c}\n' for l,c in zip('ABCDE',r['names']))+'Answer:' for r in group]
            enc=tok(prompts,return_tensors='pt',padding=True,truncation=True,max_length=args.max_length)
            x=enc.input_ids.to(device); attention=enc.attention_mask.to(device); y=mapping[x]
            oov+=int(((x!=y)*attention.bool()).sum()); tokens+=int(attention.sum())
            h=model.model(input_ids=y,attention_mask=attention,use_cache=False).last_hidden_state[:,-1,:]; logits=h.float()@head.T
            for i,r in enumerate(group):
                before=int(logits[i].argmax()); before_predictions.append(before); before_correct+=int(before==r['answer'])
                scores=feedback.adjust(logits[i],h[i],r['names']) if feedback is not None else logits[i]
                answer=int(scores.argmax()); predictions.append(answer); correct+=int(answer==r['answer'])
    result={'correct':correct,'total':len(rows),'accuracy':correct/len(rows),'oov_tokens':oov,'input_tokens':tokens,'parameters_loaded':len(seen)}
    if args.predictions: result['predictions']=predictions
    if args.compare_feedback:
        result['before_feedback']={'correct':before_correct,'total':len(rows),'accuracy':before_correct/len(rows)}
        if args.predictions: result['before_feedback']['predictions']=before_predictions
    print(json.dumps(result))
if __name__=='__main__': main()
