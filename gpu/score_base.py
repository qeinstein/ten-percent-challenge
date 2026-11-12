#!/usr/bin/env python3
"""Score uncompressed Qwen on the same saved examples as a packed package."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from qwen_nec import MODEL,REVISION,score
p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--limit',type=int,default=320); p.add_argument('--predictions',action='store_true'); p.add_argument('--max-length',type=int,default=512); a=p.parse_args()
if not torch.cuda.is_available(): raise SystemExit('CUDA is required')
tok=AutoTokenizer.from_pretrained(MODEL,revision=REVISION); tok.pad_token=tok.pad_token or tok.eos_token; tok.padding_side='left'
model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
ids=[tok.encode(' '+c,add_special_tokens=False)[0] for c in 'ABCDE']; head=model.lm_head.weight.detach()[ids].float()
rows=[json.loads(x) for x in a.input.read_text().splitlines() if x.strip()][:a.limit]
print(json.dumps(score(model,tok,rows,head,max_length=a.max_length,return_predictions=a.predictions)))
