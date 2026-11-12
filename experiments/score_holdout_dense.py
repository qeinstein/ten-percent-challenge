import json, torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from tool_prune import batched
from tool_holdout import holdout
model_id='HuggingFaceTB/SmolLM2-135M-Instruct'; revision='12fd25f77366fa6b3b4b768ec3050bf629380bac'
torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True); tok.pad_token=tok.eos_token; tok.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,dtype=torch.float32).eval()
ids=[tok.encode(' '+c,add_special_tokens=False)[0] for c in 'ABCD']; h=m.lm_head.weight.detach()[ids].float(); samples=holdout(); correct=0
with torch.inference_mode():
 for enc,y in batched(tok,samples,4):
  x=m.model(**enc).last_hidden_state[:,-1,:]; correct+=int(((x@h.T).argmax(-1)==y).sum())
print(json.dumps({'correct':correct,'total':len(samples),'accuracy':correct/len(samples)}))
