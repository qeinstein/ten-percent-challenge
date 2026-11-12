#!/usr/bin/env python3
"""Install a saved candidate into dense Qwen a layer at a time on dev data."""
import argparse,collections,gzip,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import qwen_nec as qn
from weightspace import reconstruct_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',required=True,type=Path);p.add_argument('--dev',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--limit',type=int,default=64);p.add_argument('--max-layers',type=int,default=0);p.add_argument('--per-operator',action='store_true');p.add_argument('--independent-operators',action='store_true');a=p.parse_args()
    qn.DEVICE='cuda';torch.set_grad_enabled(False);start=time.monotonic()
    rows=[json.loads(x) for x in a.dev.read_text().splitlines() if x.strip()][:a.limit]
    if not rows:raise ValueError('empty development data')
    tok=AutoTokenizer.from_pretrained(qn.MODEL,revision=qn.REVISION,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(qn.MODEL,revision=qn.REVISION,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.config.use_cache=False
    ids=[tok.encode(' '+c,add_special_tokens=False)[0] for c in qn.LETTERS]
    head=model.lm_head.weight.detach()[ids].float().clone();model.lm_head=torch.nn.Identity()
    saved=qn.read_gz(a.package/'retained.pt.gz'); head_diff=float((head.cpu()-saved['head'].float()).abs().max())
    other=dict(model.named_parameters()); mismatches=[]
    for name,val in saved['other'].items():
        if name not in other or not torch.equal(other[name].detach().cpu(),val.to(other[name].dtype)):mismatches.append(name)
    keep=saved['ids'];fallback=tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id
    mapping,_=qn.remapper(model.model.embed_tokens.num_embeddings,keep,fallback,'cuda')
    records=[]
    def measure(stage):
        result=qn.score(model,tok,rows,head,mapping,max_length=512,return_predictions=True)
        predictions=result.pop('predictions');result['prediction_counts']=[predictions.count(i) for i in range(5)];result['stage']=stage
        records.append(result);a.output.write_text(json.dumps({'head_max_abs_difference':head_diff,'other_parameter_mismatches':mismatches,'stages':records},indent=2)+'\n')
        print('AUDIT_STAGE',json.dumps(result),flush=True)
        return predictions
    dense=measure('dense')
    qn.apply_embedding(model,saved);embedded=measure('quantized_embedding')
    single=qn.score(model,tok,rows[:4],head,mapping,max_length=512,batch_size=1,return_predictions=True)['predictions']
    batched=qn.score(model,tok,rows[:4],head,mapping,max_length=512,batch_size=4,return_predictions=True)['predictions']
    modules=dict(model.named_modules());count=0
    for layer in range(min(a.max_layers,model.config.num_hidden_layers) if a.max_layers else model.config.num_hidden_layers):
        prefix=f'model.layers.{layer}.'
        names=[n for n,m in modules.items() if n.startswith(prefix) and isinstance(m,torch.nn.Linear)]
        for name in names:
            artifact=qn.read_gz(a.package/(name.replace('.','_')+'.pt.gz'))
            if tuple(artifact['shape'])!=tuple(modules[name].weight.shape):raise ValueError('shape mismatch: '+name)
            original=modules[name].weight.detach().clone() if a.independent_operators else None
            modules[name].weight.copy_(reconstruct_matrix(artifact,device='cuda').to(modules[name].weight.dtype));count+=1
            if a.per_operator or a.independent_operators: measure(('only_' if a.independent_operators else '')+'compressed_'+name)
            if original is not None:modules[name].weight.copy_(original)
        if not a.independent_operators:measure(f'compressed_through_layer_{layer}')
    result={'status':'complete','model':qn.MODEL,'revision':qn.REVISION,'package':str(a.package),'examples':len(rows),'head_max_abs_difference':head_diff,'other_parameter_mismatches':mismatches,'single_vs_batch_predictions_equal':single==batched,'operators_loaded':count,'stages':records,'elapsed_seconds':time.monotonic()-start}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('AUDIT_COMPLETE',json.dumps(result),flush=True)

if __name__=='__main__':main()
