#!/usr/bin/env python3
"""Audit a saved sparse V projection against dense and dense-int4 controls."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
import qwen_nec as qn
from weightspace import compile_matrix,reconstruct_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--dev',type=Path,required=True);p.add_argument('--calib',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=64);a=p.parse_args()
    qn.DEVICE='cuda';torch.set_grad_enabled(False);start=time.monotonic()
    rows=[json.loads(x) for x in a.dev.read_text().splitlines() if x.strip()][:a.limit]
    cal=[json.loads(x) for x in a.calib.read_text().splitlines() if x.strip()][:64]
    tok=AutoTokenizer.from_pretrained(qn.MODEL,revision=qn.REVISION,local_files_only=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left'
    model=AutoModelForCausalLM.from_pretrained(qn.MODEL,revision=qn.REVISION,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa',local_files_only=True).eval();model.config.use_cache=False
    ids=[tok.encode(' '+c,add_special_tokens=False)[0] for c in qn.LETTERS];head=model.lm_head.weight.detach()[ids].float().clone();model.lm_head=torch.nn.Identity()
    saved=qn.read_gz(a.package/'retained.pt.gz');qn.apply_embedding(model,saved)
    fallback=tok.unk_token_id if tok.unk_token_id is not None else tok.eos_token_id;mapping,_=qn.remapper(model.model.embed_tokens.num_embeddings,saved['ids'],fallback,'cuda')
    name='model.layers.0.self_attn.v_proj';mod=dict(model.named_modules())[name];original=mod.weight.detach().clone()
    stats=qn.capture(model,tok,cal,mapping,32,512,name_prefix=name)[name]
    artifact=qn.read_gz(a.package/(name.replace('.','_')+'.pt.gz'))
    sparse=reconstruct_matrix(artifact,device='cuda').to(mod.weight.dtype)
    dense4=reconstruct_matrix(compile_matrix(original,stats,density=1.,group_size=128,quantization='int4'),device='cuda').to(mod.weight.dtype)
    # Measure value-projection signals on real calibration tokens. The first
    # layer's input is unchanged when swapping only this projection.
    samples=[];positions=None
    def capture_input(_module,args):
        x=args[0].detach();index=positions.unsqueeze(-1).expand(-1,-1,x.shape[-1])
        samples.append(x.gather(1,index))
    hook=mod.register_forward_pre_hook(capture_input)
    try:
        with torch.inference_mode():
            for off in range(0,len(cal),4):
                group=cal[off:off+4];enc=tok([qn.prompt(r) for r in group],return_tensors='pt',padding=True,truncation=True,max_length=512)
                ids=enc.input_ids.to('cuda');attention=enc.attention_mask.to('cuda');lengths=attention.sum(-1);pad_start=ids.shape[1]-lengths
                positions=pad_start[:,None]+((lengths[:,None]-1)*torch.linspace(0,1,32,device='cuda')).long()
                model.model(input_ids=mapping[ids],attention_mask=attention,use_cache=False)
    finally:hook.remove()
    x=torch.cat(samples).reshape(-1,original.shape[1]);dense_signal=torch.nn.functional.linear(x,original).float();sparse_signal=torch.nn.functional.linear(x,sparse).float()
    bias=mod.bias.detach().float() if mod.bias is not None else torch.zeros(dense_signal.shape[-1],device='cuda')
    value_stats={'tokens':x.shape[0],'bias_l2':float(bias.norm()),'dense_signal_mean_l2':float(dense_signal.norm(dim=-1).mean()),'sparse_signal_mean_l2':float(sparse_signal.norm(dim=-1).mean()),'relative_signal_error':float((dense_signal-sparse_signal).norm()/dense_signal.norm().clamp_min(1e-30)),'dense_signal_variation_l2':float((dense_signal-dense_signal.mean(0)).square().mean(0).sum().sqrt()),'sparse_signal_variation_l2':float((sparse_signal-sparse_signal.mean(0)).square().mean(0).sum().sqrt())}
    print('VALUE_SIGNAL',json.dumps(value_stats),flush=True);del x,dense_signal,sparse_signal,samples
    records=[]
    def measure(label,weight):
        mod.weight.copy_(weight)
        result=qn.score(model,tok,rows,head,mapping,max_length=512,return_predictions=True)
        result['prediction_counts']=[result['predictions'].count(i) for i in range(5)];result['condition']=label;records.append(result)
        print('VALUE_AUDIT',label,result['correct'],'/',result['total'],result['prediction_counts'],flush=True)
    measure('dense_v',original);measure('saved_sparse_v',sparse);measure('dense_int4_v',dense4);measure('zero_v_weight',torch.zeros_like(original));measure('restored_dense_v',original)
    result={'status':'complete','model':qn.MODEL,'revision':qn.REVISION,'package':str(a.package),'calibration_examples':len(cal),'examples':len(rows),'value_signal':value_stats,'dense_restore_predictions_match':records[0]['predictions']==records[-1]['predictions'],'stages':records,'elapsed_seconds':time.monotonic()-start}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('VALUE_AUDIT_DONE',flush=True)
if __name__=='__main__':main()
