#!/usr/bin/env python3
"""Compare one saved layer's decoded matrices to the pinned checkpoint."""
import argparse,gzip,json
from pathlib import Path
import torch
from safetensors import safe_open
from weightspace import reconstruct_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True,type=Path);p.add_argument('--package',required=True,type=Path);p.add_argument('--layer',type=int,default=0);p.add_argument('--output',required=True,type=Path);a=p.parse_args();torch.set_num_threads(2)
    index=json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map'];out=[]
    for suffix in ('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj'):
        name=f'model.layers.{a.layer}.{suffix}';key=name+'.weight'
        with safe_open(a.snapshot/index[key],framework='pt',device='cpu') as f:original=f.get_tensor(key).float()
        with gzip.open(a.package/(name.replace('.','_')+'.pt.gz'),'rb') as f:artifact=torch.load(f,map_location='cpu',weights_only=False)
        decoded=reconstruct_matrix(artifact);norm=original.norm().clamp_min(1e-30);decoded_norm=decoded.norm()
        result={'name':name,'shape':list(original.shape),'retained_fraction':float((decoded!=0).float().mean()),'relative_frobenius_error':float((original-decoded).norm()/norm),'norm_ratio':float(decoded_norm/norm),'cosine_similarity':float((original*decoded).sum()/(norm*decoded_norm).clamp_min(1e-30)),'sign_disagreements':int(((original*decoded)<0).sum()),'zero_rows':int((decoded==0).all(dim=1).sum()),'finite':bool(torch.isfinite(decoded).all())}
        out.append(result);print('PROJECTION_AUDIT',json.dumps(result),flush=True);del original,decoded,artifact
    a.output.write_text(json.dumps({'layer':a.layer,'operators':out},indent=2)+'\n');print('PROJECTION_AUDIT_DONE',flush=True)
if __name__=='__main__':main()
