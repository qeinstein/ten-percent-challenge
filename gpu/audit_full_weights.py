#!/usr/bin/env python3
"""Stream and compare every saved operator with the pinned base checkpoint."""
import argparse,gzip,json,re,time
from pathlib import Path
import torch
from safetensors import safe_open
from weightspace import reconstruct_matrix

def main():
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True,type=Path);p.add_argument('--package',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    torch.set_num_threads(2);started=time.monotonic();index=json.loads((a.snapshot/'model.safetensors.index.json').read_text())['weight_map'];files=sorted(a.package.glob('model_layers_*.pt.gz'));rows=[]
    cfg=json.loads((a.package/'config'/'config.json').read_text());expected=cfg['num_hidden_layers']*7
    if len(files)!=expected:raise ValueError(f'expected {expected} operator files, found {len(files)}')
    pattern=re.compile(r'^model_layers_(\d+)_(self_attn|mlp)_(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.pt\.gz$')
    for i,path in enumerate(files):
        match=pattern.match(path.name)
        if not match:raise ValueError('unexpected artifact '+path.name)
        layer,branch,projection=match.groups();name=f'model.layers.{layer}.{branch}.{projection}';key=name+'.weight'
        with safe_open(a.snapshot/index[key],framework='pt',device='cpu') as f:original=f.get_tensor(key).to('cuda',dtype=torch.float32)
        with gzip.open(path,'rb') as h:artifact=torch.load(h,map_location='cpu',weights_only=False)
        if tuple(artifact['shape'])!=tuple(original.shape):raise ValueError('shape mismatch '+name)
        decoded=reconstruct_matrix(artifact,device='cuda');norm=original.norm().clamp_min(1e-30);dnorm=decoded.norm()
        result={'name':name,'retained_fraction':float((decoded!=0).float().mean()),'relative_frobenius_error':float((original-decoded).norm()/norm),'norm_ratio':float(dnorm/norm),'sign_disagreements':int(((original*decoded)<0).sum()),'zero_rows':int((decoded==0).all(dim=1).sum()),'finite':bool(torch.isfinite(decoded).all())}
        rows.append(result)
        if (i+1)%16==0:print('FULL_WEIGHT_AUDIT',i+1,'/',len(files),flush=True)
        del original,decoded,artifact
    output={'operators':len(rows),'nonfinite':sum(not r['finite'] for r in rows),'sign_disagreements':sum(r['sign_disagreements'] for r in rows),'zero_rows':sum(r['zero_rows'] for r in rows),'min_error':min(r['relative_frobenius_error'] for r in rows),'max_error':max(r['relative_frobenius_error'] for r in rows),'elapsed_seconds':time.monotonic()-started,'details':rows}
    a.output.write_text(json.dumps(output,indent=2)+'\n');print('FULL_WEIGHT_AUDIT_DONE',json.dumps({k:v for k,v in output.items() if k!='details'}),flush=True)
if __name__=='__main__':main()
