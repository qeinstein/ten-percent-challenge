#!/usr/bin/env python3
"""Read-only comparison of all serialized operator payloads in two packages."""
import argparse,gzip,json
from pathlib import Path
import torch

def load(path):
    with gzip.open(path,'rb') as h:return torch.load(h,map_location='cpu',weights_only=False)

def validate(a):
    rows,cols=a['shape'];group=a['group_size'];groups=(cols+group-1)//group
    bits=4 if a['kind']=='task_sparse_int4_v1' else 2
    errors=[]
    if a['codes'].numel()!=rows*groups*group*bits//8:errors.append('packed_code_length')
    if tuple(a['scales'].shape)!=(rows,groups):errors.append('scale_shape')
    if not torch.isfinite(a['scales']).all():errors.append('nonfinite_scale')
    if (a['scales']<0).any():errors.append('negative_scale')
    return errors

def main():
    p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    torch.set_num_threads(2)
    first={f.name:f for f in a.first.glob('model_layers_*.pt.gz')};second={f.name:f for f in a.second.glob('model_layers_*.pt.gz')}
    names=sorted(first.keys()|second.keys());results=[]
    for i,name in enumerate(names):
        if name not in first or name not in second:results.append({'name':name,'missing':True});continue
        x,y=load(first[name]),load(second[name]);results.append({'name':name,'first_kind':x['kind'],'second_kind':y['kind'],'first_shape':x['shape'],'second_shape':y['shape'],'first_density':x.get('density_actual'),'second_density':y.get('density_actual'),'same_codes':torch.equal(x['codes'],y['codes']),'same_scales':torch.equal(x['scales'],y['scales']),'first_errors':validate(x),'second_errors':validate(y)})
        if (i+1)%32==0:print('ARTIFACT_AUDIT',i+1,'/',len(names),flush=True)
    result={'operators':len(results),'same_codes':sum(r.get('same_codes',False) for r in results),'same_scales':sum(r.get('same_scales',False) for r in results),'with_errors':sum(bool(r.get('missing') or r.get('first_errors') or r.get('second_errors') or r.get('first_shape')!=r.get('second_shape')) for r in results),'details':results}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print('ARTIFACT_AUDIT_DONE',json.dumps({k:v for k,v in result.items() if k!='details'}),flush=True)

if __name__=='__main__':main()
