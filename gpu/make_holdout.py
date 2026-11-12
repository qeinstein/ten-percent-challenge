#!/usr/bin/env python3
"""Generate our separate, previously unused wording split after freezing a candidate."""
import argparse, hashlib, json
from pathlib import Path
from my_tools import make
p=argparse.ArgumentParser(); p.add_argument('--output',required=True,type=Path); p.add_argument('--count',type=int,default=320); p.add_argument('--seed',type=int,default=73421); a=p.parse_args()
if a.output.exists(): raise SystemExit('Refusing to overwrite existing holdout')
a.output.parent.mkdir(parents=True,exist_ok=True)
data=''.join(json.dumps(x,sort_keys=True)+'\n' for x in make('heldout',a.count,a.seed)).encode(); a.output.write_bytes(data)
print(json.dumps({'path':str(a.output),'count':a.count,'sha256':hashlib.sha256(data).hexdigest()}))
