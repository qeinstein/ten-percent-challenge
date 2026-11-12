#!/usr/bin/env python3
"""Freeze content-disjoint task splits before any candidate is built.

Reads only Bobliuuu's *public* C2 calibration and development files; never his
published locked set. Our held-out examples are withheld from the GPU runner.
"""
import argparse, hashlib, json, os
from pathlib import Path
from datetime import datetime,timezone

def key(r): return hashlib.sha256(r['text'].strip().encode()).hexdigest()
def save(path,rows):
    data=b''.join((json.dumps(r,sort_keys=True,ensure_ascii=False)+'\n').encode() for r in rows); path.write_bytes(data)
    return {'rows':len(rows),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default='gpu/data_sources');p.add_argument('--output',default='gpu/prepared');a=p.parse_args()
    source=Path(a.source); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): raise SystemExit('split already frozen; refusing overwrite')
    cal=[json.loads(x) for x in (source/'calib.jsonl').read_text().splitlines() if x.strip()]
    rest=[json.loads(x) for x in (source/'dev.jsonl').read_text().splitlines() if x.strip()]
    known=set(); unique=[]
    for r in rest:
        k=key(r)
        if k not in known: unique.append(r); known.add(k)
    cal=[r for r in cal if key(r) not in known]
    buckets={'dev':[],'feedback':[],'locked':[]}
    for row in unique:
        part=int(key(row)[:8],16)%10
        buckets['locked' if part<2 else 'feedback' if part<5 else 'dev'].append(row)
    files={'calib.jsonl':save(out/'calib.jsonl',cal)}
    for name,rows in buckets.items(): files[name+'.jsonl']=save(out/(name+'.jsonl'),rows)
    manifest={'schema':'qwen_tool_splits_v1','created_utc':datetime.now(timezone.utc).isoformat(),'source':'Bobliuuu/ten-percent-challenge v2/commercial/C2 public calib+dev; revision 3dbd835b02c13fe2825e8ffe645facf3f9d8999e','locked_use':'Do not inspect or score until final selection; public-source contamination cannot be ruled out.','files':files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n'); print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
