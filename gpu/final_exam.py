#!/usr/bin/env python3
"""One-use final evaluation; consumes the frozen held-out split only at the end."""
import argparse,hashlib,json,os,random,subprocess,sys,time
from pathlib import Path

def treehash(root):
    digest=hashlib.sha256()
    for f in sorted(p for p in root.rglob('*') if p.is_file()):
        digest.update(str(f.relative_to(root)).encode()+b'\0')
        with f.open('rb') as h:
            for block in iter(lambda:h.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def execute(cmd,timeout):
    if timeout<=0: raise TimeoutError('overall time budget expired')
    p=subprocess.run(cmd,check=True,text=True,stdout=subprocess.PIPE,timeout=timeout,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    for line in reversed(p.stdout.splitlines()):
        try:return json.loads(line)
        except json.JSONDecodeError:pass
    raise RuntimeError('evaluation did not return JSON')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True,type=Path);ap.add_argument('--data-dir',default='gpu/prepared',type=Path);ap.add_argument('--timeout-seconds',type=int,default=2400);a=ap.parse_args()
    root=a.output; package=root/'package'; report=json.loads((root/'report.json').read_text()); f=report['feedback']; baseline=report['baseline_dev']['accuracy']; compressed=report['compressed_dev']['accuracy']
    # Never expose the test to candidates that already fail development gates.
    actual_bytes=sum(f.stat().st_size for f in package.rglob('*') if f.is_file())
    gates={'package_size_unchanged':actual_bytes==report['artifact_bytes'],'size_at_most_10_percent':report['artifact_ratio']<=.1,'dev_at_least_base':compressed>=baseline,'feedback_generalizes':f['annotations']>0 and f['unseen_new_before']['total']>0 and f['unseen_new_after']['correct']>f['unseen_new_before']['correct'],'feedback_retains_dev':f['old_dev_after']['accuracy']>=baseline}
    if not all(gates.values()):
        outcome={'status':'development_gates_failed','gates':gates,'exam_used':False}
        (root/'gate_report.json').write_text(json.dumps(outcome,indent=2)+'\n'); print(json.dumps(outcome)); return
    manifest=json.loads((a.data_dir/'manifest.json').read_text()); test=a.data_dir/'locked.jsonl'
    sha=hashlib.sha256(test.read_bytes()).hexdigest()
    if sha!=manifest['files']['locked.jsonl']['sha256']: raise RuntimeError('locked dataset hash mismatch')
    count=manifest['files']['locked.jsonl']['rows']; frozen=treehash(package)
    claim=root/'FINAL_EXAM_CLAIMED.json'
    # Exclusive creation forbids accidental repeated locked-set selection.
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as h:json.dump({'locked_sha256':sha,'package_sha256':frozen,'started_unix':time.time()},h)
    deadline=time.monotonic()+a.timeout_seconds; py=sys.executable; reload=package/'reload_qwen.py'
    base=execute([py,'gpu/score_base.py','--input',str(test),'--limit',str(count),'--predictions'],deadline-time.monotonic())
    post=execute([py,str(reload),str(package),'--dev-jsonl',str(test),'--limit',str(count),'--predictions','--compare-feedback','--feedback-file',str(package/'feedback.pt')],deadline-time.monotonic())
    pre=post.pop('before_feedback')
    if treehash(package)!=frozen: raise RuntimeError('package changed during final evaluation')
    labels=[int(json.loads(line)['answer']) for line in test.read_text().splitlines() if line.strip()]
    def interval(candidate):
        d=[int(x==y)-int(z==y) for x,z,y in zip(candidate['predictions'],base['predictions'],labels)]
        if len(d)!=count: raise RuntimeError('prediction length mismatch')
        rng=random.Random(0); samples=sorted(sum(rng.choices(d,k=count))/count for _ in range(3000))
        return {'difference':sum(d)/count,'bootstrap_95_ci':[samples[75],samples[2924]]}
    paired={'before':interval(pre),'after':interval(post)}
    result={'status':'evaluated','locked_sha256':sha,'package_sha256':frozen,'baseline':base,'compressed_before_feedback':pre,'compressed_after_feedback':post,'paired_accuracy_difference':paired,'passes_point_estimate':pre['accuracy']>=base['accuracy'] and post['accuracy']>=base['accuracy'],'limitation':'Paired bootstrap intervals are descriptive; public-source contamination and organizer rulings remain.'}
    (root/'final_exam.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
