#!/usr/bin/env python3
"""Three-hour-budget orchestration. Does not shut down provider billing."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def run_group(cmd,timeout):
    p=subprocess.Popen(cmd,start_new_session=True)
    try:
        rc=p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
    if rc:raise subprocess.CalledProcessError(rc,cmd)

p=argparse.ArgumentParser();p.add_argument('--output',default='/workspace/qwen-direct');p.add_argument('--data-dir',default='gpu/prepared');p.add_argument('--total-seconds',type=int,default=10500);p.add_argument('--reserve-seconds',type=int,default=3000);p.add_argument('--density',type=float,default=.02);a=p.parse_args()
if a.reserve_seconds>=a.total_seconds:raise SystemExit('reserve must be below total budget')
start=time.monotonic(); deadline=start+a.total_seconds; model_time=a.total_seconds-a.reserve_seconds
cmd=[sys.executable,'gpu/qwen_nec.py','--output',a.output,'--data-dir',a.data_dir,'--density',str(a.density),'--limit-seconds',str(model_time)]
try:
    run_group(cmd,model_time+60)
    report=Path(a.output)/'report.json'; status=Path(a.output)/'status.json'
    if not report.exists() or not status.exists() or json.loads(status.read_text()).get('status')!='complete':raise SystemExit('Compression incomplete; final exam remains sealed.')
    remaining=int(deadline-time.monotonic())
    if remaining<=0:raise SystemExit('No time remains; final exam remains sealed.')
    run_group([sys.executable,'gpu/final_exam.py','--output',a.output,'--data-dir',a.data_dir,'--timeout-seconds',str(remaining)],remaining)
except subprocess.TimeoutExpired:
    raise SystemExit('Time budget reached; terminate the GPU instance to stop billing.')
