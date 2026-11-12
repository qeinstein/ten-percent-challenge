"""Build the shipped artifact directory: packed 1-bit weights + tokenizer/config + code + task definition + unlabeled neutral prompts."""
import sys, os, json, shutil, subprocess, hashlib, pathlib; sys.path.insert(0, ".")
from tpc import task
SRC = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2])
if not (OUT/"bits.bin").exists():
    subprocess.check_call([sys.executable, "scripts/pack.py", str(SRC/"model.safetensors"), str(OUT)])
for f in ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "LICENSE", "NOTICE.txt"]:
    shutil.copy2(SRC/f, OUT/f)
spec = {"instruction": task.HEAD.rstrip(), "fewshot": task.FEWSHOT, "request_tag": "Request", "answer_tag": "Skill"}
json.dump(spec, open(OUT/"task_default.json", "w"), indent=1)
assert task.load_spec(str(OUT/"task_default.json")).prefix == task.PREFIX
with open(OUT/"neutral_prompts.jsonl", "w") as f:
    for r in task.read("data/calib.jsonl"): f.write(json.dumps({"text": r["text"]}) + "\n")   # labels dropped
code = OUT/"code"; shutil.rmtree(code, ignore_errors=True); (code/"tpc").mkdir(parents=True)
for f in ["answer.py", "feedback.py"]: shutil.copy2(f, code/f)
for f in ["__init__.py", "task.py", "artifact.py", "steer.py", "learn.py", "device.py"]: shutil.copy2("tpc/"+f, code/"tpc"/f)
tot = 0; rows = []
for p in sorted(OUT.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts: rows.append((str(p.relative_to(OUT)), p.stat().st_size)); tot += p.stat().st_size
den = sum(os.path.getsize(SRC/f) for f in os.listdir(SRC) if f.endswith(".safetensors"))
json.dump({"files": rows, "total_bytes": tot, "denominator_fp16_bytes": den, "ratio": tot/den,
           "sha256": {n: hashlib.sha256(open(OUT/n, "rb").read()).hexdigest() for n, _ in rows}}, open("SIZE.json", "w"), indent=1)
print(f"total {tot} B / fp16 {den} B = {100*tot/den:.3f}%")
