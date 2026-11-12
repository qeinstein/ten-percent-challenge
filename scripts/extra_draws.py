"""More real-feedback draws on the locked exam with the FROZEN settings (no retuning; every draw is reported).
Base outputs come from the original single run (results/outputs_base.json); the base pass is redone only to get layer-20 states,
and its outputs are checked to be identical to the stored ones."""
import sys, os, json, random, hashlib, time, torch; sys.path.insert(0, ".")
from tpc import task, artifact, steer, learn
ART = sys.argv[1]; OUT = "results"
man = json.load(open("exam/MANIFEST.json"))
for f, h in man.items(): assert hashlib.sha256(open(f, "rb").read()).hexdigest() == h
exam = task.read("exam/exam.jsonl"); teach = task.read("exam/teach.jsonl")
model, tok = artifact.load_model(ART); steer.Steer.attach(model, {"labels": [], "layer": 20}); R = task.Router(model, tok)
TXT = [r["text"] for r in teach] + [r["text"] for r in exam]; NT = len(teach)
BASE, H = R.generate(TXT, capture=True)
old = json.load(open(f"{OUT}/outputs_base.json")); same = sum(a == b for a, b in zip(BASE, old["teach"] + old["exam"])) / len(TXT)
print("base outputs identical to the original run:", same, flush=True)
BASE = old["teach"] + old["exam"]
neutral = [json.loads(l)["text"] for l in open(os.path.join(ART, "neutral_prompts.jsonl"))]
Hn = learn.neutral_states(R, neutral, 20)
pool = task.read("data/feedback_pool.jsonl")
for seed in (404, 505, 606, 707, 808):
    r = random.Random(seed); p = pool[:]; r.shuffle(p)
    ann = [{"text": x["text"], "label": task.SKILLS[x["skill"]]} for x in [x for x in p if x["skill"] == "cooking"][:13] + [x for x in p if x["skill"] == "takeaway"][:12]]
    json.dump(ann, open(f"{OUT}/ann_real_{seed}.json", "w"))
    state, _ = learn.fit(model, tok, ann, Hn, verbose=False); steer.save(state, f"{OUT}/state_real_{seed}.pt")
    steer.Steer.attach(model, state); out, n = task.Router(model, tok).generate_steered(TXT, BASE, H)
    json.dump({"teach": out[:NT], "exam": out[NT:], "regenerated": n}, open(f"{OUT}/outputs_real_{seed}.json", "w")); print(seed, "done", flush=True)
