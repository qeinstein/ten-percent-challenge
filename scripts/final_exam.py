"""FINAL, single-shot evaluation on the locked files. Verifies the lock hashes, then runs every arm once and writes results/.

Arms: base (release), real feedback x3 draws, shuffled labels, new labels on unrelated ordinary requests, empty feedback.
Feedback annotations come ONLY from data/feedback_pool.jsonl (MASSIVE train); the locked teach set is MASSIVE test + HWU64."""
import sys, os, json, random, hashlib, time, math, collections, torch; sys.path.insert(0, ".")
from tpc import task, artifact, steer, learn
ART = sys.argv[1]; DRY = os.environ.get("DRY") == "1"; OUT = "/tmp/dry_results" if DRY else "results"; os.makedirs(OUT, exist_ok=True)
man = json.load(open("exam/MANIFEST.json"))
for f, h in man.items(): assert hashlib.sha256(open(f, "rb").read()).hexdigest() == h, f"lock broken: {f}"
if DRY:   # pipeline check on NON-locked data only
    exam = task.read("data/dev.jsonl")[:40]; teach = task.read("data/feedback_pool.jsonl")[-30:]
else:
    claim = "results/.exam_claimed"
    assert not os.path.exists(claim), "the locked exam was already run once"
    open(claim, "w").write(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    exam = task.read("exam/exam.jsonl"); teach = task.read("exam/teach.jsonl")
model, tok = artifact.load_model(ART)
def wsha(): return hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
W0 = wsha()
neutral = [json.loads(l)["text"] for l in open(os.path.join(ART, "neutral_prompts.jsonl"))]
pool = task.read("data/feedback_pool.jsonl")
def draw(seed):
    r = random.Random(seed); p = pool[:]; r.shuffle(p)
    c = [x for x in p if x["skill"] == "cooking"][:13]; t = [x for x in p if x["skill"] == "takeaway"][:12]
    return [{"text": x["text"], "label": task.SKILLS[x["skill"]]} for x in c + t]
steer.Steer.attach(model, {"labels": [], "layer": 20})
R = task.Router(model, tok)
TXT = [r["text"] for r in teach] + [r["text"] for r in exam]; NT = len(teach)
t0 = time.time(); BASE, H = R.generate(TXT, capture=True); print("base pass", round(time.time()-t0), "s", flush=True)
json.dump({"teach": BASE[:NT], "exam": BASE[NT:]}, open(f"{OUT}/outputs_base.json", "w"))
Hn = learn.neutral_states(R, neutral, 20)
def run(state, tag, verify=False):
    steer.Steer.attach(model, state); R1 = task.Router(model, tok)
    out, nact = R1.generate_steered(TXT, BASE, H)
    rec = {"teach": out[:NT], "exam": out[NT:], "regenerated": nact}
    if verify:   # check the zero-gate shortcut against full generation on 300 random exam requests
        ii = random.Random(9).sample(range(NT, len(TXT)), min(300, len(TXT) - NT))
        full = R1.generate([TXT[i] for i in ii]); rec["shortcut_check"] = sum(full[k] == out[i] for k, i in enumerate(ii)) / len(ii)
        print(tag, "shortcut agreement", rec["shortcut_check"], flush=True)
    json.dump(rec, open(f"{OUT}/outputs_{tag}.json", "w")); print(tag, "regenerated", nact, flush=True)
for seed in (101, 202, 303):
    ann = draw(seed); json.dump(ann, open(f"{OUT}/ann_real_{seed}.json", "w"))
    state, _ = learn.fit(model, tok, ann, Hn, verbose=False); steer.save(state, f"{OUT}/state_real_{seed}.pt")
    run(state, f"real_{seed}", verify=(seed == 101))
ann = draw(101); labs = [a["label"] for a in ann]
shuf = [{"text": a["text"], "label": "Takeaway" if a["label"] == "Cooking" else "Cooking"} for a in ann]   # every label wrong
json.dump(shuf, open(f"{OUT}/ann_shuffled.json", "w"))
state, _ = learn.fit(model, tok, shuf, Hn, verbose=False); run(state, "shuffled")
r = random.Random(5); ordinary = r.sample(task.read("data/calib.jsonl"), 25)
unrel = [{"text": o["text"], "label": l} for o, l in zip(ordinary, labs)]      # the new labels on unrelated ordinary requests
json.dump(unrel, open(f"{OUT}/ann_unrelated.json", "w"))
state, _ = learn.fit(model, tok, unrel, Hn, verbose=False); run(state, "unrelated")
run({"labels": [], "layer": 20}, "empty")
assert wsha() == W0, "weights changed"
json.dump({"weights_sha_before": W0, "weights_sha_after": wsha()}, open(f"{OUT}/weights_hash.json", "w"))
print("done; score with scripts/score.py")
