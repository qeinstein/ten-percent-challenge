"""Off-distribution check (not part of the locked exam): CLINC150, a source never used for building, tuning or the exam.
Uses the feedback state learned in the final exam run (real_101) and also a fresh continual second round teaching a third label."""
import sys, json, random, collections, torch; sys.path.insert(0, ".")
from tpc import task, artifact, steer, learn
ART = sys.argv[1]
model, tok = artifact.load_model(ART)
steer.Steer.attach(model, {"labels": [], "layer": 20})
R = task.Router(model, tok)
clinc = task.read("data/owndata_clinc.jsonl")
# a third skill taught in a second round: MASSIVE "recommendation" requests (annotation pool = MASSIVE train only)
import gzip
rec_pool = [json.loads(l) for l in gzip.open(sys.argv[2] + "/massive/train/en.json.gz", "rt")]
rec_pool = [r for r in rec_pool if r["label_text"] == "recommendation"]; random.Random(11).shuffle(rec_pool)
rec_ann = [{"text": r["text"], "label": "Recommend"} for r in rec_pool[:25]]
rec_test_m = [json.loads(l) for l in gzip.open(sys.argv[2] + "/massive/test/en.json.gz", "rt")]
rec_test = [r["text"] for r in rec_test_m if r["label_text"] == "recommendation"][:90]
TXT = [r["text"] for r in clinc] + rec_test
base, H = R.generate(TXT, capture=True)
G = [r["label"] for r in clinc]
old = [i for i, g in enumerate(G) if g != "Cooking" and base[i] == g]; cook = [i for i, g in enumerate(G) if g == "Cooking"]
neutral = [json.loads(l)["text"] for l in open(f"{ART}/neutral_prompts.jsonl")]
Hn = learn.neutral_states(R, neutral, 20)
res = {"clinc_items": len(clinc), "clinc_base_acc_old_skills": sum(base[i] == G[i] for i in range(len(G)) if G[i] != "Cooking") / sum(g != "Cooking" for g in G),
       "clinc_old_base_correct": len(old), "clinc_cooking_items": len(cook)}
def score(tag, out):
    res[tag] = {"clinc_cooking_taught": sum(out[i] == "Cooking" for i in cook) / len(cook),
                "clinc_old_kept": sum(out[i] == G[i] for i in old) / len(old),
                "recommend_taught": sum(o == "Recommend" for o in out[len(clinc):]) / len(rec_test),
                "identical_to_base": sum(a == b for a, b in zip(out, base)) / len(out)}
    print(tag, res[tag], flush=True)
score("base", base)
s1 = steer.load("results/state_real_101.pt", "mps"); steer.Steer.attach(model, s1)
out, _ = task.Router(model, tok).generate_steered(TXT, base, H); score("round1_cooking_takeaway", out)
s2, _ = learn.fit(model, tok, rec_ann, Hn, verbose=False); s12 = steer.merge(s1, s2); steer.Steer.attach(model, s12)
out, _ = task.Router(model, tok).generate_steered(TXT, base, H); score("round2_plus_recommend", out)
sh = [{"text": a["text"], "label": "Recommend"} for a in random.Random(4).sample([{"text": t} for t in neutral], 25)]
s3, _ = learn.fit(model, tok, sh, Hn, verbose=False); steer.Steer.attach(model, steer.merge(s1, s3))
out, _ = task.Router(model, tok).generate_steered(TXT, base, H); score("round2_control_label_on_unrelated", out)
json.dump(res, open("results/owndata_clinc.json", "w"), indent=1)
