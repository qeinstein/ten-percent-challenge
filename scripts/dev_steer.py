"""Dev loop for weight-free feedback: 25 annotations from the MASSIVE-train pool; teach on held-out pool items; neighbours on dev."""
import sys, json, random, time, collections, argparse, os, hashlib, torch; sys.path.insert(0,".")
from tpc import task, artifact, learn
ap = argparse.ArgumentParser(); ap.add_argument("--art", default="/tmp/art_test")
ap.add_argument("--layer", type=int, default=20); ap.add_argument("--margin", type=float, default=4.0)
ap.add_argument("--q", type=float, default=0.995); ap.add_argument("--pf", type=float, default=0.3); ap.add_argument("--temp", type=float, default=0.02)
ap.add_argument("--shuffle", action="store_true"); ap.add_argument("--ndev", type=int, default=300); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
model, tok = artifact.load_model(a.art)
w0 = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
rng = random.Random(a.seed)
pool = task.read("data/feedback_pool.jsonl"); rng.shuffle(pool)
cook = [r for r in pool if r["skill"]=="cooking"]; take = [r for r in pool if r["skill"]=="takeaway"]
ann = cook[:13] + take[:12]; test = cook[13:63] + take[12:62]
dev = task.read("data/dev.jsonl"); random.Random(1).shuffle(dev); dev = dev[:a.ndev]
calib = [r["text"] for r in task.read("data/calib.jsonl")]
R0 = task.Router(model, tok)
cf=f"/tmp/base_cache_{a.ndev}_{a.seed}.json"
if os.path.exists(cf): base_dev, base_test = json.load(open(cf))
else:
    base_dev = R0.generate([r["text"] for r in dev]); base_test = R0.generate([r["text"] for r in test]); json.dump([base_dev, base_test], open(cf,"w"))
labs = [task.SKILLS[r["skill"]] for r in ann]
if a.shuffle: labs = labs[:]; random.Random(3).shuffle(labs)
annot = [{"text": r["text"], "label": l} for r, l in zip(ann, labs)]
t = time.time(); state, st = learn.fit(model, tok, annot, calib, layer=a.layer, margin=a.margin, q_neutral=a.q, pos_frac=a.pf, temp=a.temp); print("feedback time", round(time.time()-t), flush=True)
R1 = task.Router(model, tok)
fb_test = R1.generate([r["text"] for r in test]); fb_dev = R1.generate([r["text"] for r in dev])
w1 = hashlib.sha256(b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())).hexdigest()
print("weights identical:", w0 == w1)
gold_t = [task.SKILLS[r["skill"]] for r in test]
print("TEACH base", sum(x==g for x,g in zip(base_test,gold_t))/len(test), "after", sum(x==g for x,g in zip(fb_test,gold_t))/len(test))
for sk in ("Cooking","Takeaway"):
    ids=[i for i,g in enumerate(gold_t) if g==sk]; print("  ",sk, sum(fb_test[i]==sk for i in ids)/len(ids))
gold_d = [task.SKILLS[r["skill"]] for r in dev]
ok = [i for i in range(len(dev)) if base_dev[i]==gold_d[i]]
print("DEV base-correct kept", round(sum(fb_dev[i]==gold_d[i] for i in ok)/len(ok),4), "of", len(ok), "| identical outputs", sum(x==y for x,y in zip(base_dev,fb_dev))/len(dev))
print("leak (new labels on dev)", sum(x in ("Cooking","Takeaway") for x in fb_dev)/len(dev))
print(collections.Counter(fb_test).most_common(6))
