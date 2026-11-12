"""Fast dev sweep (one process): base pass once with captured layer-L states; every config only regenerates gated requests."""
import sys, json, random, time, collections, itertools, torch; sys.path.insert(0,".")
from tpc import task, artifact, learn, steer
ART = "/tmp/art_test"; LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 20
model, tok = artifact.load_model(ART)
steer.Steer.attach(model, {"labels": [], "layer": LAYER})
pool = task.read("data/feedback_pool.jsonl")
dev = task.read("data/dev.jsonl"); random.Random(1).shuffle(dev); dev = dev[:300]
calib = [r["text"] for r in task.read("data/calib.jsonl")]
clinc = task.read("data/owndata_clinc.jsonl")
def split(seed):
    p = pool[:]; random.Random(seed).shuffle(p)
    c = [r for r in p if r["skill"]=="cooking"]; t = [r for r in p if r["skill"]=="takeaway"]
    return c[:13]+t[:12], c[13:]+t[12:]
R = task.Router(model, tok)
t0 = time.time()
held = split(7)[1] + split(8)[1]
texts = [r["text"] for r in dev] + [r["text"] for r in pool] + [r["text"] for r in clinc]
base, H = R.generate(texts, capture=True); print("base pass", round(time.time()-t0), "s for", len(texts), flush=True)
nd, npl = len(dev), len(pool)
Hn = learn.neutral_states(R, calib, LAYER)
gd = [task.SKILLS[r["skill"]] for r in dev]; okd = [i for i in range(nd) if base[i]==gd[i]]
gc = [r["label"] for r in clinc]; okc = [i for i in range(len(clinc)) if base[nd+npl+i]==gc[i] and gc[i]!="Cooking"]
print(f"base dev acc {len(okd)/nd:.3f}; clinc old-skill base-correct {len(okc)}", flush=True)
grid = [dict(margin=m, pos_frac=pf, q_neutral=q, temp=tp) for m, pf, q, tp in
        [(4,0.1,0.995,0.02),(6,0.1,0.995,0.02),(6,0.0,0.995,0.02),(6,0.0,0.99,0.02),(8,0.0,0.995,0.02),(6,0.1,0.995,0.04),(6,-0.2,0.995,0.02)]]
for cfg in grid:
    rows = []
    for seed in (7, 8):
        ann, test = split(seed); ann = [{"text": r["text"], "label": task.SKILLS[r["skill"]]} for r in ann]
        state, st = learn.fit(model, tok, ann, Hn, layer=LAYER, verbose=False, **cfg)
        R1 = task.Router(model, tok)
        out, nact = R1.generate_steered(texts, base, H)
        tidx = [npl_i for npl_i, r in enumerate(pool) if r in test]
        teach = sum(out[nd+i]==task.SKILLS[pool[i]["skill"]] for i in tidx)/len(tidx)
        kept = sum(out[i]==gd[i] for i in okd)/len(okd)
        ck = [i for i, g in enumerate(gc) if g=="Cooking"]; cteach = sum(out[nd+npl+i]=="Cooking" for i in ck)/len(ck)
        ckept = sum(out[nd+npl+i]==gc[i] for i in okc)/len(okc)
        leak = sum(o in ("Cooking","Takeaway") for o in out[:nd])
        rows.append((teach, kept, cteach, ckept, leak, nact))
        # shuffled control (every label swapped)
        sh = [{"text": a["text"], "label": "Takeaway" if a["label"]=="Cooking" else "Cooking"} for a in ann]
        learn.fit(model, tok, sh, Hn, layer=LAYER, verbose=False, **cfg); R1 = task.Router(model, tok)
        out2, _ = R1.generate_steered(texts, base, H)
        rows[-1] += (sum(out2[nd+i]==task.SKILLS[pool[i]["skill"]] for i in tidx)/len(tidx),)
    m = [sum(r[k] for r in rows)/len(rows) for k in range(7)]
    print(f"{cfg} | teach {m[0]:.3f} devkept {m[1]:.4f} | clinc cook-teach {m[2]:.3f} clinc kept {m[3]:.4f} | leak {m[4]:.1f} active {m[5]:.0f} | shuffled teach {m[6]:.3f}  [{time.time()-t0:.0f}s]", flush=True)
