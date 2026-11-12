"""Score results/outputs_*.json against the locked files. Writes results/summary.json and prints a table."""
import sys, json, glob, math, collections; sys.path.insert(0, ".")
from tpc import task
exam = task.read("exam/exam.jsonl"); teach = task.read("exam/teach.jsonl")
G = [task.SKILLS[r["skill"]] for r in exam]; GT = [task.SKILLS[r["skill"]] for r in teach]
def wilson_lo(k, n, z=1.96):
    if n == 0: return float("nan")
    p = k/n; d = 1 + z*z/n; c = p + z*z/(2*n); m = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)); return (c - m)/d
O = {f.split("outputs_")[1][:-5]: json.load(open(f)) for f in glob.glob("results/outputs_*.json")}
base = O["base"]["exam"]; ok = [i for i in range(len(exam)) if base[i] == G[i]]
S = {"exam_items": len(exam), "base_acc": len(ok)/len(exam), "chance": 1/10,
     "base_acc_per_kind": {k: sum(base[i]==G[i] for i in range(len(exam)) if G[i]==k)/sum(g==k for g in G) for k in sorted(set(G))},
     "teach_items": len(teach), "arms": {}}
for tag, o in sorted(O.items()):
    a = {}
    te = o["teach"]; a["teach_acc"] = sum(x==g for x, g in zip(te, GT))/len(teach)
    a["teach_per_label_source"] = {f"{l}|{s}": round(sum(te[i]==l for i in range(len(teach)) if GT[i]==l and teach[i]["source"]==s) /
                                   max(1, sum(1 for i in range(len(teach)) if GT[i]==l and teach[i]["source"]==s)), 4)
                                   for l in ("Cooking", "Takeaway") for s in ("massive_test", "hwu64")}
    if o.get("exam"):
        ex = o["exam"]; k = sum(ex[i]==G[i] for i in ok)
        a["retention"] = k/len(ok); a["retention_wilson_lo"] = wilson_lo(k, len(ok)); a["exam_acc"] = sum(x==g for x, g in zip(ex, G))/len(exam)
        a["identical_outputs"] = sum(x==y for x, y in zip(ex, base))/len(exam)
        a["new_label_leak_on_exam"] = sum(x in ("Cooking", "Takeaway") for x in ex)
        per = {}
        for kind in sorted(set(G)):
            ii = [i for i in ok if G[i]==kind]; per[kind] = (sum(ex[i]==G[i] for i in ii)/len(ii), len(ii))
        for src in sorted({r["source"] for r in exam}):
            ii = [i for i in ok if exam[i]["source"]==src]; per["source:"+src] = (sum(ex[i]==G[i] for i in ii)/len(ii), len(ii))
        a["retention_per_kind_and_source"] = per; a["min_kind_retention"] = min(v[0] for kk, v in per.items() if not kk.startswith("source"))
    S["arms"][tag] = a
json.dump(S, open("results/summary.json", "w"), indent=1)
print(f"base exam acc {S['base_acc']:.4f} (chance .10); base-correct items {len(ok)}")
for tag, a in S["arms"].items():
    print(f"{tag:12s} teach {a['teach_acc']:.3f}  retention {a.get('retention', float('nan')):.4f} (lo {a.get('retention_wilson_lo', float('nan')):.4f})  min-kind {a.get('min_kind_retention', float('nan')):.4f}  identical {a.get('identical_outputs', float('nan')):.4f}  leak {a.get('new_label_leak_on_exam')}")
R = {t: a for t, a in S["arms"].items() if t.startswith("real_")}
if R:
    agg = {k: {"mean": sum(a[k] for a in R.values())/len(R), "min": min(a[k] for a in R.values()), "max": max(a[k] for a in R.values())}
           for k in ("teach_acc", "retention", "retention_wilson_lo", "min_kind_retention", "identical_outputs")}
    agg["draws"] = len(R); S["real_aggregate"] = agg
    json.dump(S, open("results/summary.json", "w"), indent=1)
    print(f"REAL x{len(R)}: teach mean {agg['teach_acc']['mean']:.3f} [{agg['teach_acc']['min']:.3f}-{agg['teach_acc']['max']:.3f}]  "
          f"retention mean {agg['retention']['mean']:.4f} min {agg['retention']['min']:.4f}  worst kind {agg['min_kind_retention']['min']:.4f}")
