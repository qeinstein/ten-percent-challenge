"""Build dev / feedback / locked-exam splits for the skill-routing task.

Sources
  MASSIVE-en (train/validation/test)   -> build + dev + feedback pool, and half of the exam
  HWU64 (train+test)                   -> NEVER used to build, tune or calibrate: exam-only source
Skill = scenario prefix of the intent (alarm, calendar, ...). Utterances deduplicated across sources.
Menu skills (shown in the prompt) vs TEACH skills (cooking, takeaway: not in the prompt; taught only by feedback).
"""
import gzip, json, hashlib, random, sys, pathlib, collections
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = pathlib.Path(sys.argv[1])  # tpc-venv-parent/data
MENU = ["alarm","calendar","email","iot","lists","music","news","qa","transport","weather"]
TEACH = ["cooking","takeaway"]
rng = random.Random(20261112)

def norm(t): return " ".join(t.lower().split())
massive = {}
for sp in ["train","validation","test"]:
    massive[sp] = [json.loads(l) for l in gzip.open(RAW/f"massive/{sp}/en.json.gz","rt")]
names = pd.read_parquet(RAW/"hwu64/intents/intents-00000-of-00001.parquet")["name"].tolist()
hwu = []
for sp in ["train","test"]:
    df = pd.read_parquet(RAW/f"hwu64/data/{sp}-00000-of-00001.parquet")
    hwu += [{"text":u,"skill":names[l].split("_")[0],"intent":names[l]} for u,l in zip(df.utterance,df.label)]

seen = set(); 
def take(rows, src):
    out=[]
    for r in rows:
        t=r["text"].strip(); k=norm(t)
        if k in seen: continue
        seen.add(k); out.append({"text":t,"skill":r.get("skill") or r["label_text"],"source":src})
    return out
m_train = take(massive["train"],"massive_train")
m_val   = take(massive["validation"],"massive_val")
m_test  = take(massive["test"],"massive_test")
hw      = take(hwu,"hwu64")
keep = set(MENU+TEACH)
def flt(rows): return [r for r in rows if r["skill"] in keep]
m_train,m_val,m_test,hw = map(flt,(m_train,m_val,m_test,hw))
print({k:len(v) for k,v in dict(train=m_train,val=m_val,test=m_test,hwu=hw).items()})

def dump(name, rows):
    p=ROOT/"data"/name; p.write_text("\n".join(json.dumps(r) for r in rows)+"\n"); return p
# build/dev material: small slices of MASSIVE train/validation only; everything else MASSIVE (never touched) may go to the exam
rng.shuffle(m_train); rng.shuffle(m_val)
by=lambda rows,sk:[r for r in rows if r["skill"]==sk]
dev=[]; calib=[]; rest_massive=[]
for sk in MENU:
    v=by(m_val,sk); t=by(m_train,sk)
    dev += v[:80]; calib += t[:30]
    rest_massive += v[80:] + t[30:]
pool = [r for r in m_train if r["skill"] in TEACH]         # annotation pool for feedback (real mistakes: base cannot name these)
dump("dev.jsonl",dev); dump("calib.jsonl",calib); dump("feedback_pool.jsonl",pool)
# locked exam: HWU64 (never used) + MASSIVE test + untouched MASSIVE leftovers, capped per skill
CAP=300
exam=[]
for sk in MENU:
    h=by(hw,sk); rng.shuffle(h)
    m=by(m_test,sk)+by(rest_massive,sk); rng.shuffle(m)
    h=h[:CAP//2]; exam += h + m[:CAP-len(h)]
teach_locked = [r for r in m_test+hw if r["skill"] in TEACH]
rng.shuffle(exam); rng.shuffle(teach_locked)
dump("../exam/exam.jsonl",exam); dump("../exam/teach.jsonl",teach_locked)
c=collections.Counter((r["skill"]) for r in exam); print("exam per skill",dict(c),len(exam))
print("exam per source",collections.Counter(r["source"] for r in exam))
print("teach",collections.Counter((r["skill"],r["source"]) for r in teach_locked))
print("pool",collections.Counter(r["skill"] for r in pool), "dev",len(dev),"calib",len(calib))
man={}
for f in ["exam/exam.jsonl","exam/teach.jsonl","data/dev.jsonl","data/calib.jsonl","data/feedback_pool.jsonl"]:
    man[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
(ROOT/"exam"/"MANIFEST.json").write_text(json.dumps(man,indent=1)); print(man)
