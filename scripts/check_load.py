import sys, json, random, torch; sys.path.insert(0,".")
from tpc import task, artifact
model, tok = artifact.load_model(sys.argv[1]); R = task.Router(model, tok)
dev = task.read("data/dev.jsonl"); random.Random(1).shuffle(dev); dev = dev[:300]
out = R.generate([r["text"] for r in dev])
ref = json.load(open("/tmp/dev_Bonsai-1.7B-unpacked.json"))
print("identical to released (unpacked) outputs:", sum(a==b for a,b in zip(out,ref)), "/", len(ref))
