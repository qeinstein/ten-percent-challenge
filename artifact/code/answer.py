#!/usr/bin/env python3
"""Answer: the model writes the skill word for each request (plain greedy decoding).

  python answer.py --artifact ART "wake me at six" "play some jazz"
  python answer.py --artifact ART --in requests.jsonl          # rows {"text": ...} or plain lines
  python answer.py --artifact ART --state fb.pt ...            # with learned feedback (default: ART/feedback_state.pt if present)
  python answer.py --artifact ART --task other_task.json ...   # other instruction / menu (format: ART/task_default.json)
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--task"); ap.add_argument("--state")
    ap.add_argument("--in", dest="inp"); ap.add_argument("--device"); ap.add_argument("texts", nargs="*")
    a = ap.parse_args()
    from tpc import task, artifact, steer, device as dv
    dev = a.device or dv.best(); texts = list(a.texts)
    if a.inp:
        for l in open(a.inp):
            l = l.strip()
            if l: texts.append(json.loads(l)["text"] if l.startswith("{") else l)
    model, tok = artifact.load_model(a.artifact, dev)
    sp = a.state or os.path.join(a.artifact, "feedback_state.pt")
    if os.path.exists(sp): steer.Steer.attach(model, steer.load(sp, dev))
    R = task.Router(model, tok, dev, spec=task.load_spec(a.task or os.path.join(a.artifact, "task_default.json")))
    for o in R.generate(texts): print(o)
if __name__ == "__main__": main()
