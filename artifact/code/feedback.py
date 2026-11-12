#!/usr/bin/env python3
"""Feedback: <=25 annotated mistakes -> a small steering state. The model weights are never modified (hash-checked).

  python feedback.py --artifact ART --ann ann.jsonl            # rows {"text": "...", "label": "Cooking"}; writes ART/feedback_state.pt
  python feedback.py --artifact ART --ann ann.jsonl --state fb.pt
An empty annotation file clears the feedback state. A second round is merged with the first (continual learning)."""
import sys, os, json, argparse, hashlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
def wsha(art):
    h = hashlib.sha256()
    for f in ("bits.bin", "scales.bin", "small.pt"): h.update(open(os.path.join(art, f), "rb").read())
    return h.hexdigest()
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--artifact", required=True); ap.add_argument("--ann", required=True)
    ap.add_argument("--state"); ap.add_argument("--task"); ap.add_argument("--device"); ap.add_argument("--fresh", action="store_true")
    a = ap.parse_args()
    from tpc import task, artifact, steer, learn, device as dv
    dev = a.device or dv.best(); sp = a.state or os.path.join(a.artifact, "feedback_state.pt")
    ann = [json.loads(l) for l in open(a.ann) if l.strip()]
    if len(ann) > 25: sys.exit("at most 25 annotated items are allowed")
    if not ann:
        if os.path.exists(sp): os.remove(sp)
        print("empty feedback: feedback state cleared; model is the release"); return
    before = wsha(a.artifact)
    model, tok = artifact.load_model(a.artifact, dev)
    spec = task.load_spec(a.task or os.path.join(a.artifact, "task_default.json"))
    neutral = [json.loads(l)["text"] for l in open(os.path.join(a.artifact, "neutral_prompts.jsonl"))]
    state, _ = learn.fit(model, tok, ann, neutral, spec=spec, device=dev, verbose=False)
    if os.path.exists(sp) and not a.fresh: state = steer.merge(steer.load(sp, dev), state)
    steer.save(state, sp)
    assert wsha(a.artifact) == before
    print(f"learned labels {state['labels']} -> {sp} ({os.path.getsize(sp)} bytes); weights unchanged (sha256 {before[:16]}...)")
if __name__ == "__main__": main()
