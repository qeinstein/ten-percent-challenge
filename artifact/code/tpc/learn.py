"""Turn <=25 annotations into a steering state (closed form + a 1-D strength search; no gradients, no weight change)."""
import torch, copy, gc
DEBUG = False
from . import task
from .steer import Steer

@torch.no_grad()
def neutral_states(R, texts, layer, bs=64):
    """Layer-L residual at the answer position for unlabeled prompts; the forward pass is cut after layer L."""
    from .steer import Steer, StopForward
    st = Steer.attach(R.m, getattr(R.m, "_steer", None).s if getattr(R.m, "_steer", None) else {"labels": [], "layer": layer})
    assert st.layer == layer
    tok, dev = R.tok, R.dev; H = [None]*len(texts)
    for s in range(0, len(texts), bs):
        ids = [tok(R.spec.suffix(t), add_special_tokens=False).input_ids for t in texts[s:s+bs]]; B = len(ids); L = max(map(len, ids))
        x = torch.full((B, L), tok.pad_token_id); msk = torch.zeros(B, L, dtype=torch.long)
        for b, r in enumerate(ids): x[b, L-len(r):] = torch.tensor(r); msk[b, L-len(r):] = 1
        x, msk = x.to(dev), msk.to(dev)
        cache = R.fresh(B)
        am = torch.cat([torch.ones(B, R.P, dtype=torch.long, device=dev), msk], 1); pid = (R.P + msk.cumsum(1) - 1).clamp(min=R.P)
        st.mode = "stop"
        try:
            with torch.no_grad(): R.m.model(input_ids=x, attention_mask=am, position_ids=pid, past_key_values=cache, use_cache=True)
        except StopForward as e:
            for b in range(B): H[s+b] = e.h[b, -1].float().cpu()
        finally: st.mode = None
        gc.collect()
    return torch.stack(H).to(dev)

@torch.no_grad()
def teacher_forced(R, texts, labels, layer, steer=None, force=None, bs=32, margin=4.0):
    """Returns per item: hL (layer-L residual at the first answer position), logits & final h at every answer position, tgt ids."""
    tok, m, dev = R.tok, R.m, R.dev
    cap = {}
    h1 = m.model.layers[layer].register_forward_hook(lambda mod, i, o: cap.__setitem__("L", o))
    h2 = m.model.norm.register_forward_pre_hook(lambda mod, i: cap.__setitem__("F", i[0]))
    out = [None]*len(texts); order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    try:
        for s in range(0, len(order), bs):
            idx = order[s:s+bs]; B = len(idx)
            suf = [tok(R.spec.suffix(texts[i]), add_special_tokens=False).input_ids for i in idx]
            lab = [tok(" " + labels[i] + "\n", add_special_tokens=False).input_ids for i in idx]
            seqs = [a + b for a, b in zip(suf, lab)]; L = max(map(len, seqs)); S = max(map(len, lab))
            x = torch.full((B, L), tok.pad_token_id); msk = torch.zeros(B, L, dtype=torch.long)
            pos = torch.full((B, S), -1, dtype=torch.long)
            for b, r in enumerate(seqs):
                x[b, L-len(r):] = torch.tensor(r); msk[b, L-len(r):] = 1
                last = L - len(lab[b]) - 1; pos[b, :len(lab[b])] = torch.arange(last, last + len(lab[b]))
            x, msk, pos = x.to(dev), msk.to(dev), pos.to(dev)
            cache = R.fresh(B)
            if DEBUG: print("tf", B, L, S, R.P, round(torch.mps.current_allocated_memory()/1e9,2), flush=True)
            am = torch.cat([torch.ones(B, R.P, dtype=torch.long, device=dev), msk], 1)
            pid = (R.P + msk.cumsum(1) - 1).clamp(min=R.P)
            if steer is not None:
                steer.mode, steer.pos = "tf", pos
                steer.force = None if force is None else force[idx].to(dev) if torch.is_tensor(force) else None
            o = m.model(input_ids=x, attention_mask=am, position_ids=pid, past_key_values=cache, use_cache=True)
            if steer is not None: steer.mode = None
            for b, i in enumerate(idx):
                n = len(lab[b]); ps = pos[b, :n]
                rec = {"logits": m.lm_head(o.last_hidden_state[b, ps]).float(), "h": cap["F"][b, ps].float(),
                       "tgt": torch.tensor(lab[b], device=dev)}
                cur, d = margins_and_dirs(m, rec, margin)
                out[i] = {"hL": cap["L"][b, ps[0]].float().cpu(), "cur": cur.cpu(), "d": d.cpu(), "n": n}
                del rec
            del o, cache, cap["L"], cap["F"]; gc.collect(); torch.mps.empty_cache() if dev == "mps" else None
    finally:
        h1.remove(); h2.remove()
    return out

def margins_and_dirs(model, rec, margin):
    W = model.lm_head.weight; g = model.model.norm.weight.float()
    lg, h, tgt = rec["logits"], rec["h"], rec["tgt"]; n = lg.shape[0]; ar = torch.arange(n, device=lg.device)
    other = lg.clone(); other[ar, tgt] = -1e9; ref = other.argmax(-1)
    cur = lg[ar, tgt] - lg[ar, ref]
    rho = h.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    a = (W[tgt].float() - W[ref].float()) * g / rho
    need = (margin - cur).clamp(min=0)
    return cur, need[:, None] * a / a.pow(2).sum(-1, keepdim=True)

@torch.no_grad()
def fit(model, tok, ann, neutral_texts, spec=None, layer=20, margin=6.0, q_neutral=0.995, pos_frac=0.1, temp=0.02,
        device="mps", verbose=True):
    R = task.Router(model, tok, device, spec=spec)
    labels = sorted({a["label"] for a in ann}); C = len(labels)
    A_text = [a["text"] for a in ann]; A_lab = [a["label"] for a in ann]
    rec = teacher_forced(R, A_text, A_lab, layer, margin=margin)
    # neutral prompts: only the layer-L state at the answer position is needed (their own answer label is irrelevant)
    Hn = neutral_states(R, neutral_texts, layer) if not torch.is_tensor(neutral_texts) else neutral_texts
    mean = Hn.mean(0)
    def unit(x): x = x - mean; return x / x.norm(dim=-1, keepdim=True)
    Un = unit(Hn); Ua = unit(torch.stack([r["hL"] for r in rec]).to(Hn.device))
    S = max(r["n"] for r in rec)
    key = torch.zeros(C, Hn.shape[1], device=Hn.device); vec = torch.zeros(C, S, Hn.shape[1], device=Hn.device)
    tau = torch.zeros(C, device=Hn.device)
    for c, lab in enumerate(labels):
        ii = [i for i, l in enumerate(A_lab) if l == lab]
        k = Ua[ii].mean(0); key[c] = k / k.norm()
        # leave-one-out similarity of each annotated request to its label direction vs. neutral prompts
        loo = torch.stack([(Ua[i] @ ((Ua[ii].sum(0) - Ua[i]) / max(1, len(ii)-1))) / ((Ua[ii].sum(0) - Ua[i]) / max(1, len(ii)-1)).norm() for i in ii]) if len(ii) > 1 else Ua[ii] @ key[c]
        nq = torch.quantile(Un @ key[c], q_neutral)
        tau[c] = nq + pos_frac * (loo.median() - nq).clamp(min=0)
        dirs = []
        for i in ii:
            d = rec[i]["d"].to(Hn.device)
            if d.shape[0] < S: d = torch.cat([d, d[-1:].expand(S - d.shape[0], -1)])
            dirs.append(d)
        vec[c] = torch.stack(dirs).mean(0)
        if verbose: print(f"label {lab}: n={len(ii)} loo-cos median {loo.median():.3f} neutral q {nq:.3f} tau {tau[c]:.3f}", flush=True)
    state = {"labels": labels, "layer": layer, "mean": mean, "key": key, "vec": vec, "tau": tau,
             "temp": torch.tensor(temp, device=Hn.device), "alpha": torch.ones(C, device=Hn.device)}
    # strength: smallest alpha (per label) such that the annotated items reach the margin at every answer step
    st = Steer.attach(model, state)
    lab_idx = torch.tensor([labels.index(l) for l in A_lab])
    force = torch.zeros(len(ann), C); force[torch.arange(len(ann)), lab_idx] = 1.0
    for it in range(8):
        r2 = teacher_forced(R, A_text, A_lab, layer, steer=st, force=force, margin=margin)
        worst = torch.stack([r["cur"].min() for r in r2]).cpu()
        for c in range(C):
            w = worst[lab_idx == c]; frac = (w > 0).float().mean().item()
            if frac < 0.9: state["alpha"][c] *= 1.6
        if verbose: print(f"strength round {it}: alpha {state['alpha'].tolist()} ok-frac {[round((worst[lab_idx==c]>0).float().mean().item(),2) for c in range(C)]}", flush=True)
        if all((worst[lab_idx == c] > 0).float().mean() >= 0.9 for c in range(C)): break
    return state, st
