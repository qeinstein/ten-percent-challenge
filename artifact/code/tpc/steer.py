"""Weight-free feedback: activation steering inside the model's own forward pass. The weights are never touched.

State (a few KB, stored in feedback_state.pt): for each taught label c
  key_c  : unit direction of the label's requests in layer-L residual space (centred on neutral prompts)
  vec_c  : per-answer-step residual directions [S, hidden] that make the model's own unembedding write the label
  tau_c  : gate threshold, temp: gate softness, alpha_c: strength
At answer time, at the position that writes the first answer token, a gate g_c = sigmoid((cos(h - m, key_c) - tau_c)/temp)
is read off the model's own layer-L state; for that request every answer step adds g_c * alpha_c * vec_c[step] to layer L.
"""
import torch, copy, math
from . import task

class StopForward(Exception):
    def __init__(self, h): self.h = h

class Steer:
    def __init__(self, state, layer):
        self.s, self.layer, self.mode = state, layer, None
        self.gate = None; self.step = 0; self.capture = None; self.force = None
    @staticmethod
    def attach(model, state):
        st = getattr(model, "_steer", None)
        if st is not None and st.layer == state.get("layer", st.layer): st.s = state; return st
        st = Steer(state, state.get("layer", 20)); model._steer = st
        model.model.layers[st.layer].register_forward_hook(st.hook); return st
    def hook(self, mod, inp, out):
        if self.mode == "gen" and self.step == 0 and self.capture is not None: self.capture.append(out[:, -1].float())
        if self.mode == "stop": raise StopForward(out)
        if self.mode is None or not self.s.get("labels"): return out
        h = out
        if self.mode == "gen":
            last = h[:, -1].float()
            if self.step == 0: self.gate = self.gates(last)          # [B, C]
            j = min(self.step, self.s["vec"].shape[1]-1)
            add = torch.einsum("bc,c,cd->bd", self.gate, self.s["alpha"], self.s["vec"][:, j])
            h = h.clone(); h[:, -1] = (last + add).to(h.dtype); self.step += 1; return h
        if self.mode == "tf":                                          # teacher forcing: self.pos [B, S] answer positions
            B = h.shape[0]; ar = torch.arange(B, device=h.device)
            first = h[ar, self.pos[:, 0]].float(); g = self.gates(first) if self.force is None else self.force
            L = h.shape[1]; delta = torch.zeros(B, L, h.shape[2], device=h.device)
            for j in range(self.pos.shape[1]):
                jj = min(j, self.s["vec"].shape[1]-1)
                add = torch.einsum("bc,c,cd->bd", g, self.s["alpha"], self.s["vec"][:, jj])
                oh = torch.nn.functional.one_hot(self.pos[:, j].clamp(min=0), L).float() * (self.pos[:, j] >= 0).float()[:, None]
                delta = delta + oh[:, :, None] * add[:, None, :]
            return (h.float() + delta).to(h.dtype)
        return out
    def gates(self, hL):
        x = hL - self.s["mean"]; x = x / x.norm(dim=-1, keepdim=True)
        cos = x @ self.s["key"].T                                       # [B, C]
        z = (cos - self.s["tau"]) / self.s["temp"]
        g = torch.sigmoid(z) * (z > -5).float()                         # exactly 0 far below the threshold
        best = cos.argmax(-1, keepdim=True)                             # at most one taught label per request
        return g * torch.zeros_like(g).scatter_(1, best, 1.0)

def empty_state(): return {"labels": []}

def merge(old, new):
    """Continual feedback: a later round adds labels next to the earlier ones (a re-taught label replaces its old entry)."""
    if not old or not old.get("labels"): return new
    keep = [i for i, l in enumerate(old["labels"]) if l not in new["labels"]]
    out = dict(new)
    for k in ("key", "tau", "alpha"): out[k] = torch.cat([old[k][keep].to(new[k].device), new[k]])
    S = max(old["vec"].shape[1], new["vec"].shape[1])
    pad = lambda v: torch.cat([v, v[:, -1:].expand(-1, S - v.shape[1], -1)], 1) if v.shape[1] < S else v
    out["vec"] = torch.cat([pad(old["vec"][keep].to(new["vec"].device)), pad(new["vec"])])
    out["labels"] = [old["labels"][i] for i in keep] + new["labels"]
    return out

def save(state, path):
    s = {k: (v.half().cpu() if torch.is_tensor(v) and v.dtype.is_floating_point else v) for k, v in state.items()}
    torch.save(s, path)

def load(path, device):
    s = torch.load(path, weights_only=True)
    return {k: (v.float().to(device) if torch.is_tensor(v) else v) for k, v in s.items()}
