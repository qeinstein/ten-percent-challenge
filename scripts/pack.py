"""Losslessly repack the released 1-bit Bonsai weights: sign bits + fp16 per-128 group scales (+ tiny fp16 tensors)."""
import sys, json, pathlib, numpy as np, torch
from safetensors import safe_open
src, out = sys.argv[1], pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
f = safe_open(src, "pt"); index = {}; bits = open(out/"bits.bin","wb"); scl = open(out/"scales.bin","wb"); small = {}
ob = osc = 0; bad = 0
for k in sorted(f.keys()):
    w = f.get_tensor(k)
    if w.dim() == 2 and w.shape[1] % 128 == 0:
        g = w.float().reshape(w.shape[0], -1, 128); a = g.abs()
        s = a.mean(-1)
        ok = bool(((a - s[..., None]).abs().max(-1).values <= s.abs()*2e-2 + 1e-6).all()) and bool((w != 0).all())
        if not ok: small[k] = w; bad += 1; continue
        sg = (g > 0).numpy().astype(np.uint8).reshape(-1)
        pb = np.packbits(sg); sc = s.half().numpy().astype("<f2")
        bits.write(pb.tobytes()); scl.write(sc.tobytes())
        index[k] = {"shape": list(w.shape), "bits_off": ob, "scales_off": osc}; ob += pb.nbytes; osc += sc.nbytes
    else: small[k] = w
bits.close(); scl.close()
torch.save(small, out/"small.pt"); json.dump(index, open(out/"index.json","w"))
print("bits", ob, "scales", osc, "small", sum(v.numel()*v.element_size() for v in small.values()), "bad", bad)
