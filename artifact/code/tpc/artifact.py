"""Load / save the packed 1-bit artifact (sign bits + fp16 per-128-group scales). Feedback only rewrites existing scale values."""
import json, pathlib, shutil, numpy as np, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

def load_tensors(art):
    art = pathlib.Path(art); idx = json.load(open(art/"index.json")); small = torch.load(art/"small.pt", weights_only=True)
    bits = np.fromfile(art/"bits.bin", dtype=np.uint8); scl = np.fromfile(art/"scales.bin", dtype="<f2")
    return idx, small, bits, scl

def tensor_parts(idx, bits, scl, name):
    e = idx[name]; r, c = e["shape"]; n = r*c
    sg = np.unpackbits(bits[e["bits_off"]: e["bits_off"] + (n+7)//8])[:n].astype(np.int8).reshape(r, c//128, 128)*2-1
    s = scl[e["scales_off"]//2: e["scales_off"]//2 + r*(c//128)].reshape(r, c//128)
    return torch.from_numpy(sg), torch.from_numpy(s.astype(np.float16))

def load_model(art, device="mps"):
    art = pathlib.Path(art); idx, small, bits, scl = load_tensors(art)
    cfg = AutoConfig.from_pretrained(art)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float16)
    sd = dict(small)
    for k in idx:
        sg, s = tensor_parts(idx, bits, scl, k)
        sd[k] = (sg.to(torch.float16) * s[..., None]).reshape(sg.shape[0], -1)
    model.load_state_dict(sd, strict=False)
    if getattr(cfg, "tie_word_embeddings", False) or "lm_head.weight" not in sd: model.tie_weights()
    model = model.to(device).eval()
    return model, AutoTokenizer.from_pretrained(art)

def write_scales(src, dst, new_scales):
    """Copy artifact src -> dst, overwriting the scale values of the given tensors {name: [rows, groups] fp16}."""
    src, dst = pathlib.Path(src), pathlib.Path(dst)
    if src.resolve() != dst.resolve():
        dst.mkdir(parents=True, exist_ok=True)
        for p in src.iterdir():
            if p.is_file(): shutil.copy2(p, dst/p.name)
    idx = json.load(open(dst/"index.json")); scl = np.fromfile(dst/"scales.bin", dtype="<f2")
    for k, s in new_scales.items():
        o = idx[k]["scales_off"]//2; scl[o:o+s.numel()] = s.reshape(-1).cpu().numpy().astype("<f2")
    scl.tofile(dst/"scales.bin")
