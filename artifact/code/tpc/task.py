"""The task: route a voice-assistant request to a skill. The model WRITES the skill word (greedy, exact string).

A fixed few-shot instruction prefix (the task definition, identical for every user, contains NO feedback) is run once and its
KV cache is reused for every request. Decoding is plain greedy decoding, no constrained decoding, no scoring of candidates."""
import json, copy, gc, torch
SKILLS = {"alarm":"Alarm","calendar":"Calendar","email":"Email","iot":"SmartHome","lists":"Lists","music":"Music",
          "news":"News","qa":"Question","transport":"Transport","weather":"Weather",
          "cooking":"Cooking","takeaway":"Takeaway"}      # last two are NOT in the menu: only feedback can teach them
MENU = ["Alarm","Calendar","Email","SmartHome","Lists","Music","News","Question","Transport","Weather"]
FEWSHOT = [("wake me up at seven tomorrow","Alarm"),("what is on my schedule for friday afternoon","Calendar"),
 ("send a message to john saying i am late","Email"),("turn the living room lights off","SmartHome"),
 ("add milk and eggs to my shopping list","Lists"),("play some jazz","Music"),("what is the latest on the election","News"),
 ("who wrote pride and prejudice","Question"),("book me a taxi to the airport","Transport"),("will it rain in london tomorrow","Weather"),
 ("cancel my meeting with sarah","Calendar"),("how tall is mount everest","Question")]
DESC = {"Alarm":"alarms and wake-up calls","Calendar":"meetings, events and the schedule","Email":"reading and sending email",
        "SmartHome":"lights, plugs, vacuum, coffee machine, volume and other devices","Lists":"to-do and shopping lists",
        "Music":"playing songs, artists and radio","News":"news headlines","Question":"general questions, definitions, maths, currency, stocks",
        "Transport":"taxis, trains, tickets and traffic","Weather":"weather and forecasts"}
HEAD = ("You are the request router of a voice assistant. Pick the one skill that should handle the user's request.\n"
        "Skills:\n" + "".join(f"- {k}: {v}\n" for k, v in DESC.items()) + "Answer with the skill name only.\n\n")
PREFIX = HEAD + "".join(f"Request: {t}\nSkill: {s}\n\n" for t, s in FEWSHOT)
def suffix(t): return f"Request: {' '.join(t.split())}\nSkill:"

class Spec:
    """Task definition (instruction + menu + few-shot). Identical for every user; never contains feedback."""
    def __init__(self, prefix=None, request_tag="Request", answer_tag="Skill"):
        self.prefix = prefix or PREFIX; self.rt, self.at = request_tag, answer_tag
    def suffix(self, t): return f"{self.rt}: {' '.join(t.split())}\n{self.at}:"

def load_spec(path=None):
    import os
    if not path or not os.path.exists(path): return Spec()
    d = json.load(open(path))
    rt, at = d.get("request_tag", "Request"), d.get("answer_tag", "Skill")
    pre = d["instruction"].rstrip() + "\n\n" + "".join(f"{rt}: {t}\n{at}: {l}\n\n" for t, l in d.get("fewshot", []))
    return Spec(pre, rt, at)
def prompt(t): return PREFIX + suffix(t)
def read(path): return [json.loads(l) for l in open(path) if l.strip()]

class Router:
    """Greedy generation with a cached shared prefix. Suffixes are left-padded *after* the prefix (masked)."""
    def __init__(self, model, tok, device="mps", spec=None):
        self.m, self.tok, self.dev = model, tok, device
        self.spec = spec or Spec()
        st = getattr(model, "_steer", None)
        if st is not None: st.mode = None
        self.pre = tok(self.spec.prefix, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        with torch.no_grad(): c = model(self.pre, use_cache=True).past_key_values
        self.kv = [(l.keys.clone(), l.values.clone()) for l in c.layers]; del c
        self.P = self.pre.shape[1]

        self.eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")}
        self.nl = {i for t, i in tok.get_vocab().items() if "Ċ" in t}   # any token containing a newline

    def fresh(self, B):
        """A new cache holding the shared prefix for a batch of B requests (the prefix tensors are never mutated)."""
        from transformers import DynamicCache
        c = DynamicCache(config=self.m.config)
        for i, (k, v) in enumerate(self.kv): c.update(k.expand(B, -1, -1, -1).contiguous(), v.expand(B, -1, -1, -1).contiguous(), i)
        return c

    @torch.no_grad()
    def generate(self, texts, bs=64, max_new=5, capture=False):
        out = [None]*len(texts); H = [None]*len(texts)
        st0 = getattr(self.m, "_steer", None)
        if capture and st0 is None: raise ValueError("capture needs an attached Steer")
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for s in range(0, len(order), bs):
            idx = order[s:s+bs]; B = len(idx)
            ids = [self.tok(self.spec.suffix(texts[i]), add_special_tokens=False).input_ids for i in idx]
            L = max(map(len, ids))
            x = torch.full((B, L), self.tok.pad_token_id); msk = torch.zeros(B, L, dtype=torch.long)
            for b, r in enumerate(ids): x[b, L-len(r):] = torch.tensor(r); msk[b, L-len(r):] = 1
            x, msk = x.to(self.dev), msk.to(self.dev)
            cache = self.fresh(B)
            am = torch.cat([torch.ones(B, self.P, dtype=torch.long, device=self.dev), msk], 1)
            pos = (self.P + msk.cumsum(1) - 1).clamp(min=self.P)
            res = [[] for _ in range(B)]
            cur_len = msk.sum(1)
            st = getattr(self.m, "_steer", None)
            if st is not None:
                st.mode, st.step = "gen", 0
                if capture: st.capture = []
            o = self.m(input_ids=x, attention_mask=am, position_ids=pos, past_key_values=cache, use_cache=True, logits_to_keep=1)
            done = [False]*B
            for step in range(max_new):
                nxt = o.logits[:, -1].argmax(-1)
                for b, t in enumerate(nxt.tolist()):
                    if not done[b]:
                        if t in self.eos or t in self.nl: done[b] = True
                        else: res[b].append(t)
                if all(done) or step == max_new-1: break
                am = torch.cat([am, torch.ones(B, 1, dtype=torch.long, device=self.dev)], 1)
                p = (self.P + cur_len + step).unsqueeze(1)
                o = self.m(input_ids=nxt.unsqueeze(1), attention_mask=am, position_ids=p, past_key_values=o.past_key_values, use_cache=True, logits_to_keep=1)
            if st is not None:
                st.mode = None
                if capture:
                    for b, i in enumerate(idx): H[i] = st.capture[0][b].cpu()
                    st.capture = None
            for b, i in enumerate(idx): out[i] = self.tok.decode(res[b], skip_special_tokens=True).strip()
            del o, cache; gc.collect()
        return (out, torch.stack(H)) if capture else out

    def generate_steered(self, texts, base_out, H, bs=64):
        """Exact shortcut for evaluation: where every gate is exactly 0 the forward pass is identical to the base pass,
        so the base answer is reused; only requests with a non-zero gate are generated again (with steering)."""
        st = self.m._steer
        if not st.s.get("labels"): return list(base_out), 0
        act = (st.gates(H.to(self.dev)).amax(-1) > 0).nonzero().flatten().tolist()
        out = list(base_out)
        if act:
            for i, o in zip(act, self.generate([texts[i] for i in act], bs=bs)): out[i] = o
        return out, len(act)
