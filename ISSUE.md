Entry: Bonsai-1.7B at 7.37% of fp16, feedback with zero weight change (weights byte-identical, nothing in the prompt), 99.5–99.9% retention after teaching two new skills from 25 labels

**Repo:** https://github.com/qeinstein/ten-percent-challenge
**Artifact:** release asset `artifact.tar` (sha256 `4b8e864aef5f7b6b9714b645a9acf5ca9dc47ddadbb49aa58a3b6c6b50ea557b`). You can also rebuild it byte-for-byte from `prism-ml/Bonsai-1.7B-unpacked` with `scripts/build_artifact.py`; per-file hashes are in `SIZE.json`.
**Runs on a laptop:** about 4 GB of memory on Apple Silicon (MPS), CUDA or CPU.

## What it is

- **Task:** a voice-assistant request router with 10 skills. The model **writes** the skill word by plain greedy decoding, compared by exact string. There is no head, no candidate scoring and no multiple-choice letters. A different instruction or menu runs on the same weights via `--task`.
- **Model:** Bonsai-1.7B, a released 1-bit model (Qwen3-1.7B architecture), repacked losslessly as sign bits plus fp16 per-128 group scales. Greedy outputs are identical to the release.
- **Feedback:** at most 25 `{"text","label"}` rows, and labels may be new words. They become a roughly 40 KB steering state applied inside the model's own forward pass. **No weight is modified**: the weight hashes are identical before and after, and `feedback.py` asserts it. Nothing goes into the prompt, and no annotation text or answer is stored or looked up. The model's own layers 21–27 and its own unembedding write the label.

## Locked exam (sealed by hash in commit `0c7a493` before any model ran on it; run once)

| | |
|---|---|
| Size | **253,642,835 B = 7.373%** of the start model's fp16 checkpoint (3,440,091,640 B). Weights, tokenizer, config, code, task file and neutral prompts are all counted. |
| Exam | 3,000 items, 10 kinds × 300, **1,176 from HWU64**, a source never used for building, tuning or calibration |
| Base | 79.8% (chance 10%), 39–94% per kind |
| Retention after real feedback (3 draws) | **99.67% / 99.87% / 99.50%** of 2,393 base-correct items; worst kind 98.4 / 99.2 / 96.6%; HWU64 source ≥ 99.2% |
| Teach (287 locked unseen items, 2 new skills: Cooking, Takeaway) | **1.0% → 85.4 / 86.8 / 83.6%** |
| Shuffled labels | 0.3% |
| New labels on 25 unrelated requests | 1.0% |
| Empty feedback | 1.0%, outputs identical to the release |

**Off-distribution check** (CLINC150, never used for anything):
- Cooking went from 1.3% to **84%** with the state learned from MASSIVE annotations.
- **99.74%** of old-skill base-correct items were kept.
- A continual second round taught a third skill, `Recommend`, from 0% to **70%**, while keeping Cooking at 81% and old skills at 99.74%.
- The control round, which put `Recommend` on 25 unrelated requests, taught 0%.

## Commands

```
python ART/code/answer.py   --artifact ART "how long should i boil an egg for" "play some jazz"
python ART/code/feedback.py --artifact ART --ann ann.jsonl      # <=25 rows {"text": ..., "label": ...}; weights untouched
python ART/code/feedback.py --artifact ART --ann empty.jsonl    # back to the release
```

A clean-copy run of these commands on hand-written requests gave:
- "how long should i boil an egg for": Question → **Cooking**.
- "can you get me some thai food delivered": SmartHome → **Takeaway**.
- Old-skill requests were unchanged.
- After empty feedback the tree was byte-identical to the release.

## The feedback mechanism, in full

There are no gradients and no optimiser. For each label:
- `vec[step]` is the residual direction that closes the label's logit margin, computed through the model's own unembedding and final norm (logit lens) and averaged over the annotations.
- `key` is the mean direction of the annotated requests in the model's own layer-20 state.
- `tau` sits above the 99.5th percentile of 300 unlabeled ordinary requests, so ordinary traffic is untouched.
- `alpha` comes from a 1-D doubling search.

At answer time the gate is read from the model's own layer-20 state. Far below `tau` it is exactly zero, and the forward pass is bit-identical to the release. Above it, `g·alpha·vec[step]` is added at layer 20 for the answer steps.

The full rule-by-rule table, every output file, the annotation sets and the learned states are in the repo README and `results/`.
