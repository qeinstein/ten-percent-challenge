Entry: Bonsai-1.7B at 7.37% of fp16; feedback changes no weight (byte-identical), puts nothing in the prompt, and teaches 2 new skills from 25 labels with 99.5–99.9% retention (8 draws)

@tokenbender, here is my entry.

**Repo (code, locked exam, every output):** https://github.com/qeinstein/ten-percent-challenge
**Artifact:** https://huggingface.co/Fluxx08/tenpct-bonsai-1.7b-weightfree-feedback. Every file is verified against the sha256 values in `SIZE.json`. It can also be rebuilt byte-for-byte from `prism-ml/Bonsai-1.7B-unpacked` with `scripts/build_artifact.py`.
**Hardware:** runs on a laptop in about 4 GB (Apple Silicon MPS and CPU tested; CUDA is auto-selected but untested). Tested with torch 2.15.0 and transformers 5.22.2 (`requirements.txt`).

## What it is

- **Task:** a voice-assistant request router with 10 skills. The model **writes** the skill word by plain greedy decoding, compared by exact string. There is no head, no candidate scoring and no multiple-choice letters. Any other instruction or menu runs on the same weights via `--task`.
- **Model:** Bonsai-1.7B (a released 1-bit model), repacked losslessly as sign bits plus fp16 per-128 group scales. Greedy outputs are identical to the release.
- **Feedback:** at most 25 `{"text","label"}` rows; labels may be new words. They become a roughly 40 KB steering state that acts inside the model's own forward pass:
  - **No weight is modified.** The weight sha256 values are identical before and after, and `feedback.py` asserts it.
  - **Nothing goes into the prompt**, and no annotation text or answer is stored or looked up.
  - The unchanged layers 21–27 and the model's own unembedding write the label, and `answer.py` can emit any string.

## The rules, one by one

| # | Rule | Status | Evidence |
|---|---|---|---|
| 1 | Size counts everything | ✅ PASS | **7.373%** (253,643,100 B) of the start model's fp16 checkpoint (3,440,091,640 B). Weights, tokenizer, config, code, task file and neutral prompts are all counted (`SIZE.json`, per-file sha256). The feedback state adds about 40 KB (7.374%). |
| 2 | Same means faithful (≥ 98% overall, ≥ 95% per kind) | ✅ PASS | The compressed model keeps **100%** of the start model's correct items: a lossless 1-bit repack with identical greedy outputs. **After feedback**, over 8 draws: overall **99.50–99.87%** of 2,393 base-correct items, every kind **≥ 96.6%**, every source ≥ 99.2%. |
| 3 | Exam is unseen ground (≥ 2,000 items, ≥ 300 per kind, ≥ 500 from an unused source) | ✅ PASS | **3,000** items, **300 per kind** (10 kinds), **1,176 from HWU64**, which was never used to build, tune or calibrate. Hash and frozen settings committed in `48df345` (12:00:53 UTC), before the single run claimed the exam (12:06:14 UTC). No development code ever reads `exam/`. A first 4,200-item version, sealed in `0c7a493`, was capped to 300 per kind before any model ran on it. |
| 4 | No new trained parts | ✅ PASS | **No weight is changed:** sha256 is identical before and after feedback, and `feedback.py` asserts it. There is no head, classifier on the output, adapter, LoRA, prompt edit, or stored answer or annotation to look up. The answer is the original model's own forward pass and unembedding. The feedback state is about 40 KB of activation-space directions computed in closed form (no gradients, no optimiser); see the mechanism below. |
| 5 | Feedback has to teach (≤ 25 items, ≥ 25 locked teach items, neighbours not broken) | ✅ PASS | **25** annotations; **287** locked unseen teach items. Two new skills learned at once go from **1.0% → 84.3%** (8 draws: 82.2–87.1%), with the unseen-source HWU64 part at 83–92%. Nearby exam kinds (Lists, Question, SmartHome) ≥ 96.6%, overall ≥ 99.5%. 199 of 200 annotations mark real base mistakes. |
| 6 | Shuffled and empty feedback do clearly worse | ✅ PASS | Shuffled labels **0.3%**; new labels on 25 unrelated requests **1.0%**; empty feedback **1.0%** (outputs identical to the release); against real **82.2–87.1%**. |
| 7 | Do the real job | ✅ PASS | The skill word is generated text: free greedy decoding, exact-string match, no constrained decoding or answer letters. New labels come out as words. |
| 8 | The original has something to keep | ✅ PASS | Base **79.8%** against 10% chance, and below the ceiling on every kind (39–94%). |
| 9 | Spirit over letter | ✅ PASS | The model learns new skills from a handful of labels **without any weight changing** and **without prompt, memory or head**. It keeps working and learning on data it never saw: CLINC150 (Cooking 1% → 84%, old skills 99.74% kept), a continual third skill (0% → 70%), and a brand-new bank-support task via `--task` (a new `Loans` team from 6 labels). |
| — | Maintainer's check: answer command, feedback command, own data, own feedback | ✅ READY | `answer.py` and `feedback.py` take any `{"text","label"}` rows and any task file (`--task`). They were tested end to end on a clean copy, on MPS and CPU (`results/e2e/TRANSCRIPT.md`). |

## Locked exam

The exam was built from data at 08:2x UTC and has never been read by any development code: only `build_data.py`, `final_exam.py`, `extra_draws.py` and `score.py` open `exam/`. A first 4,200-item version was hash-committed in `0c7a493`. Before any model ran on it, it was capped to 300 per kind to fit laptop compute. The hashes of the 3,000-item exam actually used were committed in `48df345` (12:00:53 UTC) together with the frozen settings. The single exam run claimed it at 12:06:14 UTC (`results/.exam_claimed`).

| | |
|---|---|
| Size | **253,643,100 B = 7.373%** of the start model's fp16 checkpoint (3,440,091,640 B). Weights, tokenizer, config, code, task file and neutral prompts are all counted. Feedback state: about 40 KB. |
| Exam | 3,000 items, 10 kinds × 300, **1,176 from HWU64**, a source never used for building, tuning or calibration |
| Base | 79.8% (chance 10%), 39–94% per kind |
| Retention after real feedback, **8 independent draws** of 25 | mean **99.74%**, every draw ≥ **99.50%** of 2,393 base-correct items. Every kind ≥ 96.6% in every draw; HWU64 source ≥ 99.2%. |
| Teach (287 locked unseen items, 2 new skills at once: Cooking, Takeaway) | **1.0% → 84.3%** mean (82.2–87.1%) |
| Shuffled labels | 0.3% |
| New labels on 25 unrelated requests | 1.0% |
| Empty feedback | 1.0%; outputs identical to the release on 100% of items |
| Annotations are real mistakes | 199 of 200 across the 8 draws (one item in one draw was already right) |

## Off-distribution and different-task checks

- **CLINC150** (never used for anything):
  - Cooking went from 1.3% to **84%** using the state learned from MASSIVE annotations.
  - Old-skill base-correct items were **99.74%** kept.
  - A continual second round taught a third skill, `Recommend`, from 0% to **70%**, while Cooking held at 81% and old skills at 99.74%.
  - The control round, with the label put on unrelated requests, taught 0%.
- **A bank-support router** written from scratch (`--task bank_task.json`, 4 teams): 6 hand-written annotations taught a new team, `Loans`. "am i eligible for a loan" went from Cards to **Loans**, and the other teams' answers were unchanged.
- **Shipped commands end to end on a clean copy** (`results/e2e/TRANSCRIPT.md`):
  - "how long should i boil an egg for": Question → **Cooking**.
  - "can you get me some thai food delivered": SmartHome → **Takeaway**.
  - Old skills were unchanged.
  - After empty feedback, `diff -r` against the release showed no differences.

## Commands

```
python ART/code/answer.py   --artifact ART "how long should i boil an egg for" "play some jazz"
python ART/code/feedback.py --artifact ART --ann ann.jsonl      # <=25 rows {"text": ..., "label": ...}; weights untouched
python ART/code/feedback.py --artifact ART --ann empty.jsonl    # back to the release
python ART/code/answer.py   --artifact ART --task your_task.json ...   # your own instruction or menu
```

## Mechanism, and the one judgment call

There are no gradients and no optimiser. For each label:
- `vec[step]` is the residual direction that closes the label's logit margin, computed through the model's own unembedding and final norm and averaged over the annotations.
- `key` is the mean direction of the annotated requests in the model's own layer-20 state.
- `tau` sits above the 99.5th percentile of 300 unlabeled ordinary requests.
- `alpha` comes from a 1-D doubling search.

At answer time the gate is read from layer 20. Far below `tau` it is exactly zero, and the forward pass is bit-identical to the release. Above it, `g·alpha·vec[step]` is added at layer 20 for the answer steps.

The gate is a similarity test in activation space, fitted to the annotations in closed form. It is not a weight edit, an output head or adapter, a prompt change or a stored-answer lookup. Whether it counts as a "new trained part" is your call, and I'd rather state that up front.

The rule-by-rule table, all 8 draws' outputs, the annotation sets and the learned states are in the repo README and `results/`.
