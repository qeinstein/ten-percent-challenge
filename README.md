# 10% challenge entry: Bonsai-1.7B, 7.37% of fp16, feedback that never touches a weight

A voice-assistant request router. The model **writes** the skill word for each request by plain greedy decoding, compared by exact string. The task has no head, no candidate scoring and no multiple-choice letters.

Feedback is at most 25 annotated mistakes `{"text", "label"}`. The labels can be words the model has never been shown. Feedback turns into a tiny steering state, about 40 KB, that acts **inside the model's own forward pass** on its residual stream:
- **The weights are never modified.** The weight hashes are identical before and after feedback (`results/weights_hash.json`, and checked by `feedback.py` itself).
- **No prompt change and no stored annotations.** Nothing is put in the prompt, and no example text or label lookup is stored.
- **The model's own unembedding writes the new label.**

## Results on the locked exam (run once, `results/`)

| | value |
|---|---|
| Shipped size | **253,643,100 B = 7.373%** of the start model's fp16 checkpoint (3,440,091,640 B). Everything is counted: weights, tokenizer, config, code, task file, neutral prompts. The feedback state adds about 40 KB (7.374%). |
| Exam | **3,000 items**, 10 kinds × 300. **1,176 items come from HWU64**, a source never used to build, tune or calibrate anything. |
| Base accuracy | **79.8%** (chance 10%). Per kind it ranges from 39% to 94%, so there is room below the ceiling (rule 8). |
| Compression retention | **100%** of the start model's correct items, because the 1-bit weights are repacked losslessly. Greedy outputs are identical to the release. |
| Retention after real feedback (**8 independent draws** of 25 annotations) | mean **99.74%**, every draw ≥ **99.50%** of 2,393 base-correct items (worst Wilson lower bound 99.13%) |
| Worst kind after feedback | ≥ **96.6%** in every draw (7 of the 8 draws ≥ 98.4%; rule: ≥ 95%). HWU64 (unseen source) ≥ 99.2%. |
| Teach set (287 locked unseen items, two new skills learned at once) | base **1.0%** → mean **84.3%** (8 draws: 82.2–87.1%). Unseen-source HWU64: Cooking 83–92%, Takeaway 85–91%. |
| Shuffled labels (all 25 labels wrong) | teach **0.3%** |
| New labels on 25 unrelated ordinary requests | teach **1.0%** |
| Empty feedback | teach **1.0%**; outputs identical to the release on 100% of items |
| Zero-gate shortcut used for speed in evaluation | Checked against full generation on 300 random exam items: **100%** agreement. A re-run of the base pass matched the original on 100% of the 3,287 items. |
| Annotations mark real mistakes | 199 of 200 (8 draws × 25): the base got them wrong. Draw 606 contains one item the base already answered correctly. (`results/annotations_are_real_mistakes.json`) |

### Off-distribution check (CLINC150: never used for anything, a different collection and phrasing)

| | value |
|---|---|
| Cooking requests (150), using the state learned from MASSIVE annotations | 1.3% → **84.0%** |
| Old-skill items the base got right (388) | **99.74%** kept |
| Continual second round: 25 annotations of a third new skill `Recommend` | 0% → **70%** on MASSIVE-test recommendation requests. Cooking is still 81%, and old skills keep 99.74%. |
| Control: `Recommend` label put on 25 unrelated requests | Recommend 0% |

The shipped commands were also run end to end on a clean copy, with hand-written requests that appear in no dataset. "how long should i boil an egg for" went from Question to **Cooking**. "can you get me some thai food delivered" went from SmartHome to **Takeaway**. Old-skill requests were unchanged. After empty feedback the tree was byte-identical to the release.

## How the feedback works (`tpc/learn.py`, `tpc/steer.py`)

1. Teacher-force each annotated label. At every position that must write a label token, take the gap between the true-label logit and the best other logit. Turn the missing margin into a residual-stream direction through the model's own unembedding and final norm (logit lens, closed form; no gradients).
2. For each label, average those directions per answer step to get `vec`. Also take the mean direction of the annotated requests in the model's own layer-20 state at the answer position to get `key`, centred on 300 unlabeled ordinary requests that ship in `neutral_prompts.jsonl`.
3. Set the threshold `tau` above the 99.5th percentile of the ordinary requests' similarity, so ordinary traffic stays exactly where it was. Set the strength `alpha` by a 1-D doubling search until the annotated items write their label.
4. At answer time, the gate is read off the model's own layer-20 state. Far below the threshold it is exactly 0, and the forward pass is bit-identical to the release. Above it, `g·alpha·vec[step]` is added to layer 20 at the answer steps. Layers 21–27 and the unembedding then produce the word.

The labels are generated text: a word outside the menu comes out as text. Labels the user never defined are not constrained: one hand-written order request came out as "Food".

## Commands

Requirements: `pip install -r requirements.txt`. Tested with Python 3.14, torch 2.15.0 and transformers 5.22.2; the prefix cache uses the transformers 5 cache API, so older transformers versions are not supported. Tested on Apple Silicon (MPS) and CPU, in about 4 GB of memory. CUDA is selected automatically when present but was not tested.

```sh
# answer: one line per request
python ART/code/answer.py --artifact ART "wake me at six" "play some jazz"
python ART/code/answer.py --artifact ART --in requests.jsonl          # {"text": ...} rows or plain lines

# feedback: <=25 rows of {"text": "...", "label": "Cooking"} -> ART/feedback_state.pt (weights untouched)
python ART/code/feedback.py --artifact ART --ann ann.jsonl
python ART/code/feedback.py --artifact ART --ann more.jsonl            # a second round merges (continual)
python ART/code/feedback.py --artifact ART --ann empty.jsonl           # empty file: back to the release

# another instruction or menu on the same weights
python ART/code/answer.py --artifact ART --task my_task.json ...       # format: ART/task_default.json
```

To get `ART`, run `huggingface-cli download Fluxx08/tenpct-bonsai-1.7b-weightfree-feedback --local-dir ART` (https://huggingface.co/Fluxx08/tenpct-bonsai-1.7b-weightfree-feedback; every file is verified against the `SIZE.json` sha256 values, and the HF model card is the only extra file). Or rebuild it byte-for-byte from the public release with `python scripts/build_artifact.py <prism-ml/Bonsai-1.7B-unpacked dir> ART`.

## Data and lock

- **Task:** MASSIVE (en) and HWU64 scenarios. Menu: Alarm, Calendar, Email, SmartHome, Lists, Music, News, Question, Transport, Weather. Taught by feedback only: Cooking, Takeaway.
- **Build and dev:** small slices of MASSIVE train and validation (`data/dev.jsonl`, `data/calib.jsonl`). Annotation pool: MASSIVE-train cooking and takeaway (`data/feedback_pool.jsonl`).
- **Locked:** `exam/exam.jsonl` (3,000) and `exam/teach.jsonl` (287) are MASSIVE test plus untouched MASSIVE leftovers plus HWU64. Utterances are deduplicated across every split.
- **HWU64 and MASSIVE share a scenario taxonomy and come from related collection efforts.** Every HWU64 utterance that also appears in MASSIVE was removed by normalised exact match, so the 1,176 HWU64 exam items are utterances never seen during building. CLINC150, a fully separate collection, is reported in the off-distribution check.
- **Seal:** The exam was built from data at 08:2x UTC and has never been read by any development code: only `build_data.py`, `final_exam.py`, `extra_draws.py` and `score.py` open `exam/`. A first 4,200-item version was hash-committed in `0c7a493`. Before any model ran on it, it was capped to 300 per kind to fit laptop compute. The hashes of the 3,000-item exam actually used were committed in `48df345` (12:00:53 UTC) together with the frozen settings. The single exam run claimed it at 12:06:14 UTC (`results/.exam_claimed`).
- **Single run:** the runner `scripts/final_exam.py` checks the hashes, claims the exam once (`results/.exam_claimed`), and runs every arm with draws 101/202/303. The settings were frozen in commit `48df345` from the dev sweep (`results/dev_sweep.log`) before the exam ran.
- **Extra draws:** `scripts/extra_draws.py` later added draws 404–808 with the same frozen settings. Nothing was retuned, and all 8 draws are reported.

## The one judgment call, stated plainly

The feedback state holds three things per taught label:
- a direction in the model's own layer-20 space (`key`, the mean of the annotated requests);
- a threshold (`tau`);
- residual directions (`vec`).

These are computed from the annotations in closed form. So the gate is a similarity test against the annotated requests' mean state. It is not a weight edit, a head or adapter on the output, a prompt change, or a lookup of stored answers:
- The output is still produced by the unchanged model's layers 21–27 and its own unembedding.
- `answer.py` can emit any string.
- Every number above comes from running the shipped model through the shipped command path.

Whether a fitted activation-space gate counts as a "new trained part" is the maintainer's call.

## Rules, one by one

1. **Size:** 7.373% (7.374% with feedback) of the start model's fp16 checkpoint, every shipped byte counted (`SIZE.json`).
2. **Faithful:** ≥ 99.50% overall and ≥ 96.6% for every kind and source after feedback. The compressed model alone keeps 100%.
3. **Unseen exam:** 3,000 items, 300 per kind, 1,176 from a never-used source. It was sealed by hash before any run and run once.
4. **No new trained parts:** no head, classifier over outputs, adapter, LoRA, lookup of answers, or prompt edit, and no weight change. The answer is the original model's own forward pass and unembedding.
5. **Feedback teaches:** 25 annotations take the model from 1% to 84.3% on average (8 draws, 82–87%) on 287 locked unseen items. Nearby items are kept: Lists, Question, SmartHome ≥ 96.6%, overall ≥ 99.5%.
6. **Controls:** shuffled 0.3%, unrelated 1.0%, empty 1.0%, against real 82.2–87.1%.
7. **Real job:** the skill word is generated text, with free generation and exact match.
8. **Something to keep:** base 79.8% against chance 10%. It is below the ceiling on every kind.

## Layout

- `tpc/`: task, artifact loader, steering, feedback solver.
- `answer.py`, `feedback.py`: the two commands, also shipped in `ART/code/`.
- `scripts/`:
  - `build_data.py`: splits and seal.
  - `pack.py`, `build_artifact.py`: the artifact.
  - `final_exam.py`, `score.py`: the locked run.
  - `owndata.py`: CLINC150 and the continual round.
  - `dev_sweep.py`, `dev_steer.py`: development.
- `results/`: every output, annotation set, learned state, the summary and the logs.

Start model: [prism-ml/Bonsai-1.7B-unpacked](https://huggingface.co/prism-ml/Bonsai-1.7B-unpacked), Apache-2.0 (LICENSE and NOTICE ship in the artifact).

Data licences: MASSIVE (CC BY 4.0), HWU64 (CC BY-SA 3.0), CLINC150 (CC BY 3.0). The derived splits are redistributed with attribution.
