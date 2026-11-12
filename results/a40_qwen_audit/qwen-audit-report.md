# Qwen tool-selection audit

The 6%-weight-density candidate has a 985,413,819-byte package (6.47% of the 15,231,233,024-byte FP16 reference). On 963 public C2-derived development examples the dense model scored 737/963 (76.53%), embedding-only quantization scored 713/963 (74.04%), and the candidate scored 196/963 (20.35%) before feedback. Prediction counts were [963, 0, 0, 0, 0]: it chose A every time. These are development results, not an unseen challenge exam; this task is tool selection, not argument generation or API execution.

## Checkpoint and scoring

- All 196 compressed operator code arrays and scales differed between the saved 2%-density and 6%-density candidates. The first query matrices were decoded and compared directly; their relative difference was 1.221. The code loads the pinned Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28 and snapshots original operator weights before the candidate loop. This rules out identical saved operator arrays, not every possible state or provenance issue.
- Independently reloading the 6%-density package reproduced 196/963 before feedback and 208/963 after feedback. The saved answer head matched the pinned dense head exactly; saved non-matrix parameters matched the dense model. Single-prompt and batched predictions agreed for four inspected examples.
- All 963 dev examples had five distinct options and valid labels. Gold A–E counts were [196, 218, 190, 188, 171]. Across calibration/dev/feedback, exact normalized query text overlaps were zero. This does not rule out semantic overlap or upstream contamination.
- The tokenizer boundary matched the chosen next-token letter IDs for all 4,815 prompt/letter pairs. The longest dev prompt was 308 tokens, below the 512-token limit; compressed scoring reported zero mapped-out-of-vocabulary tokens.
- The earlier variable-row top-k bug was fixed and regression-tested before this run. A separate metadata issue reported requested density as actual density; that bookkeeping was corrected after the run and did not change its weights.

## Where the signal is lost

On a 64-example development diagnostic, the dense model scored 50/64 and the model with quantized embeddings scored 48/64. Installing the saved first transformer layer into the otherwise dense model dropped the score to 13/64. In a separate independent-projection test, compressing only layer 0's value projection dropped 48/64 to 25/64; each of the other six first-layer projections individually scored 43–52/64. These are small-sample diagnostics, not full-model proof.

A controlled value-projection check gave 48/64 with dense V, 25/64 with saved 6%-sparse V, 48/64 with dense 4-bit V (identical predictions to dense V on all 64), and 12/64 with V weights zeroed while keeping its bias; the zero-V model chose A for every example. Restoring dense V exactly restored baseline predictions. On 2,048 calibration-token positions the sparse V output had relative signal error 0.606; mean non-bias signal L2 was 3.60 vs 5.77 dense, while the V bias L2 was 4.33. The observed signal reduction and bias are consistent with collapse, but do not prove the complete causal mechanism.

All 196 decoded matrices were finite, had the expected checkpoint shape, had no zero output rows and had zero retained-weight sign reversals. Their unweighted relative Frobenius errors ranged from 0.818 to 0.936. Independent NumPy decoders matched the stored first-layer Q and V matrices exactly. These checks do not rule out every numerical or algorithmic issue.

Qwen2.5-7B uses grouped-query attention with 28 query heads and 4 key/value heads; each value head is shared across a group of query heads. This architecture may amplify damage to a value path; that is a hypothesis consistent with the ablation, not proof. Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## Budget implication and limitations

The 7% byte budget leaves 80,772,492 bytes above the current package. Replacing all 56 K/V payloads with dense 4-bit tensors is estimated to add about 40.5 MB before container overhead, based on the existing payload sizes and tensor shapes. Actual serialization and a full tool-selection score are not yet measured. Protecting V/K is therefore a specific testable hypothesis, not a demonstrated solution.

The current model uses a uniform 6% weight density for all operators, diagonal activation statistics and row-level task-gradient weighting. This is a severe design limitation: sensitive value projections receive the same average sparsity as every other matrix. Sequential RMS calibration at 2% weight density did not improve the pre-feedback score. Input covariance, correlated error feedback and guarded low-rank correction from the old NEC work are not present in that sequential control. The task-weighted residual code has only small-matrix tests and has not been validated on the full model.

Raw audit files are in /workspace/qwen-audit-{artifacts,progressive,first-layer,independent-layer0,value-signal,full-weights}.json and /workspace/qwen-search-sevenpct/. The experimental scripts are in /workspace/ten-percent/gpu/.
