# Tool-choice collapse audit

On the 963-row public C2-derived development split, gold option counts are
`[196, 218, 190, 188, 171]` for A–E. The corrected 2%-density int4 candidate
scored 196/963 before feedback. A fresh GPU reload returned A on all 64
inspected development examples (12/64 correct). This confirms a constant-A
collapse in that sample; the full 963 predictions were not saved.

The corrected package reloaded independently at 196/963 before feedback and
204/963 afterwards, exactly matching the in-memory evaluation. Dense Qwen on
the same 963 rows scored 737/963. Four-bit embedding quantization alone scored
713/963. No development prompt exceeded the 512-token limit (maximum 308),
and the compressed evaluation reported zero mapped-out-of-vocabulary tokens.

In the corrected package, layer 0's query projection retained 2.0089% of
weights and had relative Frobenius error 0.9380; the MLP down projection
retained 2.0006% and had error 0.9501. Both decoded tensors were finite.
These unweighted matrix errors are diagnostics, not task-loss estimates.
They support investigating signal loss from extreme compression, but they do
not rule out other implementation errors.

An earlier selector bug used unsorted `topk` output for rows with different
budgets. The fix has a regression test; this corrected Qwen candidate was run
with that fix. The subsequent sequential-prefix calibration candidate is a
separate experiment and its result must be recorded after it finishes.

The old NEC BoolQ setup included answer-formatted BoolQ training examples in
a mixed calibration corpus, kept validation documents separate, and supported
sequential Q/K/V, attention-output, MLP-input, and MLP-output calibration.
The archived full BoolQ score was 48.50% versus 79.36% dense, so that old
result does not establish that its calibration preserved BoolQ capability.
