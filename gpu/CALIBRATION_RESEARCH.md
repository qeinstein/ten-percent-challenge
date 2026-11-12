# Tool-selection calibration: findings and next experiment

The current sweep is a control. Its task-gradient allocation used an unsorted
`topk` result as though it were sorted. The selector is now fixed locally and
has a regression test. The running pod process imported the old implementation;
its results do not measure the fix.

Current calibration gathers only input-channel second moments and output-channel
gradient moments. It cannot represent correlations between input channels. It
also captures the dense trajectory once, although later compressed layers see
different activations. The current feedback test has only 16 unseen examples,
and the development task measures five-way tool selection rather than argument
generation or execution.

Next, compare the corrected selector with a sequential compiler. Capture a
bounded, diverse set of real tool-selection calibration prompts; stream block
input covariances rather than keeping every activation; process layers in model
order; and recompute downstream calibration activations after each compressed
block. Use the old NEC code's streaming covariance and low-rank correction
ideas as references, but rank candidate byte allocations on labeled tool-choice
accuracy and margin on calibration/development data. Compare covariance-only,
task-weighted, and low-rank residual variants at matched *actual package bytes*.
No parameter optimizer is required. Keep the held-out file sealed, and report
the public provenance of the current C2-derived development split.

Research: calibration distribution can materially change downstream outcomes
([Williams and Aletras, 2024](https://aclanthology.org/2024.acl-long.544/));
task-matched data is not automatically best, so compare mixtures rather than
assuming more examples guarantee improvement
([Bandari et al., 2024](https://aclanthology.org/2024.emnlp-main.1004/));
layerwise reconstruction can overfit calibration data
([Rethinking Pruning, 2024](https://aclanthology.org/2024.emnlp-main.68/));
and task-specific loss information can improve allocation in structured
pruning ([GISP, 2026](https://aclanthology.org/2026.acl-long.1653/)).

The 2%-density sequential-prefix candidate completed at 196/963 before
feedback, identical to the corrected non-sequential candidate; its reloaded
package reproduced this score. Therefore refreshing RMS alone did not help at
that density. The old NEC calibration path also used full input Gram matrices,
correlated error feedback and guarded low-rank residual corrections; the
sequential RMS experiment did not implement those components.

An additional experimental codec now fits an analytical low-rank residual in
coordinates weighted by task-input RMS and task-loss output gradients. It
stores the factors in the package and does not take optimizer steps. Unit
tests check weighted residual improvement and serialization, but neither the
proxy nor the tests prove improved tool-choice accuracy. Its intended next
test is one candidate at the current <=7% package ceiling, compared with the
same-density sparse int4 control.
