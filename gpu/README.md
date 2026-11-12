# Qwen tool-selection compression experiment

This experiment compresses Qwen2.5-7B-Instruct with task-activation-aware sparse ternary weights and a five-choice tool-selection head. It does not use the earlier NEC codec or copy the competitor's compressor. It uses public C2 tool-selection examples from Bobliuuu's repository, with query-disjoint calibration, development, feedback and internal held-out splits. Source revision and SHA-256 hashes are in `prepared/manifest.json`. The public source has previously been scored and pretraining contamination cannot be excluded; the local hash is not independent evidence of secrecy or a prior public commitment.

No model weights are trained. Compression keeps high activation-weighted coordinates within each group and stores packed ternary values with analytical scales. Feedback stores per-tool running mean hidden-state vectors and adds bounded score corrections; it does not change the model weights, store full annotated examples, or edit prompts. Whether this aggregate prototype approach satisfies the organizer's no-RAG clarification requires their judgment. The test evaluates choosing a tool from four candidates or none; it does not evaluate argument generation, real API execution, or all model capability.

## Prepared data

The pinned C2 calibration and development JSONL files are in `data_sources`; the script `prepare_data.py` generated `prepared/manifest.json` and the four corresponding splits, removing duplicate queries across splits. Do not inspect or evaluate `locked.jsonl` during development. The automatic final evaluator only opens it if the fixed development gates pass. Local hashes and single-use files are process controls, not isolation from a person with filesystem access or proof against prior exposure. C2 provenance and upstream licensing must be reviewed before redistribution.

## GPU run

Use an A40 with CUDA PyTorch, Python, `gpu/requirements.txt`, adequate host RAM, and disk for the ~15 GB Qwen checkpoint plus output. From the repository root:

```bash
python gpu/one_run.py --output /workspace/qwen-direct --total-seconds 10500 --reserve-seconds 3000 --density 0.02
```

The run measures the original model on development data; calibrates on 256 independent task requests with 32 sampled positions each; compresses all linear layers; measures the compressed model; applies one annotation per selected tool; measures unseen examples and old-task retention; and saves the feedback state inside the counted package. It streams activation RMS on GPU, batches inference, compresses matrix chunks on GPU, and overlaps gzip output with subsequent work. The default retains 3 of every 128 weights in each group, not 2.00% exactly. Full vocabulary is preserved by default.

If size, development accuracy, feedback generalization or retention fails, the final held-out exam remains untouched. Otherwise the final phase freezes and hashes the package, exclusively claims the test, scores the original and independently reloaded compressed package before and after feedback, and reports a paired bootstrap interval. Failure or timeout after the exclusive claim is recorded and is not silently retried. The final point-estimate test is not proof of statistical equivalence, and the organizer determines acceptance. The package contains custom loader/code, tokenizer, config, compressed tensors, output head and feedback state; Python, PyTorch and Transformers are excluded.

The program bounds subprocess runtime, but it cannot shut down the provider's machine or billing; stop the instance or use a provider-side limit. Three-hour completion, Qwen accuracy, 4% package size, VRAM fit, and the actual GPU speedups have not been verified. Model download, installation and final reload also consume time. A more aggressive candidate may fail at 2.3% retained; there is no trained recovery mechanism or automatic accuracy-preserving fallback. The final 426-example held-out split has limited statistical power. The current compression pipeline reconstructs dense weights at runtime; disk reduction is not a demonstrated inference-memory or latency improvement.

## Local verification

`python3 -m unittest discover -s tests -v` tests packing, calibration-related selection, feedback semantics and the exam gate. A small SmolLM2 CPU workflow is only a smoke test; the tiny model and small test do not demonstrate Qwen accuracy. GPU integration remains untested until the GPU is supplied.
