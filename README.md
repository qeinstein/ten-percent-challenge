# Tool-selection compression pilot

`experiments/tool_prune.py` runs an exploratory CPU-only calibration-aware pruning and ternary quantization control on the locally cached SmolLM2-135M checkpoint. It is **not** the NEC Schmidt/MPO plus sparse-excitation implementation. It does not implement feedback learning and is not a challenge submission.

Run with Python, PyTorch, NumPy and Transformers installed:

```sh
HF_HUB_DISABLE_PROGRESS_BARS=1 OMP_NUM_THREADS=4 python3 experiments/tool_prune.py --calibration 32 --test 32 --batch 4 --keeps 0.5,0.1,0.02,0.005
```

The model path in the script points to the cached revision on this machine. Four-choice examples are generated from a small synthetic template set using separate seeds; overlap in wording is possible, so this is not an independently locked exam. The model is a base language model and its low diagnostic accuracy makes these numbers unsuitable as evidence of tool-use retention.

The archive counts packed state-dictionary tensors, including duplicated tied embedding/head entries; the reported FP16 denominator also counts both entries. The denominator is **not** the unique-parameter model size. Tokenizer, configuration, code, and runtime libraries are excluded. The model is reconstructed densely at runtime. Do not treat the reported ratios as full-deployment or challenge ratios. `torch.load` should only be used with trusted artifacts.

The old NEC compiler at `/home/fluxx/Workspace/research/nox/sparse_schmidt.py` is the intended next method. A useful follow-up requires real task activations, actual packed MPO and sparse residuals, tied-weight-safe accounting, an instruction-capable small model, and development/locked splits with genuinely distinct examples.

The current Qwen work is the separate [direct weight-space pipeline](gpu/README.md). It uses our own synthetic tool-selection task and activation-aware sparse ternary coding; it does not invoke the older NEC codec or the competitor's task data. Its GPU implementation remains unverified.
