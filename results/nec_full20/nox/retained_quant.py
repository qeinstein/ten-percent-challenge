"""Opt-in 8-bit per-row quantization of retained dense tensors (embed_tokens, lm_head).

The retained FP16 embedding and readout matrices cost about 525 MB raw (about
18% of a compressed Llama-2-7B package).  This module quantizes them to
symmetric per-row INT8 with FP16 row scales, stores the pair in a side file
``nec_retained_q8.safetensors`` inside the deployable package, removes the FP16
originals from the shards/index, and dequantizes transparently at load time.

Decoding is exact and deterministic: ``w_hat = code * scale`` with FP16 scales,
computed in FP32 and cast to the parameter dtype.  The quantizer rounds against
the FP16-rounded scale, so the load-time weights are exactly the weights whose
error :func:`rowwise_int8_report` measures.
"""

from __future__ import annotations

import gzip
import io
import json
from pathlib import Path

import torch

Q8_FILE = "nec_retained_q8.safetensors"
Q8_MANIFEST = "nec_retained_q8.json"
DEFAULT_NAMES = ("model.embed_tokens.weight", "lm_head.weight")


def quantize_rows_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row INT8: returns ``(codes int8 [r, c], scales fp16 [r])``."""

    if tensor.ndim != 2 or not tensor.is_floating_point():
        raise ValueError("expected a floating-point [rows, columns] tensor")
    values = tensor.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("tensor contains non-finite values")
    scales = (values.abs().amax(dim=1) / 127.0).to(torch.float16)
    scales = scales.float().clamp_min(torch.finfo(torch.float16).tiny).to(torch.float16)
    codes = torch.round(values / scales.float()[:, None]).clamp(-127, 127).to(torch.int8)
    return codes.contiguous(), scales.contiguous()


def dequantize_rows_int8(
    codes: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    return (codes.float() * scales.float()[:, None]).to(dtype)


def rowwise_int8_report(tensor: torch.Tensor) -> dict:
    """Measured relative error and bytes of the INT8 form of ``tensor``."""

    codes, scales = quantize_rows_int8(tensor)
    restored = dequantize_rows_int8(codes, scales, torch.float32)
    reference = tensor.detach().float()
    original_bytes = int(tensor.numel() * tensor.element_size())
    quantized_bytes = int(codes.numel() + scales.numel() * 2)
    return {
        "relative_frobenius_error": float((reference - restored).norm() / reference.norm().clamp_min(1e-30)),
        "max_abs_error": float((reference - restored).abs().max()),
        "original_bytes": original_bytes,
        "quantized_bytes": quantized_bytes,
        "byte_saving": original_bytes - quantized_bytes,
        "scale_coefficients": int(scales.numel()),
    }


# Self-contained loader code injected into ``load_nec_package.py``; it must not
# import ``nox`` (the loader may run before ``nox`` is importable).  Kept in
# sync with ``dequantize_rows_int8`` by tests.
LOADER_FUNCTION = '''
def _apply_retained_q8(model):
    path = ROOT / "nec_retained_q8.safetensors"
    if not path.exists():
        return
    from safetensors.torch import load_file
    tensors = load_file(str(path))
    for key in sorted(tensors):
        if not key.endswith(".q8"):
            continue
        name = key[:-3]
        weight = tensors[key].float() * tensors[name + ".scale"].float()[:, None]
        parameter = model.get_parameter(name)
        parameter.data.copy_(weight.to(device=parameter.device, dtype=parameter.dtype))
'''
LOADER_CALL = "    _apply_retained_q8(model)\n    return model\n"


def patch_loader_source(source: str) -> str:
    """Add transparent INT8 dequantization to the bundled package loader (idempotent)."""

    if "_apply_retained_q8" in source:
        return source
    marker = "def load_model("
    if source.count(marker) != 1 or not source.rstrip().endswith("    return model"):
        raise ValueError("unrecognised package loader layout")
    head, tail = source.split(marker)
    tail = tail.rstrip()[: -len("    return model")] + LOADER_CALL
    return head.rstrip("\n") + "\n" + LOADER_FUNCTION + "\n\n" + marker + tail


def _shard_path(package: Path, shard: str) -> Path:
    plain = package / shard
    if plain.exists():
        return plain
    zipped = package / (shard + ".gz")
    if zipped.exists():
        return zipped
    raise FileNotFoundError(shard)


def quantize_retained_in_package(
    package_dir, names=DEFAULT_NAMES, *, require_all: bool = False,
) -> dict:
    """Replace retained FP16 tensors in a materialized package by INT8 + FP16 row scales.

    Works for plain and gzip retained containers.  Returns a report with exact
    bytes before/after (tensor payload and files) and per-tensor relative error.
    The package loader is patched so loading dequantizes transparently.
    """

    from safetensors.torch import load as load_bytes
    from safetensors.torch import save as save_bytes
    from safetensors.torch import save_file

    package = Path(package_dir)
    if (package / Q8_FILE).exists():
        raise FileExistsError("package already contains INT8 retained tensors")
    index_path = package / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    present = [name for name in names if name in weight_map]
    if require_all and len(present) != len(tuple(names)):
        raise KeyError("requested retained tensors are absent from the package index")
    quantized: dict[str, torch.Tensor] = {}
    per_tensor = {}
    by_shard: dict[str, list[str]] = {}
    for name in present:
        by_shard.setdefault(weight_map[name], []).append(name)
    removed_bytes = 0
    for shard, shard_names in by_shard.items():
        path = _shard_path(package, shard)
        raw = path.read_bytes()
        zipped = path.suffix == ".gz"
        tensors = load_bytes(gzip.decompress(raw) if zipped else raw)
        for name in shard_names:
            tensor = tensors.pop(name)
            report = rowwise_int8_report(tensor)
            codes, scales = quantize_rows_int8(tensor)
            quantized[name + ".q8"] = codes
            quantized[name + ".scale"] = scales
            report["dtype"] = str(tensor.dtype)
            per_tensor[name] = report
            removed_bytes += report["original_bytes"]
            del weight_map[name]
        if tensors:
            payload = save_bytes(tensors, metadata={"format": "pt"})
            path.write_bytes(gzip.compress(payload, compresslevel=1, mtime=0) if zipped else payload)
        else:
            path.unlink()
    if not quantized:
        return {"quantized_tensors": [], "bytes_saved_tensor_payload": 0, "per_tensor": {}}
    save_file(quantized, str(package / Q8_FILE), metadata={"format": "pt"})
    metadata = dict(index.get("metadata", {}))
    added_bytes = sum(r["quantized_bytes"] for r in per_tensor.values())
    metadata["total_size"] = int(metadata.get("total_size", 0)) - removed_bytes
    index["metadata"] = metadata
    index["weight_map"] = weight_map
    index_path.write_text(json.dumps(index, indent=2) + "\n")
    summary = {
        "quantized_tensors": sorted(per_tensor),
        "scheme": "symmetric_per_row_int8_fp16_scales",
        "container_file": Q8_FILE,
        "bytes_removed_tensor_payload": removed_bytes,
        "bytes_added_tensor_payload": added_bytes,
        "bytes_saved_tensor_payload": removed_bytes - added_bytes,
        "scale_coefficients_added": sum(r["scale_coefficients"] for r in per_tensor.values()),
        "per_tensor": per_tensor,
    }
    (package / Q8_MANIFEST).write_text(json.dumps(summary, indent=2) + "\n")
    loader = package / "load_nec_package.py"
    if loader.exists():
        loader.write_text(patch_loader_source(loader.read_text()))
    manifest_path = package / "nec_allocation_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        manifest["retained_int8_rowwise_tensors"] = summary["quantized_tensors"]
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return summary


def apply_retained_quantization(model, root) -> list[str]:
    """Library equivalent of the injected loader function (used by tests/tools)."""

    from safetensors.torch import load_file

    path = Path(root) / Q8_FILE
    if not path.exists():
        return []
    tensors = load_file(str(path))
    applied = []
    for key in sorted(tensors):
        if key.endswith(".q8"):
            name = key[:-3]
            parameter = model.get_parameter(name)
            parameter.data.copy_(
                dequantize_rows_int8(tensors[key], tensors[name + ".scale"], torch.float32)
                .to(device=parameter.device, dtype=parameter.dtype)
            )
            applied.append(name)
    return applied
