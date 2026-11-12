"""Position-aware, per-token evaluation helpers shared by the NEC evaluators.

The historical protocol (context 2048, stride 512, loss masked to the last 512
targets) scores almost exclusively tokens at positions 1536-2047.  Downstream
tasks live at positions ~20-300.  These helpers expose per-token arrays and
position-bucketed summaries so both regimes can be reported, and so paired
window-block bootstraps can be computed offline from the saved arrays.

Everything here is pure torch/numpy and testable on CPU with tiny tensors.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DEFAULT_POSITION_BUCKETS = ((0, 64), (64, 256), (256, 1024), (1024, None))
DEFAULT_BUCKET_SPEC = "0-63,64-255,256-1023,1024+"
TOKEN_ARRAY_FLOAT_KEYS = ("teacher_nll", "student_nll", "kl")
TOKEN_ARRAY_FLAG_KEYS = ("teacher_correct", "student_correct", "agreement")
TOKEN_ARRAY_INT_KEYS = ("position", "window")


def parse_position_buckets(spec):
    """Parse ``"0-63,64-255,256-1023,1024+"`` into half-open (low, high) pairs."""
    if spec is None or spec is True:
        spec = DEFAULT_BUCKET_SPEC
    buckets = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part.endswith("+"):
            buckets.append((int(part[:-1]), None))
        else:
            low, high = part.split("-")
            buckets.append((int(low), int(high) + 1))
    if not buckets:
        raise ValueError("at least one position bucket is required")
    previous_high = 0
    for low, high in buckets:
        if low < 0 or (high is not None and high <= low) or low < previous_high:
            raise ValueError(f"position buckets must be ordered and disjoint: {buckets}")
        previous_high = high if high is not None else float("inf")
    return tuple(buckets)


def bucket_label(low, high):
    return f"{low}+" if high is None else f"{low}-{high - 1}"


def scored_positions(length, target_length):
    """Token positions (index inside the context) that a window scores.

    Matches the labels-shift-left convention: position 0 is never predicted and
    only the trailing ``target_length`` tokens are scored.
    """
    first = max(length - target_length, 1)
    return torch.arange(first, length)


def window_input_ids(ids, window, bos_id=None):
    """Return the ``[1, L]`` token block for a window specification.

    Short windows set ``prepend_bos`` so every window starts with BOS.
    """
    block = ids[:, window["begin"]:window["end"]]
    if window.get("prepend_bos"):
        if bos_id is None:
            raise ValueError("a window with prepend_bos requires a BOS id")
        bos = torch.full((block.shape[0], 1), int(bos_id), dtype=block.dtype, device=block.device)
        block = torch.cat([bos, block], dim=1)
    return block


def window_length(window):
    return window["end"] - window["begin"] + (1 if window.get("prepend_bos") else 0)


def build_group_batch(ids, group, bos_id=None):
    """Build ``(input_ids, labels, mask, positions, window_rows)`` for equal-length windows.

    ``labels``/``mask``/``positions`` have shape ``[B, L-1]`` (predictions of tokens
    1..L-1).  ``positions[b, i]`` is the context position of the predicted token.
    """
    input_ids = torch.cat([window_input_ids(ids, w, bos_id) for w in group], dim=0)
    length = input_ids.shape[1]
    targets = input_ids.clone()
    for row, window in enumerate(group):
        if window["target_length"] < length:
            targets[row, :-window["target_length"]] = -100
    labels = targets[:, 1:]
    positions = torch.arange(1, length, device=input_ids.device).expand_as(labels)
    return input_ids, labels, labels != -100, positions


def group_equal_length_windows(windows, batch_size):
    groups = []
    for window in windows:
        length = window_length(window)
        if (not groups or len(groups[-1]) >= batch_size
                or length != window_length(groups[-1][0])):
            groups.append([])
        groups[-1].append(window)
    return groups


def per_token_metrics(teacher_logits, student_logits, labels):
    """Per-token NLL/KL/correctness for ``[N, V]`` logits and ``[N]`` labels."""
    teacher = teacher_logits.float()
    student = student_logits.float()
    log_teacher = F.log_softmax(teacher, dim=-1)
    log_student = F.log_softmax(student, dim=-1)
    rows = torch.arange(labels.numel(), device=labels.device)
    teacher_top1 = teacher.argmax(dim=-1)
    student_top1 = student.argmax(dim=-1)
    kl = F.kl_div(log_student, log_teacher, reduction="none", log_target=True).sum(dim=-1)
    return {
        "teacher_nll": (-log_teacher[rows, labels]).double(),
        "student_nll": (-log_student[rows, labels]).double(),
        "kl": kl.double(),
        "teacher_correct": teacher_top1 == labels,
        "student_correct": student_top1 == labels,
        "agreement": student_top1 == teacher_top1,
    }


def concatenate_token_arrays(parts):
    """Concatenate lists of per-chunk dicts of numpy arrays."""
    if not parts:
        raise ValueError("no token arrays to concatenate")
    return {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}


def to_numpy_arrays(metrics, positions, window_index):
    arrays = {key: metrics[key].detach().cpu().numpy().astype(np.float64)
              for key in TOKEN_ARRAY_FLOAT_KEYS}
    arrays.update({key: metrics[key].detach().cpu().numpy().astype(bool)
                   for key in TOKEN_ARRAY_FLAG_KEYS})
    arrays["position"] = positions.detach().cpu().numpy().astype(np.int64)
    arrays["window"] = np.full(arrays["kl"].shape, int(window_index), dtype=np.int64)
    return arrays


def bucket_report(arrays, buckets=DEFAULT_POSITION_BUCKETS):
    """Bucket NLL/KL/agreement/accuracy by position inside the context."""
    position = np.asarray(arrays["position"])
    report = {}
    for low, high in buckets:
        selected = position >= low
        if high is not None:
            selected &= position < high
        count = int(selected.sum())
        entry = {"position_low": low, "position_high": None if high is None else high - 1,
                 "scored_tokens": count}
        if count:
            teacher_nll = float(np.mean(arrays["teacher_nll"][selected], dtype=np.float64))
            student_nll = float(np.mean(arrays["student_nll"][selected], dtype=np.float64))
            entry.update({
                "teacher_mean_nll": teacher_nll,
                "teacher_perplexity": math.exp(min(teacher_nll, 700)),
                "student_mean_nll": student_nll,
                "student_perplexity": math.exp(min(student_nll, 700)),
                "teacher_kl_nats_per_token": float(np.mean(arrays["kl"][selected], dtype=np.float64)),
                "teacher_next_token_accuracy": float(np.mean(arrays["teacher_correct"][selected])),
                "student_next_token_accuracy": float(np.mean(arrays["student_correct"][selected])),
                "dense_top1_agreement": float(np.mean(arrays["agreement"][selected])),
            })
        report[bucket_label(low, high)] = entry
    return report


def save_token_arrays(path, arrays):
    """Write compressed per-token arrays (float16 NLL/KL, uint8 flags)."""
    payload = {}
    for key in TOKEN_ARRAY_FLOAT_KEYS:
        payload[key] = np.asarray(arrays[key]).astype(np.float16)
    for key in TOKEN_ARRAY_FLAG_KEYS:
        payload[key] = np.asarray(arrays[key]).astype(np.uint8)
    payload["position"] = np.asarray(arrays["position"]).astype(np.int32)
    payload["window"] = np.asarray(arrays["window"]).astype(np.int32)
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **payload)
    temporary.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_token_arrays(path):
    """Load arrays written by :func:`save_token_arrays` as float64/bool/int64."""
    with np.load(path) as data:
        arrays = {key: data[key].astype(np.float64) for key in TOKEN_ARRAY_FLOAT_KEYS}
        arrays.update({key: data[key].astype(bool) for key in TOKEN_ARRAY_FLAG_KEYS})
        arrays.update({key: data[key].astype(np.int64) for key in TOKEN_ARRAY_INT_KEYS})
    return arrays


def student_gate_arrays(arrays):
    """Map saved arrays to the names used by the paired gate."""
    return {"kl": arrays["kl"], "nll": arrays["student_nll"],
            "correct": arrays["student_correct"], "agreement": arrays["agreement"],
            "window": arrays["window"], "position": arrays["position"]}


def load_report_token_arrays(report):
    """Gate arrays referenced by an evaluation report, hash-verified; ``None`` if absent."""
    info = report.get("token_arrays")
    if not info:
        return None
    path = Path(info["path"])
    if not path.is_file():
        return None
    if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError(f"per-token arrays {path} differ from the hash recorded in the report")
    return student_gate_arrays(load_token_arrays(path))
