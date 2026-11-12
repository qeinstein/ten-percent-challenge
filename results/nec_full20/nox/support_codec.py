"""Lossless entropy coder for 1-bit excitation support masks (opt-in container codec).

The logical artifact is unchanged: a boolean support mask.  This module only
changes how that mask is *serialized*.  A raw bitmap costs exactly one bit per
weight; sparse supports with non-uniform column (or row) density have less
entropy than that, and gzip recovers only a fraction of it.

Scheme (``support_rans_v1``)
----------------------------
The mask is cut into independent streams (columns or rows; ``orientation``).
Each stream stores its density class ``c = round(density * (classes-1))`` (one
byte) and is coded, 8 mask bits at a time, with an interleaved 32-bit rANS
coder (16-bit renormalisation, 15-bit probabilities) under an i.i.d. Bernoulli
model of density ``c / (classes-1)``.  All streams are encoded/decoded in
lock step, so the Python loop runs over stream *length / 8* with every step
vectorised across streams.  ``orientation='best'`` encodes both orientations
and keeps the smaller; if neither beats the raw bitmap the bitmap is stored.
Round-trip is verified exactly by the caller-facing helpers.

Side information per stream: 1 byte class, 2 bytes word count, 4 bytes final
rANS state (7 bytes), i.e. ``7 / (stream_length / 8)`` of overhead (about 0.5%
for a 4096 x 11008 mask coded column-wise).  All of it is counted in
:func:`support_codec_payload_bytes`.
"""

from __future__ import annotations

import math

import numpy as np
import torch

CODEC_ID = "support_rans_v1"
PROB_BITS = 15
_STATE_LOW = 1 << 16
_SYMBOL_BITS = 8
_PROB_ONE = 1 << PROB_BITS
_TABLE_CACHE: dict = {}


def _tables(classes: int):
    """Frequency/cumulative/slot tables for byte symbols under Bernoulli(p_c)."""

    cached = _TABLE_CACHE.get(classes)
    if cached is not None:
        return cached
    values = np.arange(256)
    ones = np.array([bin(int(v)).count("1") for v in values], dtype=np.float64)
    freq = np.zeros((classes, 256), dtype=np.int64)
    for c in range(classes):
        p = min(max(c / (classes - 1), 0.002), 0.998)
        probability = (p ** ones) * ((1.0 - p) ** (_SYMBOL_BITS - ones))
        f = np.maximum(1, np.floor(probability * _PROB_ONE + 0.5)).astype(np.int64)
        f[int(np.argmax(f))] += _PROB_ONE - int(f.sum())
        if f.min() < 1 or int(f.sum()) != _PROB_ONE:
            raise RuntimeError("support codec frequency table construction failed")
        freq[c] = f
    cum = np.zeros((classes, 257), dtype=np.int64)
    cum[:, 1:] = np.cumsum(freq, axis=1)
    slot_symbol = np.zeros((classes, _PROB_ONE), dtype=np.uint8)
    for c in range(classes):
        slot_symbol[c] = np.repeat(np.arange(256, dtype=np.uint8), freq[c])
    cached = (freq.astype(np.uint64), cum[:, :256].astype(np.uint64), slot_symbol)
    _TABLE_CACHE[classes] = cached
    return cached


def _as_numpy_mask(mask) -> np.ndarray:
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError("support mask must be a 2-D matrix")
    return mask.astype(bool, copy=False)


def _uint8_tensor(array: np.ndarray, dtype: str) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(array.astype(dtype)).view(np.uint8).copy())


def _from_uint8(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    return tensor.cpu().numpy().view(dtype)


def _encode_streams(streams: np.ndarray, classes: int):
    """rANS-encode ``streams`` [S, N] (bool) with per-stream density classes."""

    count, length = streams.shape
    density = streams.mean(axis=1) if length else np.zeros(count)
    cls = np.rint(density * (classes - 1)).astype(np.int64)
    padded = (length + 7) // 8 * 8
    if padded != length:
        streams = np.pad(streams, ((0, 0), (0, padded - length)))
    symbols = np.packbits(streams, axis=1, bitorder="little")  # [S, T]
    steps = symbols.shape[1]
    freq, cum, _ = _tables(classes)
    state = np.full(count, _STATE_LOW, dtype=np.uint64)
    words = np.zeros((steps, count), dtype=np.uint16)
    emitted = np.zeros((steps, count), dtype=bool)
    shift = np.uint64(32 - PROB_BITS)
    prob_bits = np.uint64(PROB_BITS)
    for t in range(steps - 1, -1, -1):
        symbol = symbols[:, t].astype(np.int64)
        f = freq[cls, symbol]
        c = cum[cls, symbol]
        need = state >= (f << shift)
        emitted[t] = need
        words[t] = (state & np.uint64(0xFFFF)).astype(np.uint16)
        state = np.where(need, state >> np.uint64(16), state)
        state = ((state // f) << prob_bits) + (state % f) + c
    flat = np.ascontiguousarray(words.T)[np.ascontiguousarray(emitted.T)]
    counts = emitted.sum(axis=0)
    return cls.astype(np.uint8), flat, counts, state.astype(np.uint32)


def _decode_streams(cls, flat, counts, final_state, length: int, classes: int) -> np.ndarray:
    count = cls.shape[0]
    steps = (length + 7) // 8
    freq, cum, slot_symbol = _tables(classes)
    cls = cls.astype(np.int64)
    state = final_state.astype(np.uint64)
    pointer = np.concatenate([[0], np.cumsum(counts.astype(np.int64))[:-1]])
    flat = np.concatenate([flat.astype(np.uint64), np.zeros(1, dtype=np.uint64)])
    limit = flat.shape[0] - 1
    out = np.zeros((count, steps), dtype=np.uint8)
    mask = np.uint64(_PROB_ONE - 1)
    prob_bits = np.uint64(PROB_BITS)
    low = np.uint64(_STATE_LOW)
    for t in range(steps):
        slot = (state & mask).astype(np.int64)
        symbol = slot_symbol[cls, slot]
        f = freq[cls, symbol]
        c = cum[cls, symbol]
        state = f * (state >> prob_bits) + slot.astype(np.uint64) - c
        need = state < low
        word = flat[np.minimum(pointer, limit)]
        state = np.where(need, (state << np.uint64(16)) | word, state)
        pointer = pointer + need
        out[:, t] = symbol
    bits = np.unpackbits(out, axis=1, bitorder="little")[:, :length]
    if count and not np.array_equal(pointer, np.cumsum(counts.astype(np.int64))):
        raise ValueError("corrupt support stream")
    return bits.astype(bool)


def _streams_of(mask: np.ndarray, orientation: str) -> np.ndarray:
    return mask.T if orientation == "column" else mask


def _encode_oriented(mask: np.ndarray, orientation: str, classes: int) -> dict:
    streams = np.ascontiguousarray(_streams_of(mask, orientation))
    cls, flat, counts, state = _encode_streams(streams, classes)
    count_dtype = "<u2" if (counts.size == 0 or int(counts.max()) < 65536) else "<u4"
    return {
        "codec": CODEC_ID,
        "orientation": orientation,
        "shape": tuple(int(v) for v in mask.shape),
        "classes": int(classes),
        "count_width": 2 if count_dtype == "<u2" else 4,
        "stream_classes": torch.from_numpy(cls.copy()),
        "stream_word_counts": _uint8_tensor(counts, count_dtype),
        "stream_final_states": _uint8_tensor(state, "<u4"),
        "stream_words": _uint8_tensor(flat, "<u2"),
    }


def bitmap_bytes(mask) -> int:
    """Raw 1-bit-per-weight cost of ``mask``."""

    mask = _as_numpy_mask(mask)
    return (mask.size + 7) // 8


def encode_support_mask(mask, *, orientation: str = "best", classes: int = 64) -> dict:
    """Entropy-code a boolean support mask; never larger than the raw bitmap.

    ``orientation``: ``'column'`` / ``'row'`` code each column / row as one
    stream; ``'best'`` tries both.  ``classes`` is the number of density
    classes (>= 2).
    """

    if orientation not in ("best", "column", "row"):
        raise ValueError("orientation must be 'best', 'column' or 'row'")
    if not 2 <= int(classes) <= 256:
        raise ValueError("classes must lie in [2, 256]")
    mask = _as_numpy_mask(mask)
    candidates = []
    for name in (("column", "row") if orientation == "best" else (orientation,)):
        candidates.append(_encode_oriented(mask, name, int(classes)))
    best = min(candidates, key=support_codec_payload_bytes)
    if support_codec_payload_bytes(best) >= bitmap_bytes(mask):
        best = {
            "codec": CODEC_ID, "orientation": "bitmap",
            "shape": tuple(int(v) for v in mask.shape),
            "bitmap": torch.from_numpy(np.packbits(mask.reshape(-1), bitorder="little").copy()),
        }
    return best


def decode_support_mask(blob: dict) -> torch.Tensor:
    """Exact inverse of :func:`encode_support_mask`; returns a bool tensor [rows, cols]."""

    if blob.get("codec") != CODEC_ID:
        raise ValueError(f"unsupported support codec: {blob.get('codec')!r}")
    rows, columns = map(int, blob["shape"])
    if blob["orientation"] == "bitmap":
        bits = np.unpackbits(blob["bitmap"].cpu().numpy(), bitorder="little")[: rows * columns]
        return torch.from_numpy(bits.astype(bool).reshape(rows, columns))
    orientation = blob["orientation"]
    count, length = (columns, rows) if orientation == "column" else (rows, columns)
    classes = int(blob["classes"])
    cls = blob["stream_classes"].cpu().numpy()
    if cls.shape[0] != count:
        raise ValueError("support codec stream count does not match its shape")
    counts = _from_uint8(blob["stream_word_counts"], "<u2" if int(blob["count_width"]) == 2 else "<u4")
    states = _from_uint8(blob["stream_final_states"], "<u4")
    words = _from_uint8(blob["stream_words"], "<u2")
    streams = _decode_streams(cls, words, counts, states, length, classes)
    return torch.from_numpy(np.ascontiguousarray(streams.T if orientation == "column" else streams))


def support_codec_payload_bytes(blob: dict) -> int:
    """Exact serialized tensor bytes of a support blob (metadata excluded)."""

    return int(sum(
        value.numel() * value.element_size()
        for value in blob.values() if torch.is_tensor(value)
    ))


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def support_entropy_bound_bits(mask, orientation: str = "column") -> float:
    """Total bits of the per-stream i.i.d. Bernoulli entropy bound.

    ``sum_s N_s * H(density_s)`` for streams cut along ``orientation``: the
    ideal cost of the model this codec implements (before side information).
    """

    mask = _as_numpy_mask(mask)
    streams = _streams_of(mask, orientation)
    density = streams.mean(axis=1)
    return float((_binary_entropy(density) * streams.shape[1]).sum())


def compress_artifact_support(artifact: dict, *, orientation: str = "best", classes: int = 64) -> dict:
    """Return a copy of a sparse-excitation artifact with ``support_bits`` entropy coded.

    The logical artifact is unchanged; :func:`restore_artifact_support` rebuilds
    the identical ``support_bits`` tensor (verified bit-for-bit here).  Byte
    accounting in ``packed_schmidt_payload_bytes_from_artifact`` charges the
    coded size.  The coded artifact must be restored before reconstruction.
    """

    if artifact.get("support_bits") is None:
        raise ValueError("artifact has no support_bits to compress")
    from nox.sparse_schmidt import excitation_mask

    mask = excitation_mask(artifact, "cpu").numpy()
    blob = encode_support_mask(mask, orientation=orientation, classes=classes)
    blob["support_layout"] = artifact.get("support_layout", "row_major")
    coded = {key: value for key, value in artifact.items() if key != "support_bits"}
    coded["support_codec"] = blob
    restored = restore_artifact_support(coded)
    if not torch.equal(restored["support_bits"], artifact["support_bits"].cpu()):
        raise RuntimeError("lossless support compression verification failed")
    return coded


def restore_artifact_support(artifact: dict) -> dict:
    """Inverse of :func:`compress_artifact_support` (no-op for raw artifacts)."""

    blob = artifact.get("support_codec")
    if blob is None:
        return artifact
    mask = decode_support_mask(blob).numpy()
    layout = blob.get("support_layout", "row_major")
    if layout == "row_major":
        flat = mask.reshape(-1)
    elif layout == "column_major":
        flat = mask.T.reshape(-1)
    else:
        raise ValueError("unknown excitation support layout")
    restored = {key: value for key, value in artifact.items() if key != "support_codec"}
    restored["support_bits"] = torch.from_numpy(np.packbits(flat, bitorder="little").copy())
    return restored
