"""Independent-segment attention for packed calibration windows.

Short calibration windows pack several BOS-prefixed documents. Treating them as
one causal sequence lets later documents attend to earlier ones and shifts their
RoPE positions. These helpers run attention per segment so each document sees
only itself with positions restarting at zero.
"""

from __future__ import annotations

import torch


def segment_column_ranges(segment_starts, window_length, window_indices):
    """Chunk-local column ranges ``[(begin, end), ...]`` for the given windows.

    ``segment_starts[w]`` lists the segment start offsets inside window ``w``
    (the first is 0). Windows with one segment collapse to a single range.
    """
    ranges = []
    for local, window in enumerate(window_indices):
        starts = list(segment_starts[window]) if segment_starts is not None else [0]
        if not starts or starts[0] != 0:
            raise ValueError("segment starts must begin at 0")
        ends = starts[1:] + [window_length]
        if any(b <= a for a, b in zip(starts, ends)):
            raise ValueError("segment starts must be strictly increasing within the window")
        base = local * window_length
        ranges.extend((base + a, base + b) for a, b in zip(starts, ends))
    return ranges


def has_independent_segments(segment_starts, count=None):
    if segment_starts is None:
        return False
    return any(len(starts) > 1 for starts in (segment_starts if count is None else segment_starts[:count]))


def segmented_attention(attention, q, k, v, ranges, window_length):
    """Run ``attention(q, k, v, seq_len)`` independently over each column range.

    Full windows (range length == ``window_length``) are batched in one call when
    they are contiguous; other segments run individually with ``seq_len`` equal
    to the segment length.
    """
    if all(b - a == window_length for a, b in ranges):
        return attention(q, k, v, window_length)
    output = torch.empty_like(q)
    for begin, end in ranges:
        output[:, begin:end] = attention(q[:, begin:end], k[:, begin:end], v[:, begin:end], end - begin)
    return output
