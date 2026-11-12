"""Sensitivity-driven byte allocation across operators at a fixed total budget.

Model: distortion of an operator falls by a factor of four per extra bit per
weight (6 dB per bit), so its measured teacher KL at the current rate b0 is
``KL_i * 4**-(b_i - b0)``. Minimizing the sum under a fixed total number of
stored bits gives the reverse water-filling rule

    b_i - b0 = log4(r_i) - sum_j n_j log4(r_j) / sum_j n_j,  r_i = KL_i / n_i,

where ``n_i`` is the operator's weight count. Single-operator KL measurements
ignore interactions between operators, so the plan is a first-order guide and
must be confirmed by full-model evaluation.
"""

from __future__ import annotations

import math


def plan_operator_multipliers(sensitivity, weight_counts, *, base_bits_per_weight=2.6,
                              min_delta_bits=-.6, max_delta_bits=.8, floor_kl=1e-6):
    """Return ``{key: multiplier}`` of relative bit budgets (mean weighted by size is 1).

    ``sensitivity`` maps operator keys to measured KL when only that operator is
    compressed; ``weight_counts`` maps the same keys to dense weight counts.
    The delta-bit clamp keeps every operator within a range where the
    rate-distortion model is plausible.
    """
    if set(sensitivity) != set(weight_counts) or not sensitivity:
        raise ValueError("sensitivity and weight counts must describe the same non-empty operator set")
    if base_bits_per_weight <= 0 or min_delta_bits >= 0 or max_delta_bits <= 0:
        raise ValueError("invalid allocation bit limits")
    total = float(sum(weight_counts.values()))
    log_ratio = {key: math.log(max(float(sensitivity[key]), floor_kl) / weight_counts[key], 4)
                 for key in sensitivity}
    delta = {}
    offset = sum(weight_counts[key] * log_ratio[key] for key in log_ratio) / total
    for key in log_ratio:
        delta[key] = min(max_delta_bits, max(min_delta_bits, log_ratio[key] - offset))
    # Clamping breaks the zero-sum constraint; restore it by shifting the
    # unclamped operators so total stored bits are unchanged.
    for _ in range(50):
        imbalance = sum(weight_counts[key] * delta[key] for key in delta) / total
        if abs(imbalance) < 1e-9:
            break
        free = [key for key in delta
                if (imbalance > 0 and delta[key] > min_delta_bits)
                or (imbalance < 0 and delta[key] < max_delta_bits)]
        if not free:
            break
        free_total = float(sum(weight_counts[key] for key in free))
        for key in free:
            delta[key] = min(max_delta_bits, max(min_delta_bits, delta[key] - imbalance * total / free_total))
    return {key: (base_bits_per_weight + delta[key]) / base_bits_per_weight for key in delta}


def predicted_total_kl(sensitivity, weight_counts, multipliers, *, base_bits_per_weight=2.6):
    """Sum of single-operator KLs predicted by the 4**-bits model for a plan."""
    return sum(
        float(sensitivity[key]) * 4.0 ** (-(multipliers[key] - 1.) * base_bits_per_weight)
        for key in sensitivity)
