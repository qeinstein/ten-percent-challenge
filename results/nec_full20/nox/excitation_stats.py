"""Sufficient-statistics reconstruction of sparse NEC excitations.

Fixed-support amplitude solves, covariance feedback and fit-error scoring only
need the input Gram matrix ``C = X X^T``, the target cross moment
``Y X^T`` and the target energy ``||Y||^2`` summed over the fit tokens. These
are accumulated chunk by chunk, so calibration sets much larger than GPU memory
(hundreds of thousands of tokens) can be used. A small resident guard set keeps
candidate acceptance honest. Construction-only; nothing here is stored.
"""

from __future__ import annotations

import torch

from nox.sparse_schmidt import (
    _select_excitation_construction, exact_supported_normal_solve, excitation_indices,
    excitation_values, _prior_anchored_pcg,
)


class InputGram:
    """Running ``sum_t x_t x_t^T`` shared by every operator that reads the same input."""

    def __init__(self, dimension, device):
        self.gram = torch.zeros(dimension, dimension, dtype=torch.float32, device=device)
        self.count = 0

    @torch.no_grad()
    def add(self, x):
        x = x.float()
        self.gram.addmm_(x, x.T)
        self.count += x.shape[1]


class CrossMoments:
    """Running ``sum_t y_t x_t^T`` and ``sum_t ||y_t||^2`` for one operator target."""

    def __init__(self, output_dimension, input_dimension, device):
        self.cross = torch.zeros(output_dimension, input_dimension, dtype=torch.float32, device=device)
        self.energy = 0.
        self.count = 0

    @torch.no_grad()
    def add(self, x, y):
        x, y = x.float(), y.float()
        self.cross.addmm_(y, x.T)
        self.energy += float(y.double().square().sum())
        self.count += x.shape[1]


@torch.no_grad()
def statistics_fit_error_function(base, gram, cross, energy):
    """Relative fit error of ``sparse`` against ``Y - base x`` from statistics only.

    ``||S x - (Y - B x)||^2 = tr(S C S^T) - 2 <S, R X^T> + ||R||^2`` with
    ``R X^T = Y X^T - B C`` and ``||R||^2 = ||Y||^2 - 2 <B, Y X^T> + <B C, B>``.
    Accumulated in float64 to avoid cancellation.
    """
    base_gram = base @ gram
    residual_cross = cross - base_gram
    residual_energy = (energy - 2. * float((base.double() * cross.double()).sum())
                       + float((base_gram.double() * base.double()).sum()))
    denominator = max(energy, 1e-30)

    def error(sparse):
        quadratic = float(((sparse @ gram).double() * sparse.double()).sum())
        linear = float((sparse.double() * residual_cross.double()).sum())
        return (quadratic - 2. * linear + residual_energy) / denominator

    return error, residual_cross


@torch.no_grad()
def reconstruct_excitation_operator_from_statistics(
        weight, artifact, gram_holder, moments, guard_inputs, guard_targets, *,
        methods=("pcg_mse", "pcg_nonzero", "feedback_fixed", "feedback_support", "feedback_nonzero",
                 "pcg_canonical", "feedback_canonical"),
        output_bits=None, matmul_precision='highest', amplitude_solver='pcg', pcg_iterations=16,
        ridge_ratios=None, feedback_ridge_ratios=None):
    """Recompile an NEC operator from streamed statistics plus a resident guard set.

    Same candidates, budgets and train/guard acceptance as
    ``nox.sparse_schmidt.reconstruct_excitation_operator``; the fit objective
    is evaluated exactly from statistics (Euclidean output metric only).
    """
    from nox.packed_schmidt import reconstruct_packed_schmidt
    allowed = {'pcg_mse', 'pcg_nonzero', 'feedback_fixed', 'feedback_support', 'feedback_nonzero',
               'pcg_canonical', 'feedback_canonical', 'feedback_adaptive', 'feedback_adaptive_nonzero'}
    if set(methods) - allowed:
        raise ValueError('unknown excitation construction method')
    if output_bits is None:
        output_bits = int(artifact['amplitude_codec']['bits'])
        allow_precision_reallocation = False
    else:
        from nox.residual_codec import SUPPORTED_BITS
        if output_bits not in SUPPORTED_BITS:
            raise ValueError('unsupported excitation precision')
        allow_precision_reallocation = True
    gram, count = gram_holder.gram, gram_holder.count
    if moments.count != count or count < 1:
        raise ValueError('input Gram and cross moments cover different token counts')
    device = gram.device
    base = reconstruct_packed_schmidt(artifact['base_artifact'], device=device)
    fit_error, residual_cross = statistics_fit_error_function(base, gram, moments.cross, moments.energy)
    guard_x, guard_y = guard_inputs.float(), guard_targets.float()
    guard_residual = guard_y - base @ guard_x
    guard_energy = guard_y.square().sum().clamp_min(1e-30)

    def score_sparse(sparse):
        guard = float((sparse @ guard_x - guard_residual).square().sum() / guard_energy)
        return fit_error(sparse), guard

    covariance = gram / count
    variance = covariance.diagonal().clamp_min(1e-20)

    def solve_delta(ratio):
        indices = excitation_indices(artifact, device)
        prior = excitation_values(artifact, device, indices=indices)
        prior_dense = torch.zeros_like(base).scatter_(1, indices, prior)
        rhs = ((residual_cross - prior_dense @ gram) / count).gather(1, indices)
        anchoring = ratio * variance[indices]
        if amplitude_solver == 'cholesky':
            return exact_supported_normal_solve(covariance, rhs, indices, anchoring)
        if amplitude_solver != 'pcg':
            raise ValueError('unknown excitation solver')

        def dense(values):
            return torch.zeros_like(base).scatter_(1, indices, values)

        delta, _ = _prior_anchored_pcg(
            lambda values: (dense(values) @ covariance).gather(1, indices) + anchoring * values,
            rhs, variance[indices] + anchoring, iterations=pcg_iterations, tolerance=1e-4)
        return delta

    problem = {
        'device': device, 'base': base, 'variance': variance,
        'score_sparse': score_sparse, 'solve_delta': solve_delta,
        'feedback': {'fit_inputs': None, 'fit_targets': None,
                     'statistics': (gram, residual_cross, count)},
        'fit_tokens': count, 'guard_tokens': guard_x.shape[1],
    }
    best, decoded, report = _select_excitation_construction(
        problem, weight, artifact, methods=methods, output_bits=output_bits,
        allow_precision_reallocation=allow_precision_reallocation,
        matmul_precision=matmul_precision, amplitude_solver=amplitude_solver,
        ridge_ratios=ridge_ratios, feedback_ridge_ratios=feedback_ridge_ratios)
    report['statistics_streamed'] = True
    return best, decoded, report
