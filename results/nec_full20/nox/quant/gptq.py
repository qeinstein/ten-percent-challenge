"""RTN and GPTQ (Hessian error feedback) with group-wise quantization.

Convention: W has shape (d_out, d_in) and the layer computes y = W x, so the
calibration Hessian H = E[x x^T] has shape (d_in, d_in). The objective is
||(W - W_hat) X||_F^2 = tr((W - W_hat) H (W - W_hat)^T) up to the token count.

Defaults (symmetric grid, max-abs scale) reproduce the original quantizer and
are a *handicapped* baseline: they waste one of 2**b codes and never search the
group scale (at 3 bits relative MSE on Gaussian g64 is 0.063 versus 0.033 for
``quant_grid='nonzero', scale_policy='mse'``; 2 bits: 0.52 versus 0.11).  Use
the opt-in ``quant_grid`` / ``scale_policy`` options for a fair dense control.
"""

import torch

from nox.residual_codec import grid_limits, mse_group_scales


def bits_per_weight(bits, group_size, scale_bits=16):
    """Payload bits per weight including one fp16 scale per group."""
    return bits + scale_bits / group_size


def _check_options(quant_grid, scale_policy):
    if quant_grid not in ("symmetric", "nonzero") or scale_policy not in ("maxabs", "mse"):
        raise ValueError("unknown quant_grid or scale_policy")


def _group_scale(group, bits, quant_grid, scale_policy, fp16_scales, mse_fractions, mse_iterations):
    """Scale for ``group`` of shape [rows, group_size] -> [rows, 1]."""
    if quant_grid == "symmetric" and scale_policy == "maxabs":
        scale = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / max(2 ** (bits - 1) - 1, 1)
    elif scale_policy == "mse":
        scale = mse_group_scales(
            group.unsqueeze(1), bits=bits, quant_grid=quant_grid,
            fractions=mse_fractions, iterations=mse_iterations,
        ).reshape(-1, 1).clamp(min=1e-8)
    else:
        _, qmax = grid_limits(bits, quant_grid)
        scale = group.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    if fp16_scales:
        scale = scale.to(torch.float16).float().clamp(min=torch.finfo(torch.float16).tiny)
    return scale


def _quantize_column(value, scale, bits, quant_grid):
    if quant_grid == "nonzero":
        qmin, _ = grid_limits(bits, "nonzero")
        codes = ((value / scale - qmin) * .5).round().clamp(0, (1 << bits) - 1)
        return (codes * 2 + qmin) * scale
    qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(value / scale), qmin, qmax) * scale


def rtn_quantize(weight, bits=4, group_size=128, quant_grid="symmetric", scale_policy="maxabs",
                 fp16_scales=False, mse_fractions=(.50, .65, .80, .95, 1.0), mse_iterations=3):
    """Round-to-nearest, groups along d_in for each output row.

    ``quant_grid='nonzero'`` uses all 2**b even-spaced zero-free levels and
    ``scale_policy='mse'`` searches the group scale (see module docstring).
    """
    _check_options(quant_grid, scale_policy)
    w = weight.float()
    d_out, d_in = w.shape
    assert d_in % group_size == 0, "d_in must be a multiple of group_size"
    groups = w.reshape(d_out, d_in // group_size, group_size)
    if quant_grid == "symmetric" and scale_policy == "maxabs" and not fp16_scales:
        scale = groups.abs().amax(dim=2, keepdim=True).clamp(min=1e-8) / max(2 ** (bits - 1) - 1, 1)
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        return (torch.clamp(torch.round(groups / scale), qmin, qmax) * scale).reshape(d_out, d_in)
    flat = groups.reshape(-1, group_size)
    scale = _group_scale(flat, bits, quant_grid, scale_policy, fp16_scales, mse_fractions, mse_iterations)
    return _quantize_column(flat, scale, bits, quant_grid).reshape(d_out, d_in)


@torch.no_grad()
def gptq_quantize(weight, hessian, bits=4, group_size=128, damp=0.01, act_order=False,
                  block_size=128, quant_grid="symmetric", scale_policy="maxabs",
                  fp16_scales=False, mse_fractions=(.50, .65, .80, .95, 1.0), mse_iterations=3):
    """GPTQ: quantize columns in order, pushing each column's error onto later ones.

    Scales for each group are fixed from the error-updated weights when the
    group is first reached, as in the reference implementation.  Updates inside
    a block are applied eagerly and propagated to later blocks lazily (once per
    block), as in the reference ``blocksize`` scheme.

    ``act_order=True`` quantizes columns by descending Hessian diagonal (groups
    then run over the permuted order, so a deployment stores the permutation:
    ``2 * d_in`` bytes per matrix, negligible against ``d_out * d_in`` weights).
    ``quant_grid='nonzero'`` uses all ``2**bits`` even-spaced, zero-free levels
    and ``scale_policy='mse'`` picks each group scale by the same FP16-aware
    least-squares search as ``residual_codec.encode_groupwise``.
    ``fp16_scales`` rounds each max-abs scale to FP16 before use.
    """
    _check_options(quant_grid, scale_policy)
    w = weight.clone().float()
    d_out, d_in = w.shape
    assert d_in % group_size == 0, "d_in must be a multiple of group_size"
    assert block_size % group_size == 0, "block_size must be a multiple of group_size"
    h = hessian.clone().float().to(w.device)
    dead = torch.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    perm = None
    if act_order:
        perm = torch.argsort(torch.diag(h), descending=True)
        w, h = w[:, perm], h[perm][:, perm]
    h = h + damp * torch.mean(torch.diag(h)) * torch.eye(d_in, device=h.device)
    chol = torch.linalg.cholesky(h)
    h_inv = torch.cholesky_inverse(chol)
    u = torch.linalg.cholesky(h_inv, upper=True)

    q_out = torch.zeros_like(w)
    scale = None
    for start in range(0, d_in, block_size):
        end = min(start + block_size, d_in)
        w_blk = w[:, start:end].clone()
        err_blk = torch.zeros_like(w_blk)
        u_blk = u[start:end, start:end]
        for i in range(end - start):
            col = start + i
            if col % group_size == 0:
                group = w_blk[:, i:i + group_size]
                scale = _group_scale(group, bits, quant_grid, scale_policy, fp16_scales,
                                     mse_fractions, mse_iterations)
            wc = w_blk[:, i:i + 1]
            qc = _quantize_column(wc, scale, bits, quant_grid)
            q_out[:, col:col + 1] = qc
            e = (wc - qc) / u_blk[i, i]
            w_blk[:, i:] -= e @ u_blk[i:i + 1, i:]
            err_blk[:, i:i + 1] = e
        w[:, end:] -= err_blk @ u[start:end, end:]
    if perm is not None:
        inv = torch.argsort(perm)
        q_out = q_out[:, inv]
    return q_out
