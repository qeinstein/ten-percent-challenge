"""Experimental dense vector-coded correction to an existing Schmidt backbone.

Research codec only: bitpacked indexes, FP16 codebook/scales, independent
activation-aware fitting, and strict tensor-payload/coefficient accounting.
It is not registered in the NEC package loader and offers no native kernel.
"""
from __future__ import annotations

import itertools

import torch

from nox.residual_codec import _pack_codes, _unpack_codes

CODEC_TYPE = 'experimental_vector_excitation_v1'


def _nearest(vectors, centers, batch_size=4096):
    norms = centers.square().sum(dim=1)
    choices = []
    for start in range(0, len(vectors), batch_size):
        x = vectors[start:start + batch_size]
        choices.append((norms[None, :] - 2 * x @ centers.T).argmin(dim=1))
    return torch.cat(choices)


def _fit_codebook(vectors, *, iterations, index_bits):
    # A four-level Cartesian grid is a useful deterministic starting point;
    # alternating assignment/means uses no gradients.
    grid = list(itertools.product(range(4), repeat=4))
    if index_bits == 7:
        # A symmetric checkerboard subset includes 128 distinct 4-vectors.
        grid = [indices for indices in grid if sum(indices) % 2 == 0]
    levels = (-3., -1., 1., 3.)
    centers = torch.tensor([[levels[i] for i in indices] for indices in grid],
                           dtype=vectors.dtype, device=vectors.device) * .45
    size = 1 << index_bits
    for _ in range(iterations):
        ids = _nearest(vectors, centers)
        counts = torch.bincount(ids, minlength=size)
        sums = torch.zeros_like(centers).index_add_(0, ids, vectors)
        keep = counts > 0
        centers[keep] = sums[keep] / counts[keep, None]
    return centers.to(torch.float16).float()


@torch.no_grad()
def compile_vector_excitation(residual, fit_inputs, *, activation_aware, seed=20261029,
                              training_vectors=40000, iterations=10, group_size=64,
                              index_bits=8):
    """Fit a vector-coded residual on weights and fit inputs; never reads guard data.

    Each 4-vector uses one 7- or 8-bit index. A learned codebook contains 128
    or 256 FP16 4-vectors; every 64 original values have one FP16 scale. With
    activation awareness the stored FP16 per-column input scale is derived
    from the *fit* inputs. No index or other side information is free.
    """
    if residual.ndim != 2 or fit_inputs.ndim != 2 or fit_inputs.shape[0] != residual.shape[1]:
        raise ValueError('incompatible matrix and input shapes')
    rows, columns = residual.shape
    if group_size <= 0 or columns % group_size or group_size % 4:
        raise ValueError('columns must be divisible by group_size, itself divisible by four')
    if index_bits not in (7, 8):
        raise ValueError('index_bits must be 7 or 8')
    if fit_inputs.shape[1] < 1 or training_vectors < 1 or iterations < 1:
        raise ValueError('positive fit samples, vector sample count and iterations required')
    source = residual.detach().float()
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(fit_inputs).all()):
        raise ValueError('inputs and residual must be finite')
    if activation_aware:
        rms = fit_inputs.float().square().mean(dim=1).sqrt()
        rms = (rms / rms.mean().clamp_min(1e-12)).clamp(.25, 4.)
        input_scale = rms.to(torch.float16)
        work = source * input_scale.float()[None, :]
    else:
        input_scale = None
        work = source
    groups = work.reshape(-1, group_size)
    scales = groups.square().mean(dim=1).sqrt().clamp_min(torch.finfo(torch.float16).tiny)
    scales = scales.to(torch.float16)
    vectors = (groups / scales.float()[:, None]).reshape(-1, 4)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    samples = torch.randperm(len(vectors), generator=generator)[:min(training_vectors, len(vectors))]
    centers = _fit_codebook(vectors[samples.to(vectors.device)], iterations=iterations,
                            index_bits=index_bits)
    codes = _nearest(vectors, centers).to(torch.uint8)
    artifact = {
        'codec_type': CODEC_TYPE, 'shape': (int(rows), int(columns)),
        'group_size': int(group_size), 'vector_size': 4, 'index_bits': index_bits,
        'indices': _pack_codes(codes, index_bits).cpu(), 'codebook': centers.half().cpu(),
        'scales': scales.cpu(), 'input_scale': None if input_scale is None else input_scale.cpu(),
        'activation_aware': bool(activation_aware),
    }
    return artifact


def vector_payload_bytes(artifact):
    return sum(int(t.numel() * t.element_size()) for t in
               (artifact['indices'], artifact['codebook'], artifact['scales'], artifact['input_scale'])
               if t is not None)


def vector_coefficient_count(artifact):
    """Numerical codebook and scale entries, excluding discrete index choices."""
    return sum(int(t.numel()) for t in
               (artifact['codebook'], artifact['scales'], artifact['input_scale']) if t is not None)


@torch.no_grad()
def decode_vector_excitation(artifact, *, device='cpu'):
    if artifact.get('codec_type') != CODEC_TYPE:
        raise ValueError('unknown vector excitation codec')
    rows, columns = map(int, artifact['shape'])
    group_size = int(artifact['group_size'])
    if columns % group_size or group_size % 4:
        raise ValueError('invalid group dimensions')
    codebook = artifact['codebook'].to(device=device, dtype=torch.float32)
    bits = int(artifact['index_bits'])
    if bits not in (7, 8):
        raise ValueError('invalid vector index width')
    ids = _unpack_codes(artifact['indices'].to(device=device), rows * columns // 4, bits).long()
    scales = artifact['scales'].to(device=device, dtype=torch.float32)
    if codebook.shape != (1 << bits, 4) or ids.numel() != rows * columns // 4 or scales.numel() != rows * columns // group_size:
        raise ValueError('vector codec payload shape mismatch')
    result = (codebook[ids].reshape(-1, group_size) * scales[:, None]).reshape(rows, columns)
    input_scale = artifact.get('input_scale')
    if input_scale is not None:
        input_scale = input_scale.to(device=device, dtype=torch.float32)
        if input_scale.shape != (columns,) or not bool((input_scale > 0).all()):
            raise ValueError('invalid vector codec input scale')
        result = result / input_scale[None, :]
    return result
