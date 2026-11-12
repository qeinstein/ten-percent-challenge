"""Compare existing NEC sparse corrections with vector corrections at fixed budgets.

This is a research gate, not a production decoder. The same fixed backbone is
used in all hybrid arms; candidates that exceed the original artifact's tensor
payload, serialized size, or numerical coefficient cap cannot be selected.
"""
from __future__ import annotations

import io
import math

import torch

from nox.experimental_vector_excitation import (
    compile_vector_excitation, decode_vector_excitation,
    vector_coefficient_count, vector_payload_bytes,
)
from nox.packed_schmidt import (
    packed_schmidt_coefficient_count_from_artifact,
    packed_schmidt_payload_bytes_from_artifact, reconstruct_packed_schmidt,
)
from nox.residual_codec import codec_payload_bytes, decode_groupwise, encode_groupwise


def serialized_bytes(artifact):
    stream = io.BytesIO()
    torch.save(artifact, stream)
    return stream.tell()


def reconstruct_comparison_artifact(artifact, *, device='cpu'):
    """Decode a saved research selection, including a vector hybrid if chosen."""
    if artifact.get('codec_type') == 'experimental_schmidt_vector_hybrid_v1':
        base = reconstruct_packed_schmidt(artifact['base_artifact'], device=device)
        return base + decode_vector_excitation(artifact['vector_codec'], device=device)
    return reconstruct_packed_schmidt(artifact, device=device)


def _error(weight, decoded, x):
    target = weight.float() @ x.float()
    numerator = (decoded.float() @ x.float() - target).double().square().sum()
    denominator = target.double().square().sum().clamp_min(1e-30)
    return float(numerator / denominator)


def projection_relative_squared_error(weight, decoded, inputs):
    return _error(weight, decoded, inputs)


def _tensor_digest(tensor):
    import hashlib
    source = tensor.detach().cpu().contiguous()
    return hashlib.sha256(source.numpy().tobytes()).hexdigest()


@torch.no_grad()
def compare_vector_excitation(weight, baseline, fit_inputs, guard_inputs, test_inputs=None, *,
                              seed=20261029, training_vectors=40000, iterations=10,
                              include_direct_control=True, candidate_sink=None):
    """Freeze alternatives using fit inputs and guard only; score test after selection."""
    if baseline.get('codec_type') not in ('nec_sparse_excitation_schmidt_v1',
                                          'nec_sparse_excitation_schmidt_v2'):
        raise ValueError('baseline must be a sparse excitation artifact')
    if tuple(weight.shape) != tuple(baseline['orig_shape']):
        raise ValueError('weight and sparse artifact dimensions differ')
    inputs = (fit_inputs, guard_inputs) if test_inputs is None else (fit_inputs, guard_inputs, test_inputs)
    if any(x.ndim != 2 or x.shape[0] != weight.shape[1] or x.shape[1] == 0 for x in inputs):
        raise ValueError('input sets must be nonempty [input_features, tokens]')
    if len({_tensor_digest(x) for x in inputs}) != len(inputs):
        raise ValueError('input tensors must be distinct')
    device = weight.device
    base = baseline['base_artifact']
    decoded_base = reconstruct_packed_schmidt(base, device=device)
    decoded_baseline = reconstruct_packed_schmidt(baseline, device=device)
    cap = {
        'tensor_payload_bytes': packed_schmidt_payload_bytes_from_artifact(baseline),
        'serialized_artifact_bytes': serialized_bytes(baseline),
        'numerical_coefficients': packed_schmidt_coefficient_count_from_artifact(baseline),
    }
    base_bytes = packed_schmidt_payload_bytes_from_artifact(base)
    base_coeff = packed_schmidt_coefficient_count_from_artifact(base)
    baseline_guard = _error(weight, decoded_baseline, guard_inputs)
    candidates = []
    best_name, best_artifact, best_decoded, best_guard = 'sparse_nec', baseline, decoded_baseline, baseline_guard
    residual = weight.float() - decoded_base
    for bits, aware in ((7, False), (7, True), (8, False), (8, True)):
        codec = compile_vector_excitation(
            residual, fit_inputs, activation_aware=aware, seed=seed,
            training_vectors=training_vectors, iterations=iterations, index_bits=bits)
        decoded = decoded_base + decode_vector_excitation(codec, device=device)
        # Explicitly store the complete base to charge serialization and metadata.
        artifact = {'codec_type': 'experimental_schmidt_vector_hybrid_v1',
                    'orig_shape': tuple(weight.shape), 'base_artifact': base, 'vector_codec': codec}
        cost = {'tensor_payload_bytes': base_bytes + vector_payload_bytes(codec),
                'serialized_artifact_bytes': serialized_bytes(artifact),
                'numerical_coefficients': base_coeff + vector_coefficient_count(codec)}
        fits = all(cost[k] <= cap[k] for k in cap)
        guard_error = _error(weight, decoded, guard_inputs)
        name = f'vector_{bits}bit_' + ('activation_rms' if aware else 'weight')
        record = {'name': name, **cost, 'fits_all_caps': fits, 'guard_relative_squared_output_error': guard_error,
                  'categorical_index_count': int(weight.numel() // 4),
                  'categorical_index_bits': int(weight.numel() // 4 * bits),
                  'packed_index_bytes': int(codec['indices'].numel())}
        candidates.append(record)
        if fits and candidate_sink is not None:
            candidate_sink[name] = (artifact, decoded)
        if fits and math.isfinite(guard_error) and guard_error < best_guard:
            best_name, best_artifact, best_decoded, best_guard = name, artifact, decoded, guard_error
    # Selection is frozen here. Test is never passed to constructors or the guard gate.
    reports = {'sparse_nec': {'guard_relative_squared_output_error': baseline_guard},
               'selected': {'name': best_name, 'guard_relative_squared_output_error': best_guard}}
    if test_inputs is not None:
        reports['sparse_nec']['test_relative_squared_output_error'] = _error(weight, decoded_baseline, test_inputs)
        reports['selected']['test_relative_squared_output_error'] = _error(weight, best_decoded, test_inputs)
    direct = []
    if include_direct_control:
        for bits in (2, 3):
            codec = encode_groupwise(weight.float(), bits=bits, group_size=64,
                                     quant_grid='nonzero', scale_policy='mse')
            decoded = decode_groupwise(codec, device=device)
            # Count all logical dense entries and all numerical scales.
            numerical = int(weight.numel() + codec['group_scales'].numel())
            cost = {'tensor_payload_bytes': codec_payload_bytes(codec),
                    'serialized_artifact_bytes': serialized_bytes(codec),
                    'numerical_coefficients': numerical}
            control = {'name': f'direct_{bits}bit', **cost,
                           'fits_all_caps': all(cost[k] <= cap[k] for k in cap),
                           'fits_storage_caps': (cost['tensor_payload_bytes'] <= cap['tensor_payload_bytes']
                                                 and cost['serialized_artifact_bytes'] <= cap['serialized_artifact_bytes']),
                           'guard_relative_squared_output_error': _error(weight, decoded, guard_inputs)}
            if test_inputs is not None:
                control['test_relative_squared_output_error'] = _error(weight, decoded, test_inputs)
            direct.append(control)
    result = {
        'schema': 'nec_vector_excitation_comparison_v1', 'shape': list(weight.shape),
        'fit_tokens': int(fit_inputs.shape[1]), 'guard_tokens': int(guard_inputs.shape[1]),
        'test_tokens': None if test_inputs is None else int(test_inputs.shape[1]), 'input_tensor_sha256': {
            name: _tensor_digest(x) for name, x in zip(('fit', 'guard', 'test'), inputs)},
        'baseline_caps': cap, 'vector_candidates': candidates, 'scores': reports,
        'direct_quantization_controls': direct, 'selection_uses_test': False,
        'categorical_indexes_counted_in_physical_bytes': True,
        'categorical_indexes_counted_as_numerical_coefficients': False,
        'limitations': ['single projection, not whole model', 'base checkpoint and data provenance supplied by caller',
                        'construction guard selects candidates and is not a final test',
                        'codec is experimental and not registered in package loader'],
    }
    return result, best_artifact, best_decoded
