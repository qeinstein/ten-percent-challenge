"""Model-level acceptance for bounded, gradient-free NEC construction search."""
from __future__ import annotations
import math


def construction_improves(candidate, current, *, minimum_kl_gain=1e-6):
    """Require better teacher KL with no NLL/accuracy/agreement regression.

    Both scores must cover exactly the same token identities, checked by the
    caller. Integer prediction counts avoid rounding accuracy percentages.
    """
    if candidate['scored_tokens'] != current['scored_tokens']:
        raise ValueError('construction scores use different token counts')
    keys = ('mean_nll', 'teacher_kl_nats_per_token')
    if not all(math.isfinite(float(s[k])) for s in (candidate, current) for k in keys):
        return False
    return (
        candidate['teacher_kl_nats_per_token'] < current['teacher_kl_nats_per_token'] - minimum_kl_gain
        and candidate['mean_nll'] <= current['mean_nll']
        and candidate['next_token_correct'] >= current['next_token_correct']
        and candidate['dense_agreement_correct'] >= current['dense_agreement_correct']
    )


# ---------------------------------------------------------------------------
# Paired window-block bootstrap gate (opt-in; ``construction_improves`` above is
# unchanged and remains the default for reproducibility).
#
# The count-conjunction gate on ~8k correlated tokens is noise-level: a handful
# of next-token flips can veto a candidate with clearly better KL.  The paired
# gate instead asks whether the mean per-token KL difference is below zero with
# one-sided confidence, while accuracy and teacher agreement need only be
# non-inferior within a small margin.  Whole windows (optionally sub-blocks of
# ``block_tokens`` positions) are resampled together so within-window
# correlation is respected.
# ---------------------------------------------------------------------------
import numpy as np

PAIRED_REQUIRED_KEYS = ('kl', 'nll', 'correct', 'agreement', 'window')


def arrays_from_matrices(kl, nll, correct, agreement):
    """Build gate arrays from ``[windows, tokens]`` matrices (numpy or torch)."""
    def convert(value, dtype):
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value).astype(dtype)
    kl_matrix = convert(kl, np.float64)
    windows, tokens = kl_matrix.shape
    return {
        'kl': kl_matrix.reshape(-1), 'nll': convert(nll, np.float64).reshape(-1),
        'correct': convert(correct, bool).reshape(-1),
        'agreement': convert(agreement, bool).reshape(-1),
        'window': np.repeat(np.arange(windows), tokens),
        'position': np.tile(np.arange(1, tokens + 1), windows),
    }


def _validated_arrays(arrays, name):
    missing = [key for key in PAIRED_REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f'{name} arrays are missing {missing}')
    size = len(arrays['kl'])
    for key in PAIRED_REQUIRED_KEYS + (('position',) if 'position' in arrays else ()):
        if len(arrays[key]) != size:
            raise ValueError(f'{name} array {key!r} has inconsistent length')
    return size


def _block_ids(arrays, block_tokens):
    window = np.asarray(arrays['window']).astype(np.int64)
    if block_tokens:
        if 'position' not in arrays:
            raise ValueError('block_tokens requires position arrays')
        sub = np.asarray(arrays['position']).astype(np.int64) // int(block_tokens)
        key = window * (int(sub.max()) + 1) + sub
    else:
        key = window
    return np.unique(key, return_inverse=True)[1]


def construction_improves_paired(
    candidate_arrays, baseline_arrays, *, primary='kl', mode='improvement', alpha=0.05,
    resamples=2000, seed=20240607, agreement_margin=0.002, accuracy_margin=0.002,
    kl_margin=0.0, min_gain=0.0, nll_margin=None, block_tokens=None, min_blocks=8,
):
    """Paired block-bootstrap acceptance of a candidate over a baseline.

    ``*_arrays``: dicts of equal-length 1-D arrays ``kl`` (per-token teacher KL),
    ``nll``, ``correct`` (next-token hit), ``agreement`` (top-1 equals teacher) and
    ``window`` (block identity), plus optional ``position``.  Both must describe
    identical tokens.  Margins are fractions (0.002 = 0.2 points).

    Accept iff
      * ``mode='improvement'``: the one-sided ``1-alpha`` upper bound of the mean
        candidate-minus-baseline ``primary`` ('kl' or 'nll') difference is below
        ``-min_gain``; ``mode='non_inferiority'``: that bound is <= ``kl_margin``;
      * the one-sided lower bounds of the agreement and accuracy differences are
        >= ``-margin`` (non-inferiority, not strict count conjunctions);
      * optionally the NLL upper bound <= ``nll_margin``;
      * there are at least ``min_blocks`` resampling blocks (fail closed).
    Returns a dict with ``decision``, ``reasons`` and all statistics.
    """
    if primary not in ('kl', 'nll'):
        raise ValueError("primary must be 'kl' or 'nll'")
    if mode not in ('improvement', 'non_inferiority'):
        raise ValueError("mode must be 'improvement' or 'non_inferiority'")
    if not 0 < alpha < 0.5 or resamples < 1:
        raise ValueError('invalid alpha or resamples')
    size = _validated_arrays(candidate_arrays, 'candidate')
    if _validated_arrays(baseline_arrays, 'baseline') != size:
        raise ValueError('construction scores use different token counts')
    if not np.array_equal(candidate_arrays['window'], baseline_arrays['window']):
        raise ValueError('candidate and baseline window identities differ')
    if 'position' in candidate_arrays and 'position' in baseline_arrays and not np.array_equal(
            candidate_arrays['position'], baseline_arrays['position']):
        raise ValueError('candidate and baseline token positions differ')
    blocks = _block_ids(candidate_arrays, block_tokens)
    n_blocks = int(blocks.max()) + 1 if size else 0
    counts = np.bincount(blocks, minlength=n_blocks).astype(np.float64)
    resampled = np.random.default_rng(seed).integers(
        0, max(n_blocks, 1), size=(resamples, max(n_blocks, 1)))
    denominator = counts[resampled].sum(axis=1) if n_blocks else None

    statistics, finite = {}, True
    for key in ('kl', 'nll', 'correct', 'agreement'):
        a = np.asarray(candidate_arrays[key]).astype(np.float64)
        b = np.asarray(baseline_arrays[key]).astype(np.float64)
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            finite = False
            statistics[key] = {'finite': False}
            continue
        diff = a - b
        sums = np.bincount(blocks, weights=diff, minlength=n_blocks)
        boot = sums[resampled].sum(axis=1) / denominator if n_blocks else np.array([np.nan])
        statistics[key] = {
            'finite': True, 'candidate_mean': float(a.mean()) if size else float('nan'),
            'baseline_mean': float(b.mean()) if size else float('nan'),
            'mean_difference': float(diff.mean()) if size else float('nan'),
            'upper_bound': float(np.quantile(boot, 1 - alpha)),
            'lower_bound': float(np.quantile(boot, alpha)),
        }
    usable = finite and n_blocks >= min_blocks
    if usable:
        bound = statistics[primary]['upper_bound']
        primary_ok = bound < -min_gain if mode == 'improvement' else bound <= kl_margin
        agreement_ok = statistics['agreement']['lower_bound'] >= -agreement_margin
        accuracy_ok = statistics['correct']['lower_bound'] >= -accuracy_margin
        nll_ok = (nll_margin is None or primary == 'nll'
                  or statistics['nll']['upper_bound'] <= nll_margin)
    else:
        primary_ok = agreement_ok = accuracy_ok = nll_ok = False
    reasons = []
    if not finite:
        reasons.append('non_finite_scores')
    if n_blocks < min_blocks:
        reasons.append(f'too_few_blocks:{n_blocks}<{min_blocks}')
    if usable:
        for ok, reason in ((primary_ok, f'{primary}_not_improved_or_not_non_inferior'),
                           (agreement_ok, 'agreement_inferior'),
                           (accuracy_ok, 'accuracy_inferior'), (nll_ok, 'nll_inferior')):
            if not ok:
                reasons.append(reason)
    return {
        'gate': 'paired_block_bootstrap', 'decision': usable and not reasons,
        'reasons': reasons, 'primary_ok': primary_ok, 'agreement_ok': agreement_ok,
        'accuracy_ok': accuracy_ok, 'nll_ok': nll_ok, 'usable': usable,
        'primary': primary, 'mode': mode, 'alpha': alpha, 'resamples': resamples,
        'seed': seed, 'n_tokens': size, 'n_blocks': n_blocks, 'block_tokens': block_tokens,
        'min_blocks': min_blocks, 'agreement_margin': agreement_margin,
        'accuracy_margin': accuracy_margin, 'kl_margin': kl_margin, 'min_gain': min_gain,
        'nll_margin': nll_margin, 'statistics': statistics,
    }


def evaluate_construction_gate(kind, candidate_summary, current_summary,
                               candidate_arrays=None, current_arrays=None, **paired_kwargs):
    """Dispatch ``legacy`` (count conjunction) or ``paired`` acceptance; returns (bool, details)."""
    if kind == 'legacy':
        return construction_improves(candidate_summary, current_summary), {'gate': 'legacy'}
    if kind != 'paired':
        raise ValueError("gate must be 'legacy' or 'paired'")
    if candidate_arrays is None or current_arrays is None:
        raise ValueError('the paired gate requires per-token arrays for both scores')
    details = construction_improves_paired(candidate_arrays, current_arrays, **paired_kwargs)
    return details['decision'], details


def compare_evaluations(reference, candidate, gate='legacy', reference_arrays=None,
                        candidate_arrays=None, **paired_kwargs):
    """Non-degradation of a candidate evaluation report against a reference report.

    ``legacy``: the original per-metric comparisons of the student summaries
    (PPL and KL no higher, accuracy and agreement no lower).  ``paired``:
    non-inferiority via the paired block bootstrap on the saved per-token
    arrays (``mode='non_inferiority'`` unless overridden).  Returns
    ``(gates, details)`` where ``gates`` maps names to booleans.
    """
    if reference['input_token_ids_sha256'] != candidate['input_token_ids_sha256']:
        raise ValueError('quality comparisons require identical token IDs')
    if gate == 'legacy':
        a, b = reference['student'], candidate['student']
        gates = {
            **{k: b[k] <= a[k] for k in ('perplexity', 'teacher_kl_nats_per_token')},
            **{k: b[k] >= a[k] for k in ('next_token_accuracy', 'dense_top1_agreement')},
        }
        return gates, {'gate': 'legacy'}
    if gate != 'paired':
        raise ValueError("gate must be 'legacy' or 'paired'")
    if reference_arrays is None or candidate_arrays is None:
        raise ValueError('the paired gate requires per-token arrays '
                         '(evaluate with --save-token-arrays)')
    paired_kwargs.setdefault('mode', 'non_inferiority')
    details = construction_improves_paired(candidate_arrays, reference_arrays, **paired_kwargs)
    gates = {'paired_primary': details['primary_ok'],
             'paired_accuracy_non_inferior': details['accuracy_ok'],
             'paired_agreement_non_inferior': details['agreement_ok'],
             'paired_nll_non_inferior': details['nll_ok'],
             'paired_usable': details['usable']}
    return gates, details


# Release gates shared by the run scripts' CLI defaults (the CURRENT gates).
DEFAULT_MAX_PPL_EXCESS = 1.0
DEFAULT_MAX_KL = 0.5
DEFAULT_MIN_ACCURACY = 0.54
DEFAULT_MAX_ACCURACY = 0.62
DEFAULT_MIN_AGREEMENT = 0.85


def add_quality_gate_arguments(parser):
    """Register the shared absolute quality-gate arguments (defaults = current gates)."""
    group = parser.add_argument_group('quality gates')
    group.add_argument('--max-ppl-excess', type=float, default=DEFAULT_MAX_PPL_EXCESS,
                       help='Student PPL must be <= teacher PPL + this (default %(default)s).')
    group.add_argument('--max-kl', type=float, default=DEFAULT_MAX_KL,
                       help='Teacher KL (nats/token) must be <= this (default %(default)s).')
    group.add_argument('--min-accuracy', type=float, default=DEFAULT_MIN_ACCURACY,
                       help='Student next-token accuracy lower bound (default %(default)s).')
    group.add_argument('--max-accuracy', type=float, default=DEFAULT_MAX_ACCURACY,
                       help='Student next-token accuracy upper bound (default %(default)s).')
    group.add_argument('--min-agreement', type=float, default=DEFAULT_MIN_AGREEMENT,
                       help='Dense top-1 agreement lower bound (default %(default)s).')
    return group


def add_paired_gate_arguments(parser):
    """Register ``--gate`` and the paired-bootstrap options."""
    group = parser.add_argument_group('candidate acceptance gate')
    group.add_argument('--gate', choices=('legacy', 'paired'), default='legacy',
                       help='legacy: strict count conjunction (reproducible default); '
                            'paired: paired window-block bootstrap with non-inferiority '
                            'margins (needs per-token arrays).')
    group.add_argument('--paired-alpha', type=float, default=0.05)
    group.add_argument('--paired-resamples', type=int, default=2000)
    group.add_argument('--paired-seed', type=int, default=20240607)
    group.add_argument('--paired-margin-points', type=float, default=0.2,
                       help='Accuracy/agreement non-inferiority margin in percentage points.')
    group.add_argument('--paired-block-tokens', type=int, default=0,
                       help='Sub-block size inside windows for the bootstrap. Default 0 resamples whole '
                            'windows: sub-blocks of one window share a document-level effect, so '
                            'treating them as independent makes the gate anti-conservative.')
    group.add_argument('--paired-nll-margin', type=float, default=0.005,
                       help='Allowed NLL upper bound (nats/token) so perplexity cannot regress.')
    group.add_argument('--paired-min-blocks', type=int, default=8)
    group.add_argument('--paired-kl-margin', type=float, default=0.005,
                       help='Non-degradation comparisons: allowed KL upper bound (nats/token).')
    return group


def paired_kwargs_from_args(args):
    """Translate parsed ``add_paired_gate_arguments`` options to gate keyword arguments."""
    margin = args.paired_margin_points / 100.0
    return dict(alpha=args.paired_alpha, resamples=args.paired_resamples, seed=args.paired_seed,
                agreement_margin=margin, accuracy_margin=margin, kl_margin=args.paired_kl_margin,
                block_tokens=args.paired_block_tokens or None, min_blocks=args.paired_min_blocks,
                nll_margin=args.paired_nll_margin)


def quality_gates(teacher, student, args):
    """Evaluate the absolute release gates; ``args`` from ``add_quality_gate_arguments``."""
    return {
        'perplexity': student['perplexity'] <= teacher['perplexity'] + args.max_ppl_excess,
        'teacher_kl': student['teacher_kl_nats_per_token'] <= args.max_kl,
        'accuracy': args.min_accuracy <= student['next_token_accuracy'] <= args.max_accuracy,
        'agreement': student['dense_top1_agreement'] >= args.min_agreement,
    }
