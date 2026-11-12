"""Pure diagnostics for lm-eval multiple-choice logs (BoolQ yes/no behaviour).

Accuracy alone hides a compressed model that collapsed to "always no": these
report the predicted-"no" fraction, probability mass assigned to the two answer
tokens, and the threshold-free AUC of the yes-minus-no margin against labels.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np


def _first_float(value):
    while isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty response entry")
        value = value[0]
    return float(value)


def _rank_auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties; ``None`` if one class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def yes_no_diagnostic(logprob_no, logprob_yes, labels):
    """Summarise BoolQ answer-token behaviour.

    Arguments are equal-length sequences of log-likelihoods of the continuations
    " no" / " yes" and gold labels (1 = yes).  Returns:
    ``fraction_predicted_no``, ``mean_probability_mass`` (mean of P(yes)+P(no)),
    ``mean_p_yes``, ``mean_p_no``, ``auc_yes_minus_no_margin``, ``accuracy``,
    ``label_yes_fraction`` and ``samples``.
    """
    no = np.asarray(logprob_no, dtype=np.float64)
    yes = np.asarray(logprob_yes, dtype=np.float64)
    gold = np.asarray(labels).astype(int)
    if not (len(no) == len(yes) == len(gold)) or len(no) == 0:
        raise ValueError("yes/no diagnostics need equal, non-empty inputs")
    if not (np.isfinite(no).all() and np.isfinite(yes).all()):
        raise ValueError("non-finite log-likelihoods")
    predicted_yes = yes > no  # ties resolve to "no", like argmax over [no, yes]
    p_yes, p_no = np.exp(yes), np.exp(no)
    return {
        "samples": int(len(no)),
        "fraction_predicted_no": float(np.mean(~predicted_yes)),
        "mean_probability_mass": float(np.mean(p_yes + p_no)),
        "mean_p_yes": float(np.mean(p_yes)),
        "mean_p_no": float(np.mean(p_no)),
        "auc_yes_minus_no_margin": _rank_auc(yes - no, gold == 1),
        "accuracy": float(np.mean(predicted_yes == (gold == 1))),
        "label_yes_fraction": float(np.mean(gold == 1)),
    }


def extract_boolq_logprobs(samples):
    """Parse lm-eval ``samples['boolq']`` rows (choices ``["no", "yes"]``).

    Returns ``(logprob_no, logprob_yes, labels)``.
    """
    no, yes, labels = [], [], []
    for sample in samples:
        responses = sample.get("resps") or sample.get("filtered_resps")
        if responses is None or len(responses) != 2:
            raise ValueError("BoolQ sample must have exactly two choice responses")
        no.append(_first_float(responses[0]))
        yes.append(_first_float(responses[1]))
        label = sample.get("target", sample.get("doc", {}).get("label"))
        labels.append(int(label))
    return no, yes, labels


def sample_log_records(samples):
    """Compact per-sample records (no prompt text) for ``<task>_samples.jsonl.gz``."""
    records = []
    for sample in samples:
        records.append({
            "doc_id": sample.get("doc_id"), "target": sample.get("target"),
            "resps": sample.get("resps"), "filtered_resps": sample.get("filtered_resps"),
            **{k: sample[k] for k in ("acc", "acc_norm", "exact_match") if k in sample},
        })
    return records


def write_sample_logs(samples, path):
    """Write one JSON object per line, gzip-compressed; returns the record count."""
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    records = sample_log_records(samples)
    with gzip.open(temporary, "wt") as stream:
        for record in records:
            stream.write(json.dumps(record, default=lambda v: v.item() if hasattr(v, "item") else str(v)) + "\n")
    temporary.replace(path)
    return len(records)


def boolq_diagnostic_from_samples(samples):
    no, yes, labels = extract_boolq_logprobs(samples)
    result = yes_no_diagnostic(no, yes, labels)
    if any(isinstance(v, float) and not math.isfinite(v) for v in result.values()):
        raise ValueError("non-finite BoolQ diagnostic")
    return result
