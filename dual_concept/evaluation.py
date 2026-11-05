"""Frozen safety-only calibration and count-based risk detection metrics."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np

from .selection import binary_auroc


def calibrate_threshold(safe_logits: np.ndarray, target_fpr: float = 0.05) -> dict[str, Any]:
    """Most permissive representable >= threshold under a finite-sample budget.

    Ties are handled conservatively: at most floor(n * target_fpr) calibration
    examples are flagged. This empirical budget is not a test-distribution bound.
    """
    scores = np.asarray(safe_logits, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Calibration needs a nonempty finite vector of safe logits")
    if not np.isfinite(target_fpr) or not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must lie in [0, 1)")
    budget = math.floor(len(scores) * target_fpr)
    boundary = np.sort(scores)[::-1][budget]
    threshold = float(np.nextafter(boundary, np.inf))
    if not np.isfinite(threshold):
        raise ValueError("Cannot represent a finite threshold above the calibration boundary")
    false_positives = int(np.count_nonzero(scores >= threshold))
    if false_positives > budget:
        raise RuntimeError("Calibration exceeded its false-positive budget")
    return {"threshold": threshold, "score_domain": "logit", "decision_rule": ">=",
            "target_fpr": float(target_fpr), "n_safe": len(scores), "fp_budget": budget,
            "observed_fp": false_positives, "observed_fpr": false_positives / len(scores)}


def binary_metrics(labels: np.ndarray, logits: np.ndarray, threshold: float) -> dict[str, Any]:
    labels, logits = np.asarray(labels), np.asarray(logits, dtype=np.float64)
    if labels.ndim != 1 or logits.shape != labels.shape or not len(labels):
        raise ValueError("Evaluation requires matching nonempty vectors")
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(logits).all() or not np.isfinite(threshold):
        raise ValueError("Invalid evaluation labels, scores or threshold")
    positive, predicted = labels == 1, logits >= threshold
    tp = int((positive & predicted).sum())
    fp = int((~positive & predicted).sum())
    fn, tn = int((positive & ~predicted).sum()), int((~positive & ~predicted).sum())
    return {"n": len(labels), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "tpr": tp / (tp + fn) if tp + fn else None,
            "fpr": fp / (fp + tn) if fp + tn else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "accuracy": (tp + tn) / len(labels),
            "auroc": binary_auroc(labels, logits) if len(np.unique(labels)) == 2 else None}


def grouped_metrics(records: Sequence[Mapping[str, Any]], logits: np.ndarray,
                    threshold: float) -> dict[str, Any]:
    from .data import canonical_family
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (len(records),):
        raise ValueError("Records and scores must align")
    labels = np.asarray([row["label"] for row in records])
    result = {"overall": binary_metrics(labels, scores, threshold), "by_kind": {}, "by_attack_family": {}}
    for column, key in (("kind", "by_kind"), ("attack_family", "by_attack_family")):
        values = np.asarray([canonical_family(row.get(column)) if column == "attack_family" else row[column]
                             for row in records])
        for name in sorted(set(values) - {""}):
            mask = values == name
            result[key][name] = binary_metrics(labels[mask], scores[mask], threshold)
    result["exposure"] = sorted({row["exposure"] for row in records})
    return result
