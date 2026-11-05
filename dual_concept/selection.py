"""Concept-space complementary layer selection using development folds only.

Each fold supplies already-frozen operating thresholds learned on that fold's
training reference rows. No threshold or concept parameter is fit here. Labels
are branch-specific: 1/0 for AUROC and -1 for auxiliary controls. ``safe_mask``
marks actually safe requests independently of those labels; an original harmful
request can be a negative example for manipulation without being safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class SelectionConfig:
    min_layers: int = 2
    max_layers: int = 4
    auroc_min: float = 0.55
    selection_fpr: float = 0.05  # Recorded protocol value; caller fits thresholds.
    false_positive_penalty: float = 1.0
    correlation_penalty: float = 0.05
    worst_coverage_weight: float = 1.0
    mean_coverage_weight: float = 0.25
    min_gain: float = 0.0
    auroc_aggregation: str = "worst"

    def __post_init__(self) -> None:
        if not 2 <= self.min_layers <= self.max_layers <= 4:
            raise ValueError("Each concept branch must select between two and four layers")
        if not 0 <= self.auroc_min <= 1 or not 0 <= self.selection_fpr < 1:
            raise ValueError("AUROC/FPR configuration is out of range")
        for name in ("false_positive_penalty", "correlation_penalty", "worst_coverage_weight",
                     "mean_coverage_weight", "min_gain"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.auroc_aggregation not in {"worst", "mean"}:
            raise ValueError("auroc_aggregation must be 'worst' or 'mean'")


@dataclass
class SelectionResult:
    layers: list[int]
    layers_by_budget: dict[int, list[int]]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann–Whitney AUROC with average ranks for tied scores."""
    y, score = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or score.shape != y.shape or not np.isfinite(score).all():
        raise ValueError("AUROC requires matching finite vectors")
    if not np.isin(y, [0, 1]).all() or not np.any(y == 0) or not np.any(y == 1):
        raise ValueError("AUROC requires both binary classes")
    order = np.argsort(score, kind="stable")
    sorted_scores = score[order]
    ranks = np.empty(len(y), dtype=np.float64)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    positive = y == 1
    count = int(positive.sum())
    return float((ranks[positive].sum() - count * (count + 1) / 2) / (count * (~positive).sum()))


def _validate_fold(fold: Mapping[str, Any], layers: int) -> dict[str, np.ndarray]:
    scores = np.asarray(fold["scores"], dtype=np.float64)
    labels = np.asarray(fold["labels"])
    safe = np.asarray(fold["safe_mask"])
    groups = np.asarray(fold["attack_groups"], dtype=str)
    thresholds = np.asarray(fold["thresholds"], dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != layers or not len(scores):
        raise ValueError("Each fold's scores must have shape [samples, layers]")
    if not np.isfinite(scores).all() or thresholds.shape != (layers,) or np.isnan(thresholds).any():
        raise ValueError("Scores must be finite and thresholds must match the layer axis")
    if labels.shape != (len(scores),) or not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError("Fold labels must be -1, 0 or 1")
    if safe.shape != labels.shape or safe.dtype.kind != "b" or groups.shape != labels.shape:
        raise ValueError("safe_mask/attack_groups must align with fold rows; safe_mask must be boolean")
    if np.any(safe & (labels == 1)):
        raise ValueError("A safe control cannot be a branch-positive example")
    if not safe.any() or not np.any(labels == 0) or not np.any(labels == 1):
        raise ValueError("Every fold needs safe controls and both branch classes")
    if np.any(groups[labels == 1] == ""):
        raise ValueError("Every positive row needs an attack group")
    safe_groups = np.asarray(fold.get("safe_groups", np.full(len(scores), "safe")), dtype=str)
    if safe_groups.shape != labels.shape or np.any(safe_groups[safe] == ""):
        raise ValueError("Safe groups must align with rows and name every safe control")
    return {"scores": scores, "labels": labels, "safe_mask": safe, "attack_groups": groups,
            "safe_groups": safe_groups, "thresholds": thresholds}


def _correlation(folds: list[dict[str, np.ndarray]]) -> np.ndarray:
    scores = np.concatenate([fold["scores"] for fold in folds], axis=0)
    centered = scores - scores.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    denominator = norms[:, None] * norms[None, :]
    correlation = np.divide(centered.T @ centered, denominator,
                            out=np.zeros_like(denominator), where=denominator > 0)
    # Two identical constants are redundant too, despite undefined Pearson r.
    constant = norms == 0
    for left in np.flatnonzero(constant):
        for right in np.flatnonzero(constant):
            correlation[left, right] = float(np.allclose(scores[:, left], scores[:, right]))
    np.fill_diagonal(correlation, 1.0)
    return np.clip(np.abs(correlation), 0.0, 1.0)


def select_layers(
    folds: Sequence[Mapping[str, Any]],
    layer_ids: Sequence[int],
    config: SelectionConfig | None = None,
) -> SelectionResult:
    """Greedily select 2–4 branch layers using grouped held-out errors.

    Candidate AUROC defaults to the worst attack group across internal folds.
    Greedy utility = weighted worst/mean group recall gain minus macro added
    safe FPR and maximum absolute response-correlation penalties. Qualifying
    layers are added until min_layers, then only while utility > min_gain.
    Ties favor higher mean AUROC and the smaller physical layer number. Fewer
    than min_layers qualifying candidates is an explicit error, never fallback.
    """
    config = config or SelectionConfig()
    ids = np.asarray(layer_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu" or len(np.unique(ids)) != len(ids) or np.any(ids < 1):
        raise ValueError("layer_ids must be unique positive physical layer numbers")
    if not folds:
        raise ValueError("At least one internal validation fold is required")
    validated = [_validate_fold(fold, len(ids)) for fold in folds]
    positive_masks, safe_masks, auc_rows, group_names = [], [], [], []
    for fold_number, fold in enumerate(validated):
        positive = []
        for group in sorted(set(fold["attack_groups"][fold["labels"] == 1])):
            mask = (fold["labels"] == 1) & (fold["attack_groups"] == group)
            positive.append(mask)
            both = mask | (fold["labels"] == 0)
            auc_rows.append([binary_auroc(fold["labels"][both], fold["scores"][both, layer])
                             for layer in range(len(ids))])
            group_names.append({"fold": fold_number, "attack_group": group})
        positive_masks.append(positive)
        safe_masks.append([fold["safe_mask"] & (fold["safe_groups"] == group)
                           for group in sorted(set(fold["safe_groups"][fold["safe_mask"]]))])
    auc = np.asarray(auc_rows)
    candidate_auc = auc.min(axis=0) if config.auroc_aggregation == "worst" else auc.mean(axis=0)
    candidates = np.flatnonzero(candidate_auc >= config.auroc_min).tolist()
    if len(candidates) < config.min_layers:
        raise ValueError(f"Only {len(candidates)} layers meet AUROC >= {config.auroc_min}; "
                         f"at least {config.min_layers} are required")
    hits = [fold["scores"] >= fold["thresholds"][None, :] for fold in validated]
    correlation = _correlation(validated)

    def coverage(selected: list[int]) -> dict[str, Any]:
        attack_rates, safe_rates = [], []
        for number, hit in enumerate(hits):
            union = hit[:, selected].any(axis=1) if selected else np.zeros(len(hit), dtype=bool)
            attack_rates.extend(float(union[mask].mean()) for mask in positive_masks[number])
            safe_rates.extend(float(union[mask].mean()) for mask in safe_masks[number])
        return {"worst_tpr": min(attack_rates), "mean_tpr": float(np.mean(attack_rates)),
                "macro_safe_fpr": float(np.mean(safe_rates)), "attack_group_tpr": attack_rates,
                "safe_group_fpr": safe_rates}

    selected: list[int] = []
    trace = []
    stop_reason = "max_layers"
    while len(selected) < min(config.max_layers, len(candidates)):
        current = coverage(selected)
        options = []
        for layer in candidates:
            if layer in selected:
                continue
            following = coverage(selected + [layer])
            worst_gain = following["worst_tpr"] - current["worst_tpr"]
            mean_gain = following["mean_tpr"] - current["mean_tpr"]
            added_fpr = following["macro_safe_fpr"] - current["macro_safe_fpr"]
            redundancy = float(correlation[layer, selected].max()) if selected else 0.0
            utility = (config.worst_coverage_weight * worst_gain + config.mean_coverage_weight * mean_gain
                       - config.false_positive_penalty * added_fpr - config.correlation_penalty * redundancy)
            options.append((utility, float(auc[:, layer].mean()), -int(ids[layer]), layer,
                            {"layer": int(ids[layer]), "utility": utility,
                             "worst_tpr_gain": worst_gain, "mean_tpr_gain": mean_gain,
                             "added_macro_safe_fpr": added_fpr, "max_abs_response_correlation": redundancy,
                             "coverage": following}))
        best = max(options, key=lambda item: item[:3])
        if len(selected) >= config.min_layers and best[0] <= config.min_gain:
            stop_reason = "no_positive_incremental_utility"
            break
        selected.append(best[3])
        trace.append({**best[4], "minimum_budget_step": len(selected) <= config.min_layers})
    if len(selected) == len(candidates) and len(selected) < config.max_layers:
        stop_reason = "eligible_candidates_exhausted"
    chosen = [int(ids[index]) for index in selected]
    return SelectionResult(
        layers=chosen,
        layers_by_budget={budget: chosen[:budget] for budget in range(config.min_layers, len(chosen) + 1)},
        diagnostics={"candidate_layers": [int(ids[index]) for index in candidates],
                     "candidate_auroc": {str(int(ids[index])): float(candidate_auc[index]) for index in range(len(ids))},
                     "mean_auroc": {str(int(ids[index])): float(auc[:, index].mean()) for index in range(len(ids))},
                     "group_auroc": auc.tolist(), "attack_groups": group_names,
                     "response_correlation": correlation.tolist(), "trace": trace,
                     "stop_reason": stop_reason, "final_coverage": coverage(selected)},
    )
