"""Training-only rank-one concept construction and anchored cosine readout.

Pair indices are relative to the supplied training array. Risk pairs are ordered
``[safe, harmful]`` and manipulation pairs ``[original_harmful, jailbreak]``.
The caller owns split isolation; this module never reads files or fits on queries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

EPS = 1e-12


def _states(value: np.ndarray, name: str = "states") -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) < 1:
        raise ValueError(f"{name} must have nonempty shape [samples, layers, width]")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def _indices(value: Sequence[int], size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a nonempty vector of integer indices")
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    if len(np.unique(array)) != len(array):
        raise ValueError(f"{name} contains duplicate indices")
    return array.astype(np.int64)


def _pairs(value: Sequence[Sequence[int]], size: int, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != 2 or not len(array) or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must have nonempty integer shape [pairs, 2]")
    if np.any(array < 0) or np.any(array >= size):
        raise ValueError(f"{name} contains an out-of-range index")
    if np.any(array[:, 0] == array[:, 1]):
        raise ValueError(f"{name} cannot pair a row with itself")
    if len(np.unique(array, axis=0)) != len(array):
        raise ValueError(f"{name} contains duplicate pairs")
    return array.astype(np.int64)


def rank_one_direction(differences: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the uncentered first right singular vector, with fixed sign.

    Sign follows the mean paired difference. A zero mean alignment is resolved
    by making the largest absolute coordinate positive. A rank-zero matrix has
    no identifiable concept and raises instead of returning an arbitrary vector.
    """
    delta = np.asarray(differences, dtype=np.float64)
    if delta.ndim != 2 or min(delta.shape) < 1 or not np.isfinite(delta).all():
        raise ValueError("differences must be a finite nonempty matrix")
    # The smaller Gram matrix avoids a full [samples, width] SVD while retaining
    # the exact (untruncated) leading eigensolution, as in the existing baseline.
    if delta.shape[0] <= delta.shape[1]:
        eigenvalues, eigenvectors = np.linalg.eigh(delta @ delta.T)
        leading = float(eigenvalues[-1])
        if leading <= 0:
            raise ValueError("Cannot construct a concept from a zero difference matrix")
        direction = delta.T @ eigenvectors[:, -1] / np.sqrt(leading)
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(delta.T @ delta)
        leading = float(eigenvalues[-1])
        if leading <= 0:
            raise ValueError("Cannot construct a concept from a zero difference matrix")
        direction = eigenvectors[:, -1]
    direction /= np.linalg.norm(direction)
    alignment = float(direction @ delta.mean(axis=0))
    sign_tie = abs(alignment) <= EPS * max(1.0, float(np.linalg.norm(delta.mean(axis=0))))
    if alignment < 0 and not sign_tie:
        direction = -direction
    elif sign_tie and direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    residual = float(np.linalg.norm(delta.T @ (delta @ direction) - leading * direction) / leading)
    if residual > 1e-7:
        raise ValueError("Concept eigensolution failed its residual check")
    positive_eigenvalues = np.maximum(eigenvalues, 0)
    second = float(positive_eigenvalues[-2]) if len(eigenvalues) > 1 else 0.0
    diagnostics = {
        "singular_value": float(np.sqrt(leading)),
        "second_singular_value": float(np.sqrt(second)),
        "top_energy_fraction": float(leading / positive_eigenvalues.sum()),
        "sign_alignment": float(direction @ delta.mean(axis=0)),
        "sign_tie": bool(sign_tie),
        "eigen_residual": residual,
        "pairs": len(delta),
    }
    return direction, diagnostics


@dataclass
class ConceptBank:
    layer_ids: np.ndarray
    risk_direction: np.ndarray
    manipulation_direction: np.ndarray
    safe_anchor: np.ndarray
    harmful_anchor: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = np.asarray(self.layer_ids)
        if ids.ndim != 1 or not len(ids) or ids.dtype.kind not in "iu" or np.any(ids < 1):
            raise ValueError("layer_ids must contain positive physical layer numbers")
        if len(np.unique(ids)) != len(ids):
            raise ValueError("layer_ids must be unique")
        self.layer_ids = ids.astype(np.int64)
        shape = None
        for name in ("risk_direction", "manipulation_direction", "safe_anchor", "harmful_anchor"):
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != len(ids) or array.shape[1] < 1:
                raise ValueError(f"{name} must have shape [layers, width]")
            if not np.isfinite(array).all() or (shape is not None and array.shape != shape):
                raise ValueError("Concept bank arrays must be finite and have identical shapes")
            setattr(self, name, array)
            shape = array.shape
        for name in ("risk_direction", "manipulation_direction"):
            if not np.allclose(np.linalg.norm(getattr(self, name), axis=1), 1.0, atol=1e-8):
                raise ValueError("Concept directions must have unit norm")

    def arrays(self) -> dict[str, np.ndarray]:
        """Numeric arrays suitable for ``np.savez`` without pickle."""
        return {name: getattr(self, name).copy() for name in (
            "layer_ids", "risk_direction", "manipulation_direction", "safe_anchor", "harmful_anchor"
        )}


def fit_concepts(
    states: np.ndarray,
    risk_pairs: Sequence[Sequence[int]],
    manipulation_pairs: Sequence[Sequence[int]],
    safe_indices: Sequence[int],
    original_harmful_indices: Sequence[int],
    layer_ids: Sequence[int] | None = None,
) -> ConceptBank:
    """Fit directions and reference anchors from training rows only."""
    values = _states(states)
    risk = _pairs(risk_pairs, len(values), "risk_pairs")
    manipulation = _pairs(manipulation_pairs, len(values), "manipulation_pairs")
    safe = _indices(safe_indices, len(values), "safe_indices")
    harmful = _indices(original_harmful_indices, len(values), "original_harmful_indices")
    if np.intersect1d(safe, harmful).size:
        raise ValueError("Safe and original-harmful anchors cannot share rows")
    ids = np.arange(1, values.shape[1] + 1) if layer_ids is None else np.asarray(layer_ids)
    if len(ids) != values.shape[1]:
        raise ValueError("layer_ids must match the state layer axis")
    risk_direction, manipulation_direction, details = [], [], []
    for layer in range(values.shape[1]):
        risk_vector, risk_detail = rank_one_direction(values[risk[:, 1], layer] - values[risk[:, 0], layer])
        manipulation_vector, manipulation_detail = rank_one_direction(
            values[manipulation[:, 1], layer] - values[manipulation[:, 0], layer]
        )
        risk_direction.append(risk_vector)
        manipulation_direction.append(manipulation_vector)
        details.append({"layer": int(ids[layer]), "risk": risk_detail, "manipulation": manipulation_detail,
                        "direction_cosine": float(risk_vector @ manipulation_vector)})
    return ConceptBank(
        layer_ids=ids,
        risk_direction=np.stack(risk_direction),
        manipulation_direction=np.stack(manipulation_direction),
        safe_anchor=values[safe].mean(axis=0),
        harmful_anchor=values[harmful].mean(axis=0),
        diagnostics={"layers": details, "risk_pairs": len(risk), "manipulation_pairs": len(manipulation),
                     "safe_anchor_count": len(safe), "harmful_anchor_count": len(harmful)},
    )


def anchored_cosine(states: np.ndarray, anchors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Cosine of query-minus-anchor and a direction; zero offset scores zero."""
    values = _states(states)
    anchors = np.asarray(anchors, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if anchors.shape != values.shape[1:] or directions.shape != anchors.shape:
        raise ValueError("Anchor/direction shape does not match states")
    if not np.isfinite(anchors).all() or not np.isfinite(directions).all():
        raise ValueError("Anchors and directions must be finite")
    direction_norm = np.linalg.norm(directions, axis=-1)
    if np.any(direction_norm <= EPS):
        raise ValueError("Directions cannot have zero norm")
    delta = values - anchors
    denominator = np.linalg.norm(delta, axis=-1) * direction_norm
    numerator = np.einsum("nld,ld->nl", delta, directions)
    scores = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    return np.clip(scores, -1.0, 1.0)


def concept_scores(states: np.ndarray, bank: ConceptBank) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(risk[N,L], manipulation[N,L])`` in bank.layer_ids order."""
    return (anchored_cosine(states, bank.safe_anchor, bank.risk_direction),
            anchored_cosine(states, bank.harmful_anchor, bank.manipulation_direction))
