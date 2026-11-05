"""Standardized linear logits with a group-lasso penalty per physical layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


def _matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty matrix")
    return array


def build_feature_matrix(
    risk_scores: np.ndarray,
    manipulation_scores: np.ndarray,
    risk_layers: Sequence[int],
    manipulation_layers: Sequence[int],
    layer_ids: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate risk then manipulation; return X, physical layers, branches.

    Overlapping physical layers retain both coordinates but share a penalty
    group in fit_sparse_fusion. Layer numbers are physical (one-based).
    """
    risk, manipulation = _matrix(risk_scores, "risk_scores"), _matrix(manipulation_scores, "manipulation_scores")
    ids = np.asarray(layer_ids)
    if risk.shape != manipulation.shape or ids.shape != (risk.shape[1],):
        raise ValueError("Score arrays and layer_ids must align")
    if ids.dtype.kind not in "iu" or np.any(ids < 1) or len(np.unique(ids)) != len(ids):
        raise ValueError("layer_ids must be unique positive physical layer numbers")
    positions = {int(layer): index for index, layer in enumerate(ids)}
    requested = []
    for name, layers in (("risk", risk_layers), ("manipulation", manipulation_layers)):
        array = np.asarray(layers)
        if array.ndim != 1 or not len(array) or array.dtype.kind not in "iu" or len(np.unique(array)) != len(array):
            raise ValueError(f"{name}_layers must be a nonempty unique integer vector")
        if any(int(layer) not in positions for layer in array):
            raise ValueError(f"Unknown {name} layer")
        requested.append([int(layer) for layer in array])
    first, second = requested
    features = np.concatenate((risk[:, [positions[layer] for layer in first]],
                               manipulation[:, [positions[layer] for layer in second]]), axis=1)
    return (features, np.asarray(first + second, dtype=np.int64),
            np.asarray(["risk"] * len(first) + ["manipulation"] * len(second)))


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    tail = np.exp(-np.abs(logits))
    return np.where(logits >= 0, 1.0 / (1.0 + tail), tail / (1.0 + tail))


@dataclass(frozen=True)
class FusionConfig:
    regularization: float = 0.01
    max_iter: int = 10000
    tol: float = 1e-7

    def __post_init__(self) -> None:
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative")
        if not isinstance(self.max_iter, int) or self.max_iter < 1 or not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("max_iter/tol must be positive")


@dataclass
class SparseFusionModel:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    feature_layers: np.ndarray
    regularization: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("mean", "scale", "coef"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=np.float64))
        self.feature_layers = np.asarray(self.feature_layers, dtype=np.int64)
        if self.coef.ndim != 1 or not len(self.coef) or any(
            array.shape != self.coef.shape for array in (self.mean, self.scale, self.feature_layers)
        ):
            raise ValueError("Fusion parameter arrays must be matching nonempty vectors")
        if not all(np.isfinite(array).all() for array in (self.mean, self.scale, self.coef)) or np.any(self.scale <= 0):
            raise ValueError("Fusion numeric parameters must be finite with positive scales")
        if not np.isfinite(self.intercept) or not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("Invalid fusion intercept/regularization")
        if np.any(self.feature_layers < 1):
            raise ValueError("Feature groups must be positive physical layer numbers")

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        values = _matrix(features, "features")
        if values.shape[1] != len(self.coef):
            raise ValueError("Feature width differs from the fitted model")
        return ((values - self.mean) / self.scale) @ self.coef + self.intercept

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Positive-class probabilities, shape [samples]."""
        return _sigmoid(self.decision_function(features))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"mean": self.mean.copy(), "scale": self.scale.copy(), "coef": self.coef.copy(),
                "intercept": np.asarray(self.intercept), "feature_layers": self.feature_layers.copy(),
                "regularization": np.asarray(self.regularization)}


def fit_sparse_fusion(
    features: np.ndarray,
    labels: np.ndarray,
    feature_layers: Sequence[int],
    regularization: float = 0.01,
    max_iter: int = 10000,
    tol: float = 1e-7,
    *,
    config: FusionConfig | None = None,
) -> SparseFusionModel:
    """Fit mean BCE + lambda * sum_l ||w_l||_2 using proximal gradient.

    Standardization uses only supplied training features (population std; constant
    columns use scale=1). The intercept is unpenalized. Group sizes are not
    reweighted, matching the report's sum of layer norms. Diagnostics expose
    convergence; a caller should not freeze a nonconverged candidate silently.
    """
    settings = config or FusionConfig(regularization, max_iter, tol)
    x = _matrix(features, "features")
    y, layer_ids = np.asarray(labels), np.asarray(feature_layers)
    if y.shape != (len(x),) or not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2:
        raise ValueError("Fusion training requires both binary classes")
    if layer_ids.shape != (x.shape[1],) or layer_ids.dtype.kind not in "iu" or np.any(layer_ids < 1):
        raise ValueError("feature_layers must name a physical layer for every coordinate")
    groups = [np.flatnonzero(layer_ids == layer) for layer in np.unique(layer_ids)]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale <= 1e-12] = 1.0
    z = (x - mean) / scale
    augmented = np.column_stack((z, np.ones(len(z))))
    # Global smooth-loss Lipschitz bound; backtracking also checks each step.
    lipschitz = max(float(np.linalg.norm(augmented, ord=2) ** 2 / (4 * len(z))), 1e-12)
    step = 1.0 / lipschitz
    parameters = np.zeros(z.shape[1] + 1)
    parameters[-1] = np.log(y.mean() / (1 - y.mean()))

    def smooth(theta: np.ndarray) -> float:
        logits = augmented @ theta
        return float(np.mean(np.logaddexp(0.0, logits) - y * logits))

    def penalty(theta: np.ndarray) -> float:
        return settings.regularization * sum(float(np.linalg.norm(theta[group])) for group in groups)

    objective = smooth(parameters) + penalty(parameters)
    history = [objective]
    converged = False
    stationarity = np.inf
    for iteration in range(1, settings.max_iter + 1):
        current_smooth = smooth(parameters)
        gradient = augmented.T @ (_sigmoid(augmented @ parameters) - y) / len(y)
        for _ in range(60):
            proposed = parameters - step * gradient
            for group in groups:
                norm = float(np.linalg.norm(proposed[group]))
                if norm <= step * settings.regularization:
                    proposed[group] = 0.0
                else:
                    proposed[group] *= 1 - step * settings.regularization / norm
            delta = proposed - parameters
            candidate_smooth = smooth(proposed)
            bound = current_smooth + float(gradient @ delta) + float(delta @ delta) / (2 * step)
            if candidate_smooth <= bound + 1e-12:
                break
            step *= 0.5
        else:
            raise RuntimeError("Sparse fusion backtracking failed")
        objective = candidate_smooth + penalty(proposed)
        stationarity = float(np.linalg.norm(delta) / step)
        parameters = proposed
        history.append(objective)
        if stationarity <= settings.tol * max(1.0, float(np.linalg.norm(parameters))):
            converged = True
            break
    group_norms = {str(int(layer)): float(np.linalg.norm(parameters[np.flatnonzero(layer_ids == layer)]))
                   for layer in np.unique(layer_ids)}
    return SparseFusionModel(
        mean=mean, scale=scale, coef=parameters[:-1], intercept=float(parameters[-1]),
        feature_layers=layer_ids, regularization=settings.regularization,
        diagnostics={"converged": converged, "iterations": iteration, "objective": objective,
                     "bce": smooth(parameters), "group_penalty": penalty(parameters),
                     "proximal_gradient_norm": stationarity, "group_norms": group_norms,
                     "active_layers": [int(layer) for layer, norm in group_norms.items() if norm > 0],
                     "objective_history": history},
    )
