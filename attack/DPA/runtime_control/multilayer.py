"""
Multi-layer detection via per-dimension diagonal Mahalanobis distance,
aggregated across all layers with a Ledoit-Wolf Mahalanobis score.

Calibration: collect delta vectors (h_{l+1} - h_l) at every layer from
clean examples. Compute per-dimension μ and σ per layer. Fit a
Ledoit-Wolf covariance on the [n_cal, num_layers] matrix of per-layer
diag_mahal scores.

Detection: at step 0, compute per-layer diag_mahal, z-score per layer,
aggregate via Ledoit-Wolf Mahalanobis distance. Single threshold.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from sklearn.covariance import LedoitWolf

from .controllers import RuntimeController
from .projection import get_num_layers
from .types import DecodeConfig, PreparedInputs, StepDecision, StepState


# ---------------------------------------------------------------------------
# Calibration artifact
# ---------------------------------------------------------------------------
@dataclass
class MultiLayerCalibration:
    artifact_id: str
    model_name: str
    num_layers: int
    hidden_dim: int
    n_calibration: int
    target_fpr: float
    detection_threshold: float

    # Per-layer delta statistics — stored as numpy arrays
    # layer_delta_mu[l]: shape [hidden_dim]
    # layer_delta_sigma[l]: shape [hidden_dim]
    layer_delta_mu: List[np.ndarray]
    layer_delta_sigma: List[np.ndarray]

    # Per-layer scalar score statistics
    layer_score_mu: np.ndarray   # [num_layers]
    layer_score_sigma: np.ndarray  # [num_layers]

    # 32-dim aggregation (Ledoit-Wolf)
    agg_mu: np.ndarray        # [num_layers]
    agg_precision: np.ndarray  # [num_layers, num_layers]

    # Clean score distribution for reference
    clean_agg_scores: np.ndarray  # [n_calibration]


def _compute_diag_mahal_score(delta: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Per-dimension z-score, then L2 norm."""
    z = (delta - mu) / sigma
    return float(np.linalg.norm(z))


# ---------------------------------------------------------------------------
# Calibration function
# ---------------------------------------------------------------------------
def build_multilayer_calibration(
    model,
    tokenizer,
    clean_examples: List[dict],
    build_model_inputs_fn,
    decode_config: DecodeConfig,
    target_fpr: float = 0.10,
    max_examples: int = 50,
    model_name: Optional[str] = None,
) -> MultiLayerCalibration:
    """
    Calibrate multi-layer detection from clean examples.
    Collects delta vectors at all layers during prefill (step 0).

    If ``model_name`` is not provided, it is derived from
    ``model.config._name_or_path``.
    """
    num_layers = get_num_layers(model)
    device = next(model.parameters()).device
    max_cal = min(len(clean_examples), max_examples)
    examples = clean_examples[:max_cal]

    print(f"[LCF] Calibrating on {max_cal} clean examples, {num_layers} layers")

    # Collect deltas at all layers
    layer_deltas = {l: [] for l in range(num_layers)}

    model.eval()
    with torch.inference_mode():
        for ex in examples:
            inputs, _ = build_model_inputs_fn(
                tokenizer,
                ex["instruction"],
                ex.get("input", ""),
                decode_config.prompt_template,
            )
            input_ids = inputs["input_ids"].to(device)

            out = model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            hs = out.hidden_states  # tuple of (num_layers+1,) tensors

            for l in range(num_layers):
                h_curr = hs[l + 1][:, -1, :].float()
                h_prev = hs[l][:, -1, :].float()
                delta = (h_curr - h_prev).squeeze(0).cpu().numpy()
                layer_deltas[l].append(delta)

    # Compute per-layer statistics
    hidden_dim = layer_deltas[0][0].shape[0]
    layer_delta_mu = []
    layer_delta_sigma = []
    score_matrix = np.zeros((max_cal, num_layers))

    for l in range(num_layers):
        D = np.stack(layer_deltas[l])  # [n_cal, hidden_dim]
        mu = D.mean(axis=0)
        sigma = D.std(axis=0)
        sigma = np.maximum(sigma, 1e-8)
        layer_delta_mu.append(mu)
        layer_delta_sigma.append(sigma)

        # Per-example diag_mahal scores
        for i in range(max_cal):
            score_matrix[i, l] = _compute_diag_mahal_score(D[i], mu, sigma)

    # Per-layer score statistics
    layer_score_mu = score_matrix.mean(axis=0)
    layer_score_sigma = score_matrix.std(axis=0)
    layer_score_sigma = np.maximum(layer_score_sigma, 1e-8)

    # Z-score the score matrix
    z_matrix = (score_matrix - layer_score_mu) / layer_score_sigma  # [n_cal, num_layers]

    # Fit 32-dim Mahalanobis via Ledoit-Wolf
    agg_mu = z_matrix.mean(axis=0)
    lw = LedoitWolf().fit(z_matrix)
    agg_precision = lw.precision_

    # Compute calibration Mahalanobis scores
    clean_agg_scores = np.zeros(max_cal)
    for i in range(max_cal):
        diff = z_matrix[i] - agg_mu
        clean_agg_scores[i] = float(np.sqrt(diff @ agg_precision @ diff))

    # Threshold at target FPR
    threshold = float(np.percentile(clean_agg_scores, (1 - target_fpr) * 100))
    threshold = max(threshold, 1.0)

    # Artifact ID
    data_hash = hashlib.sha256(
        json.dumps([ex.get("instruction", "")[:50] for ex in examples]).encode()
    ).hexdigest()[:12]
    artifact_id = f"lcf_{data_hash}_{num_layers}L"

    # Resolve model_name from parameter or model config
    if model_name is None:
        name_or_path = getattr(model.config, "_name_or_path", "") or ""
        model_name = os.path.basename(name_or_path.rstrip("/")) or "unknown"

    cal = MultiLayerCalibration(
        artifact_id=artifact_id,
        model_name=model_name,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        n_calibration=max_cal,
        target_fpr=target_fpr,
        detection_threshold=threshold,
        layer_delta_mu=layer_delta_mu,
        layer_delta_sigma=layer_delta_sigma,
        layer_score_mu=layer_score_mu,
        layer_score_sigma=layer_score_sigma,
        agg_mu=agg_mu,
        agg_precision=agg_precision,
        clean_agg_scores=clean_agg_scores,
    )

    print(f"[LCF] Calibration complete:")
    print(f"  Artifact: {artifact_id}")
    print(f"  Threshold: {threshold:.3f} (at {target_fpr*100:.0f}% FPR)")
    print(f"  Clean score range: [{clean_agg_scores.min():.2f}, {clean_agg_scores.max():.2f}]")

    return cal


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------
class MultiLayerController(RuntimeController):
    """
    LCF controller: multi-layer diag_mahal detection at prefill step 0.
    Abstains with a safe message if the aggregated Mahalanobis score exceeds
    the calibrated threshold.
    """

    def __init__(
        self,
        model,
        decode_config: DecodeConfig,
        calibration: MultiLayerCalibration,
    ):
        self.model = model
        self.decode_config = decode_config
        self.calibration = calibration
        self._reset()

    def _reset(self):
        self.anomaly_detected = False
        self.detection_score = 0.0
        self.layer_z_scores = None

    def begin_sequence(self, prepared_inputs: PreparedInputs):
        self._reset()

    def modify_step(self, step_state: StepState) -> StepDecision:
        telemetry = {
            "runtime_mode": "lcf",
            "step_index": step_state.step_index,
        }

        # Only detect at step 0
        if step_state.step_index > 0:
            telemetry["detection_score"] = round(self.detection_score, 4)
            return StepDecision(logits=step_state.logits, telemetry=telemetry)

        cal = self.calibration
        hs = step_state.hidden_states

        # Compute per-layer diag_mahal
        layer_scores = np.zeros(cal.num_layers)
        for l in range(cal.num_layers):
            h_curr = hs[l + 1][:, -1, :].float()
            h_prev = hs[l][:, -1, :].float()
            delta = (h_curr - h_prev).squeeze(0).cpu().numpy()
            layer_scores[l] = _compute_diag_mahal_score(
                delta, cal.layer_delta_mu[l], cal.layer_delta_sigma[l]
            )

        # Z-score per layer
        z_scores = (layer_scores - cal.layer_score_mu) / cal.layer_score_sigma

        # Ledoit-Wolf Mahalanobis aggregation
        diff = z_scores - cal.agg_mu
        agg_score = float(np.sqrt(diff @ cal.agg_precision @ diff))

        self.detection_score = agg_score
        self.layer_z_scores = z_scores

        telemetry["detection_score"] = round(agg_score, 4)
        telemetry["threshold"] = round(cal.detection_threshold, 4)
        telemetry["top_layers"] = [int(l) for l in np.argsort(-z_scores)[:5]]
        telemetry["top_scores"] = [round(float(z_scores[l]), 2)
                                    for l in np.argsort(-z_scores)[:5]]

        is_anomalous = agg_score > cal.detection_threshold
        telemetry["anomaly_detected"] = bool(is_anomalous)

        if not is_anomalous:
            return StepDecision(logits=step_state.logits, telemetry=telemetry)

        self.anomaly_detected = True
        return StepDecision(
            logits=step_state.logits,
            telemetry=telemetry,
            abstain_message="[Output withheld by safety filter.]",
        )

    def end_sequence(self):
        return {
            "runtime_mode": "lcf",
            "detection_score": round(self.detection_score, 4),
            "anomaly_detected": self.anomaly_detected,
        }
