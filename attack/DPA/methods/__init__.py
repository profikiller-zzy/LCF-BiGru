"""
Detection method registry for multi-layer delta analysis.

Each submodule defines a single detection method with:
- ``NAME``: string identifier for the method
- ``evaluate(**ctx) -> dict``: runs the method against a context dict and
  returns ``{"cal": {...}, "matched": float}``

Context keys (populated by the orchestrator in multilayer_detection_200cal.py):
    cal_z, clean_z, trig_z         : (n, num_layers) z-scored diag_mahal matrices
    cal_scores, clean_scores, trig_scores : raw diag_mahal score matrices
    n3                              : index of the N-3 layer (num_layers - 3)
    num_layers                      : total number of transformer layers
    target_fpr                      : target false-positive rate (default 0.10)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers — used by every method file
# ---------------------------------------------------------------------------
def threshold_at_fpr(scores: np.ndarray, target_fpr: float = 0.10) -> float:
    return float(np.percentile(scores, 100 * (1 - target_fpr)))


def calibrated_eval(cal_s, clean_s, trig_s, target_fpr: float = 0.10) -> dict:
    """Threshold from calibration scores; measure FPR/ASR on test."""
    thr = threshold_at_fpr(cal_s, target_fpr)
    fpr = float((clean_s >= thr).mean())
    asr = float(1.0 - (trig_s >= thr).mean())
    det_rate = float((trig_s >= thr).mean())
    return {"asr": asr, "fpr": fpr, "det_rate": det_rate, "threshold": float(thr)}


def matched_eval(clean_s, trig_s, target_fpr: float = 0.10) -> float:
    """Threshold from test clean scores (matched-FPR oracle)."""
    thr = threshold_at_fpr(clean_s, target_fpr)
    return float(1.0 - (trig_s >= thr).mean())


def standard_result(cal_s, clean_s, trig_s, target_fpr: float = 0.10) -> dict:
    """Convenience wrapper returning both cal-calibrated and matched-FPR results."""
    return {
        "cal": calibrated_eval(cal_s, clean_s, trig_s, target_fpr),
        "matched": matched_eval(clean_s, trig_s, target_fpr),
    }
