"""
Ledoit-Wolf Mahalanobis on all-layer z-scores with LOO calibration.

This is the main method reported in the LCF paper: the threshold comes
from leave-one-out scores on the calibration set (unbiased), while the
deployed scoring function is fit on the full calibration set.
"""

import numpy as np
from sklearn.covariance import LedoitWolf

from . import matched_eval
from .multi_lw import _fit, _score

NAME = "multi_lw_loo"


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, verbose=True, **_):
    n_cal = cal_z.shape[0]

    if verbose:
        print("  Computing LOO calibration for LW Mahalanobis...")

    # LOO: for each cal example, refit LW on the other n-1 and score it
    loo_scores = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool)
        mask[i] = False
        cal_loo = cal_z[mask]
        mu_loo = cal_loo.mean(axis=0)
        prec_loo = LedoitWolf().fit(cal_loo).precision_
        diff = cal_z[i] - mu_loo
        loo_scores[i] = float(np.sqrt(diff @ prec_loo @ diff))

    thr = float(np.percentile(loo_scores, 100 * (1 - target_fpr)))
    thr = max(thr, 1.0)

    # Deployed scoring function: fit on full cal set
    mu, prec = _fit(cal_z)
    clean_s = _score(clean_z, mu, prec)
    trig_s = _score(trig_z, mu, prec)

    fpr = float((clean_s >= thr).mean())
    asr = float(1.0 - (trig_s >= thr).mean())
    det = float((trig_s >= thr).mean())

    if verbose:
        print(f"  LOO threshold: {thr:.3f}, FPR: {fpr:.1%}, ASR: {asr:.1%}")

    return {
        "cal": {"asr": asr, "fpr": fpr, "det_rate": det, "threshold": thr},
        "matched": matched_eval(clean_s, trig_s, target_fpr),
    }
