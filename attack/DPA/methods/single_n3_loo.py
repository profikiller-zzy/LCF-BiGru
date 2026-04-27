"""Single-layer N-3 diag_mahal with leave-one-out calibration."""

import numpy as np

from . import matched_eval

NAME = "single_n3_loo"


def evaluate(cal_z, clean_z, trig_z, cal_scores, n3, target_fpr=0.10, **_):
    n_cal = cal_scores.shape[0]

    # LOO z-scores on the raw N-3 diag_mahal score
    loo_scores = np.zeros(n_cal)
    for i in range(n_cal):
        mask = np.ones(n_cal, bool)
        mask[i] = False
        mu_i = cal_scores[mask, n3].mean()
        sig_i = max(cal_scores[mask, n3].std(), 1e-8)
        loo_scores[i] = (cal_scores[i, n3] - mu_i) / sig_i

    thr = float(np.percentile(loo_scores, 100 * (1 - target_fpr)))
    thr = max(thr, 1.0)

    fpr = float((clean_z[:, n3] >= thr).mean())
    asr = float(1.0 - (trig_z[:, n3] >= thr).mean())
    det = float((trig_z[:, n3] >= thr).mean())

    return {
        "cal": {"asr": asr, "fpr": fpr, "det_rate": det, "threshold": thr},
        "matched": matched_eval(clean_z[:, n3], trig_z[:, n3], target_fpr),
    }
