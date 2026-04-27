"""
Sequential test: N-3 first, then fall back to the layer most independent
of N-3 (minimum |correlation| with N-3 over the calibration set).

The FPR budget is split across the two stages with alpha1 + alpha2*(1-alpha1)
= target_fpr, using alpha1 = 0.08 by default.
"""

import numpy as np

NAME = "sequential"


def evaluate(cal_z, clean_z, trig_z, n3, num_layers, target_fpr=0.10, alpha1=0.08, **_):
    # Find the layer least correlated with N-3 on the calibration set
    corr_n3 = np.array([
        np.corrcoef(cal_z[:, n3], cal_z[:, l])[0, 1] if l != n3 else 1.0
        for l in range(num_layers)
    ])
    comp_layer = int(np.argmin(np.abs(corr_n3)))

    alpha2 = (target_fpr - alpha1) / (1 - alpha1)
    n3_thr = np.percentile(cal_z[:, n3], 100 * (1 - alpha1))

    cal_pass = cal_z[:, n3] < n3_thr
    if cal_pass.sum() > 3:
        comp_thr = np.percentile(cal_z[cal_pass, comp_layer], 100 * (1 - alpha2))
    else:
        comp_thr = np.percentile(cal_z[:, comp_layer], 100 * (1 - alpha2))

    def _detect(z_n3, z_comp):
        n3_flag = z_n3 >= n3_thr
        comp_flag = (~n3_flag) & (z_comp >= comp_thr)
        return (n3_flag | comp_flag).astype(float)

    clean_det = _detect(clean_z[:, n3], clean_z[:, comp_layer])
    trig_det = _detect(trig_z[:, n3], trig_z[:, comp_layer])

    return {
        "name": f"seq_n3+L{comp_layer}",
        "comp_layer": comp_layer,
        "comp_corr": float(corr_n3[comp_layer]),
        "cal": {
            "asr": float(1.0 - trig_det.mean()),
            "fpr": float(clean_det.mean()),
            "det_rate": float(trig_det.mean()),
            "threshold": 0,
        },
        "matched": float(1.0 - trig_det.mean()),
    }
