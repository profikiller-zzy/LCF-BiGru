"""Higher Criticism statistic for sparse signal detection across layers."""

import numpy as np
from scipy.stats import norm

from . import standard_result

NAME = "higher_crit"


def _score(z):
    K = z.shape[1]
    p = np.clip(1 - norm.cdf(z), 1e-15, 1.0)
    scores = np.zeros(len(z))
    for i in range(len(z)):
        ps = np.sort(p[i])
        k = np.arange(1, K + 1)
        denom = np.sqrt(ps * (1 - ps) + 1e-15)
        scores[i] = (np.sqrt(K) * (k / K - ps) / denom).max()
    return scores


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, **_):
    return standard_result(
        _score(cal_z), _score(clean_z), _score(trig_z), target_fpr
    )
