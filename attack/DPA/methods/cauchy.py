"""Cauchy combination test across all layers."""

import numpy as np
from scipy.stats import norm

from . import standard_result

NAME = "cauchy_all"


def _score(z):
    p = np.clip(1 - norm.cdf(z), 1e-15, 1.0)
    return np.tan((0.5 - p) * np.pi).sum(axis=1) / z.shape[1]


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, **_):
    return standard_result(
        _score(cal_z), _score(clean_z), _score(trig_z), target_fpr
    )
