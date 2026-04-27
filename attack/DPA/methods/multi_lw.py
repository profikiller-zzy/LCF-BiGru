"""Ledoit-Wolf Mahalanobis on all-layer z-scores (naive calibration)."""

import numpy as np
from sklearn.covariance import LedoitWolf

from . import standard_result

NAME = "multi_lw"


def _fit(cal_z):
    mu = cal_z.mean(axis=0)
    prec = LedoitWolf().fit(cal_z).precision_
    return mu, prec


def _score(z, mu, prec):
    d = z - mu
    return np.sqrt(np.sum(d @ prec * d, axis=1))


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, **_):
    mu, prec = _fit(cal_z)
    cal_s = _score(cal_z, mu, prec)
    clean_s = _score(clean_z, mu, prec)
    trig_s = _score(trig_z, mu, prec)
    return standard_result(cal_s, clean_s, trig_s, target_fpr)
