"""PCA residual: norm of z-score vector minus its top-k PCA reconstruction."""

import numpy as np
from sklearn.decomposition import PCA

from . import standard_result

NAME_TEMPLATE = "pca_resid_k{k}"


def evaluate(cal_z, clean_z, trig_z, k, target_fpr=0.10, **_):
    pca = PCA(n_components=k).fit(cal_z)

    def _residual(z):
        return np.linalg.norm(z - pca.inverse_transform(pca.transform(z)), axis=1)

    return standard_result(
        _residual(cal_z), _residual(clean_z), _residual(trig_z), target_fpr
    )
