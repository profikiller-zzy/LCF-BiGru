"""Single-layer N-3 diag_mahal (naive calibration baseline)."""

from . import standard_result

NAME = "single_n3"


def evaluate(cal_z, clean_z, trig_z, n3, target_fpr=0.10, **_):
    return standard_result(
        cal_z[:, n3], clean_z[:, n3], trig_z[:, n3], target_fpr
    )
