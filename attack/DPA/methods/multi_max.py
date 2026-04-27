"""Max z-score across all layers."""

from . import standard_result

NAME = "multi_max"


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, **_):
    return standard_result(
        cal_z.max(axis=1), clean_z.max(axis=1), trig_z.max(axis=1), target_fpr
    )
