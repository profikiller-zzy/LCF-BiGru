"""Max z-score over the final 5 layers only."""

from . import standard_result

NAME = "max_last5"


def evaluate(cal_z, clean_z, trig_z, target_fpr=0.10, **_):
    return standard_result(
        cal_z[:, -5:].max(axis=1),
        clean_z[:, -5:].max(axis=1),
        trig_z[:, -5:].max(axis=1),
        target_fpr,
    )
