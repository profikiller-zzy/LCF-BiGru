"""Aggregate stage 8 results across backbones into one comparison table (policy A).

Reads <dir>/stage8_<model>/cross_model_results.csv and fold_layers.json. Averages are over held-out semantic strategies
(equal weight per strategy); the worst-strategy value is reported next to each mean. The W(B) columns average only over
the five strategies that have W(B) controls. Layer positions are shown as relative depth (layer / num_hidden_layers).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

ORDER = ["Qwen2.5-1.5B-Instruct", "Qwen2.5-7B-Instruct", "Qwen3-4B", "Qwen3-8B", "Llama-3.1-8B-Instruct"]
ARMS = ["length_rule", "bow_lr", "hidden_probe", "dccf_risk", "dccf_orig_fusion", "dccf_wrap_fusion"]


def f(x):
    return float(x) if x not in ("", "nan", None) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = Path(a.runs)
    rows, meta = [], {}
    for m in ORDER:
        p = runs / f"stage8_{m}"
        if (p / "cross_model_results.csv").exists():
            rows += list(csv.DictReader(open(p / "cross_model_results.csv")))
            meta[m] = json.loads((p / "fold_layers.json").read_text())
    models = [m for m in ORDER if m in meta]
    lines = []
    for mode, title in (("mixed_wb", "threshold = calibrate + half the long-benign pool + fit-group W(B)  (the policy-A operating point)"),
                        ("short", "threshold = 200 calibrate requests only")):
        lines.append(f"\n### {title}\n")
        lines.append("| model | arm | J TPR mean (worst) | H TPR | B FPR | long FPR | W(B) flagged | AUROC J vs W(B) |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for m in models:
            for arm in ARMS:
                rs = [x for x in rows if x["model"] == m and x["arm"] == arm and x["threshold_mode"] == mode]
                if not rs:
                    continue
                g = lambda k, fn=np.nanmean: fn([f(x[k]) for x in rs])
                lines.append(f"| {m} | {arm} | {g('tpr_J'):.2f} ({g('tpr_J', np.nanmin):.2f}) | {g('tpr_H'):.2f} | {g('fpr_B'):.2f} | "
                             f"{g('fpr_long'):.2f} | {g('rate_WB'):.2f} | {g('auroc_J_vs_WB'):.2f} |")
    lines.append("\n### Chosen layers per model (relative depth, mode over the 7 folds)\n")
    lines.append("| model | layers | risk | J-minus-H manipulation | wrapper manipulation |")
    lines.append("|---|---|---|---|---|")
    for m in models:
        folds = meta[m]["folds"]
        depth = meta[m]["num_hidden_layers"]
        def top(key):
            c = Counter(round(l / depth, 2) for fd in folds.values() if "error" not in fd for l in fd[key])
            return ", ".join(f"{d:.2f}x{n}" for d, n in c.most_common(3))
        lines.append(f"| {m} | {depth - 1} of {depth} | {top('risk')} | {top('orig_manip')} | {top('wrap_manip')} |")
    nonconv = Counter((x["model"], x["arm"]) for x in rows if x["converged"] == "False")
    lines.append("\n### Fusion fits that did not converge\n")
    lines.append(", ".join(f"{k[0]}/{k[1]} x{v // 2}" for k, v in sorted(nonconv.items())) or "none")
    Path(a.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
