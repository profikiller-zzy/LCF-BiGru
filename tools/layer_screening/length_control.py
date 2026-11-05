"""Length confound control for the manipulation branch.

Every J is longer than its exact H, so a high AUROC(H vs J) could reflect input length alone.
This script (1) measures how much of each layer manipulation response is explained by token
count and (2) re-evaluates AUROC(H vs J) after removing that length trend using only held-out
H references (a within-layer linear fit of the response on log token count, fitted on H rows).
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np
from common import LAYERS, Panel, auroc
from stage2_matrices import build_bank, fold_scores
from dual_concept.concepts import concept_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1"); ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--probe-summary", required=True)
    a = ap.parse_args()
    panel = Panel(a.data, a.cache)
    cand = [int(l) for l in json.loads(Path(a.probe_summary).read_text())["C_probe"]]
    li = [l - 1 for l in cand]
    ntok = np.array([r["full_input_token_count"] for r in panel.rows], dtype=float)
    train_groups = {g for g, s, k in zip(panel.group, panel.split, panel.kind) if s == "train" and k == "harmful"}
    rows = []
    for held in sorted(set(panel.family[(panel.kind == "jailbreak") & (panel.split == "train")])):
        jm = np.flatnonzero((panel.kind == "jailbreak") & (panel.family == held) & (panel.split == "train"))
        hg = {panel.group[i] for i in jm}
        bank, _, _ = build_bank(panel, train_groups - hg, li)
        ref = np.flatnonzero((panel.split == "train") & np.isin(panel.group, list(train_groups - hg)) & (panel.kind == "harmful"))
        ev_j = jm
        ev_h = np.array([panel.index[f"{panel.group[i]}_h"] for i in jm])
        _, m_ref = concept_scores(panel.states[np.ix_(ref, li)], bank)
        _, m_j = concept_scores(panel.states[np.ix_(ev_j, li)], bank)
        _, m_h = concept_scores(panel.states[np.ix_(ev_h, li)], bank)
        x_ref = np.log(ntok[ref])
        for k, layer in enumerate(cand):
            slope, icpt = np.polyfit(x_ref, m_ref[:, k], 1)
            # fraction of H-reference variance explained by length
            r2 = float(np.corrcoef(x_ref, m_ref[:, k])[0, 1] ** 2)
            adj_j = m_j[:, k] - (slope * np.log(ntok[ev_j]) + icpt)
            adj_h = m_h[:, k] - (slope * np.log(ntok[ev_h]) + icpt)
            lab = np.r_[np.ones(len(ev_j)), np.zeros(len(ev_h))]
            rows.append({"family": held, "layer": layer, "n_J": len(ev_j), "ref_length_r2": r2,
                         "auroc_raw": auroc(lab, np.r_[m_j[:, k], m_h[:, k]]),
                         "auroc_length_adjusted": auroc(lab, np.r_[adj_j, adj_h]),
                         "length_only_auroc": auroc(lab, np.r_[ntok[ev_j], ntok[ev_h]])})
    out = Path(a.out) / "stage2"; out.mkdir(parents=True, exist_ok=True)
    with (out / "length_control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("rows", len(rows))

if __name__ == "__main__":
    main()
