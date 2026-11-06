"""Length-controlled stage 2: manipulation and risk responses against length-matched controls.

For a held-out strategy m the concepts are rebuilt without m's goal groups (same as stage 2).
Each held-out J is then compared with controls that carry the same formatted token count:
  manipulation: AUROC(H_pad vs J) and paired response difference c^J(J) - c^J(H_pad)
  risk:         AUROC(B_pad vs J)
H_pad / B_pad are neutral-prose + request; they have no manipulation element, so a signal that
survives is not explained by length. The directions and anchors are still fitted on the original
(unpadded) pairs, exactly as in stage 2, so only the evaluation negatives change.
Layer selection then reruns `select_layers` with the padded controls as the negatives.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as _common
from common import MIN_CELL_GROUPS, Panel, auroc, bootstrap_ci
from dual_concept.backbone import StateCache
from dual_concept.concepts import concept_scores
from dual_concept.data import read_jsonl
from dual_concept.evaluation import calibrate_threshold
from dual_concept.selection import SelectionConfig, select_layers
from stage2_matrices import build_bank

FPR = 0.05


def load_controls(path: Path, cache_root: Path, protocol: dict):
    cache = StateCache(cache_root / "rows")
    rows, states = [], []
    for split in ("train", "select"):
        for r in read_jsonl(path / "development" / f"{split}.jsonl"):
            s = cache.load(r["id"], r["text"], protocol, expected_shape=(len(_common.LAYERS), int(protocol["hidden_size"])))
            if s is None:
                raise FileNotFoundError(f"Missing control state {r['id']}")
            rows.append(r); states.append(s)
    return rows, np.stack(states).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-summary", required=True)
    ap.add_argument("--min-n", type=int, default=20)
    a = ap.parse_args()
    out = Path(a.out) / "stage2_length_controlled"; out.mkdir(parents=True, exist_ok=True)
    panel = Panel(a.data, a.cache)
    crow, cstate = load_controls(Path(a.controls), Path(a.cache), panel.protocol)
    cid = {r["id"]: i for i, r in enumerate(crow)}
    cand = [int(l) for l in json.loads(Path(a.probe_summary).read_text())["C_probe"]]
    li = [l - 1 for l in cand]
    train_groups = {g for g, s, k in zip(panel.group, panel.split, panel.kind) if s == "train" and k == "harmful"}
    tj = (panel.split == "train") & (panel.kind == "jailbreak")
    fam_names = sorted(set(panel.family[tj]))
    cnt = {s: int((panel.strategy[tj] == s).sum()) for s in set(panel.strategy[tj])}
    strat_names = sorted(s for s, n in cnt.items() if n >= a.min_n)

    long_rows, raw_folds = [], {"family": [], "strategy": []}
    for level, names, arr in (("family", fam_names, panel.family), ("strategy", strat_names, panel.strategy)):
        for held in names:
            for ev_split in ("train", "select"):
                jm = np.flatnonzero((panel.kind == "jailbreak") & (arr == held) & (panel.split == ev_split))
                jm = np.array([i for i in jm if f"{panel.rows[i]['id']}__pad_h" in cid])
                if len(jm) == 0:
                    continue
                held_groups = {panel.group[i] for i in jm if panel.split[i] == "train"}
                if ev_split == "select":
                    held_groups = {panel.group[i] for i in np.flatnonzero((panel.kind == "jailbreak") & (arr == held) & (panel.split == "train"))}
                fit_groups = train_groups - held_groups
                bank, _, _ = build_bank(panel, fit_groups, li)
                # Reference responses for the operating thresholds come from LENGTH-MATCHED controls of the fit
                # groups (padded H for the manipulation threshold, padded B for the risk threshold). Thresholds
                # calibrated on short unpadded requests would flag every long padded control and re-introduce length.
                fit_j = np.flatnonzero((panel.kind == "jailbreak") & (panel.split == "train") & np.isin(panel.group, list(fit_groups)))
                fit_j = np.array([i for i in fit_j if f"{panel.rows[i]['id']}__pad_h" in cid])
                ref_h = np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in fit_j])
                ref_b = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in fit_j])
                _, m_ref = concept_scores(cstate[np.ix_(ref_h, li)], bank)
                r_ref, _ = concept_scores(cstate[np.ix_(ref_b, li)], bank)
                thr_r = np.array([calibrate_threshold(r_ref[:, k], FPR)["threshold"] for k in range(len(li))])
                thr_m = np.array([calibrate_threshold(m_ref[:, k], FPR)["threshold"] for k in range(len(li))])
                r_j, m_j = concept_scores(panel.states[np.ix_(jm, li)], bank)
                ph = np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in jm])
                pb = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in jm])
                r_pb, m_pb = concept_scores(cstate[np.ix_(pb, li)], bank)
                _, m_ph = concept_scores(cstate[np.ix_(ph, li)], bank)
                groups = panel.group[jm]
                for k, layer in enumerate(cand):
                    lab = np.r_[np.ones(len(jm)), np.zeros(len(jm))]
                    d = {}
                    for g, dv in zip(groups, m_j[:, k] - m_ph[:, k]):
                        d.setdefault(g, []).append(dv)
                    per_group = {g: float(np.mean(v)) for g, v in d.items()}
                    lo, hi = bootstrap_ci(per_group)
                    n_groups = len(set(groups))
                    for branch, sj, sneg, thr in (("manip", m_j[:, k], m_ph[:, k], thr_m[k]), ("risk", r_j[:, k], r_pb[:, k], thr_r[k])):
                        row = {"branch": branch, "level": level, "strategy": held, "layer": layer,
                               "eval": "train_heldout" if ev_split == "train" else "select_recheck",
                               "auroc": auroc(lab, np.r_[sj, sneg]), "tpr": float((sj >= thr).mean()),
                               "control_rate": float((sneg >= thr).mean()), "n_J": len(jm), "n_goal_groups": n_groups,
                               "low_sample": n_groups < MIN_CELL_GROUPS}
                        if branch == "manip":
                            row.update(delta_mean=float(np.mean(list(per_group.values()))),
                                       delta_ci_low=lo, delta_ci_high=hi)
                        long_rows.append(row)
                if ev_split == "train":
                    raw_folds[level].append((held, jm, r_j, m_j, r_pb, m_pb, m_ph, thr_r, thr_m))
    cols = ["branch", "level", "strategy", "layer", "eval", "low_sample", "auroc", "tpr", "control_rate",
            "n_J", "n_goal_groups", "delta_mean", "delta_ci_low", "delta_ci_high"]
    with (out / "matrix_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in long_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- selection with padded controls as the negatives ----
    # Positives: held-out J. Negatives: length-matched padded controls (H_pad for manipulation, B_pad for risk).
    # The safe-FPR term uses the held-out groups' padded benign B (length-matched, actually safe) under the
    # same fold's readout, so no fold depends on unpadded short requests.
    cfg = SelectionConfig()
    selection = {}
    for level, folds in raw_folds.items():
        for branch in ("risk", "manip"):
            fl = []
            for held, jm, r_j, m_j, r_pb, m_pb, m_ph, thr_r, thr_m in folds:
                n = len(jm)
                if branch == "risk":
                    scores = np.vstack([r_j, r_pb])                  # negatives are the safe controls themselves
                    labels = np.r_[np.ones(n), np.zeros(n)]
                    safe = np.r_[np.zeros(n, bool), np.ones(n, bool)]
                    thr = thr_r
                else:
                    scores = np.vstack([m_j, m_ph, m_pb])
                    labels = np.r_[np.ones(n), np.zeros(n), -np.ones(n)]   # -1: safe control, not a branch negative
                    safe = np.r_[np.zeros(2 * n, bool), np.ones(n, bool)]
                    thr = thr_m
                fl.append({"scores": scores, "labels": labels, "safe_mask": safe,
                           "attack_groups": np.full(len(labels), held), "thresholds": thr})
            try:
                res = select_layers(fl, cand, cfg)
                selection[f"{level}/{branch}"] = {"layers": res.layers, "layers_by_budget": res.layers_by_budget,
                                                  "diagnostics": res.diagnostics}
            except ValueError as error:
                selection[f"{level}/{branch}"] = {"error": str(error)}
    (out / "selection.json").write_text(json.dumps(selection, indent=1,
                                                    default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: (v.get("layers") or v.get("error")) for k, v in selection.items()}, indent=1))
    print("matrix rows", len(long_rows))


if __name__ == "__main__":
    main()
