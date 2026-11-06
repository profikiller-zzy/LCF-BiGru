"""Stage 2: risk and manipulation "strategy x layer" tables and complementary layer selection.

Concepts are rebuilt on each fold's training goal groups only. For a held-out strategy m,
every goal group that contains an attack of strategy m is removed from concept fitting
(its H, B and all of its J), so neither the strategy nor its H/B endpoints leak. Responses
for the held-out groups are then read out with frozen directions and anchors.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import common as _common
from common import MIN_CELL_GROUPS, Panel, auroc, bootstrap_ci, sha256_file
from dual_concept.concepts import concept_scores, fit_concepts
from dual_concept.evaluation import calibrate_threshold
from dual_concept.selection import SelectionConfig, select_layers

FPR = 0.05
MIN_TRAIN_GROUPS_PER_FOLD = 100


def build_bank(panel: Panel, fit_groups: set[str], layer_idx: list[int], data_split: str = "train"):
    """Fit directions/anchors on train rows of the given goal groups, restricted to layer_idx."""
    rows = np.flatnonzero((panel.split == data_split) & np.isin(panel.group, list(fit_groups)))
    local = {panel.rows[i]["id"]: n for n, i in enumerate(rows)}
    rid = {panel.rows[i]["id"]: i for i in rows}
    risk, manip, safe, harm = [], [], [], set()
    for g in sorted(fit_groups):
        h, b = f"{g}_h", f"{g}_b"
        if h in local and b in local:
            risk.append([local[b], local[h]])
            safe.append(local[b])
            for i in rows:
                r = panel.rows[i]
                if r["group_id"] == g and r["kind"] == "jailbreak":
                    manip.append([local[h], local[r["id"]]]); harm.add(local[h])
    states = panel.states[np.ix_(rows, layer_idx)]
    bank = fit_concepts(states, risk, manip, sorted(set(safe)), sorted(harm), [_common.LAYERS[i] for i in layer_idx])
    return bank, len(risk), len(manip)


def fold_scores(panel, bank, idx, layer_idx):
    return concept_scores(panel.states[np.ix_(idx, layer_idx)], bank)


def run_fold(panel: Panel, held: str, level: str, layer_idx: list[int], eval_split: str, train_groups_all: set[str]):
    """Return per-row risk and manipulation scores plus reference thresholds for one held-out strategy."""
    strat = panel.strategy if level == "strategy" else panel.family
    j_rows = np.flatnonzero((panel.kind == "jailbreak") & (strat == held))
    held_groups = {panel.group[i] for i in j_rows if panel.split[i] == "train"}
    fit_groups = train_groups_all - held_groups
    bank, n_risk, n_manip = build_bank(panel, fit_groups, layer_idx)
    # reference responses from the training rows used to build the bank
    ref_rows = np.flatnonzero((panel.split == "train") & np.isin(panel.group, list(fit_groups)))
    r_ref, m_ref = fold_scores(panel, bank, ref_rows, layer_idx)
    safe_ref = panel.kind[ref_rows] == "safe"
    harm_ref = panel.kind[ref_rows] == "harmful"
    thr_risk = np.array([calibrate_threshold(r_ref[safe_ref, k], FPR)["threshold"] for k in range(len(layer_idx))])
    thr_manip = np.array([calibrate_threshold(m_ref[harm_ref, k], FPR)["threshold"] for k in range(len(layer_idx))])
    if eval_split == "train":
        ev_groups = held_groups
        ev = np.flatnonzero((panel.split == "train") & np.isin(panel.group, list(ev_groups)))
    else:   # select: whole select panel, restrict to groups of that strategy for J, all B for risk controls
        ev = np.flatnonzero(panel.split == "select")
    r_ev, m_ev = fold_scores(panel, bank, ev, layer_idx)
    return {"ev": ev, "risk": r_ev, "manip": m_ev, "thr_risk": thr_risk, "thr_manip": thr_manip,
            "held_groups": held_groups, "n_pairs": (n_risk, n_manip)}


def cell_metrics(panel, fold, held, level, branch, k, eval_split):
    ev = fold["ev"]
    strat = panel.strategy if level == "strategy" else panel.family
    kind, grp = panel.kind[ev], panel.group[ev]
    is_j = (kind == "jailbreak") & (strat[ev] == held)
    if eval_split == "train":
        j_groups = {g for g in grp[is_j]}
    else:
        j_groups = {g for g in grp[is_j]}
    out = {}
    if branch == "risk":
        s = fold["risk"][:, k]; thr = fold["thr_risk"][k]
        neg = kind == "safe"
        if eval_split == "train":     # controls from the same held-out goal groups only
            pass
        pos = is_j
        ref = "B"
    else:
        s = fold["manip"][:, k]; thr = fold["thr_manip"][k]
        # exact H endpoint of each held-out J
        ev_ids = {panel.rows[i]["id"]: n for n, i in enumerate(ev)}
        h_pos = [ev_ids.get(f"{panel.group[i]}_h") for i in ev[is_j]]
        keep = [p for p in h_pos if p is not None]
        neg = np.zeros(len(ev), dtype=bool); neg[keep] = True
        pos = is_j
        ref = "H"
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    lab = np.r_[np.ones(pos.sum()), np.zeros(neg.sum())]
    sc = np.r_[s[pos], s[neg]]
    row = {"auroc": auroc(lab, sc), "tpr": float((s[pos] >= thr).mean()),
           "control_rate": float((s[neg] >= thr).mean()), "n_J": int(pos.sum()), "n_control": int(neg.sum()),
           "n_goal_groups": len(j_groups), "control": ref}
    if branch == "manip":
        # paired delta per goal group
        d = {}
        ev_ids = {panel.rows[i]["id"]: n for n, i in enumerate(ev)}
        for n in np.flatnonzero(pos):
            g = panel.group[ev[n]]
            hn = ev_ids.get(f"{g}_h")
            if hn is not None:
                d.setdefault(g, []).append(s[n] - s[hn])
        per_group = {g: float(np.mean(v)) for g, v in d.items()}
        row["delta_mean"] = float(np.mean(list(per_group.values()))) if per_group else float("nan")
        row["delta_median"] = float(np.median(list(per_group.values()))) if per_group else float("nan")
        lo, hi = bootstrap_ci(per_group)
        row["delta_ci_low"], row["delta_ci_high"] = lo, hi
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-summary", required=True)
    ap.add_argument("--min-n", type=int, default=20, help="min J rows for a strategy-level column")
    args = ap.parse_args()
    out = Path(args.out) / "stage2"; out.mkdir(parents=True, exist_ok=True)
    panel = Panel(args.data, args.cache)
    probe = json.loads(Path(args.probe_summary).read_text())
    if probe["status"] != "ok":
        raise SystemExit(f"Stage 1 status {probe['status']}: stopping, no thresholds are relaxed")
    cand = [int(l) for l in probe["C_probe"]]
    layer_idx = [l - 1 for l in cand]
    train_groups = {g for g, s, k in zip(panel.group, panel.split, panel.kind) if s == "train" and k == "harmful"}

    train_j = (panel.split == "train") & (panel.kind == "jailbreak")
    fam_names = sorted(set(panel.family[train_j]))
    strat_count = {s: int((panel.strategy[train_j] == s).sum()) for s in set(panel.strategy[train_j])}
    strat_names = sorted(s for s, n in strat_count.items() if n >= args.min_n)
    small = sorted(s for s, n in strat_count.items() if n < args.min_n)
    levels = {"family": fam_names, "strategy": strat_names}
    (out / "columns.json").write_text(json.dumps({"family": fam_names, "strategy": strat_names,
                                                  "pooled_small_strategies": small, "candidate_layers": cand}, indent=1))

    long_rows = []
    folds_for_selection = {"family": {"risk": [], "manip": []}, "strategy": {"risk": [], "manip": []}}
    for level, names in levels.items():
        for held in names:
            for eval_split in ("train", "select"):
                if eval_split == "select" and not ((panel.split == "select") & (panel.kind == "jailbreak") &
                                                    ((panel.strategy if level == "strategy" else panel.family) == held)).any():
                    continue
                fold = run_fold(panel, held, level, layer_idx, eval_split, train_groups)
                for branch in ("risk", "manip"):
                    for k, layer in enumerate(cand):
                        m = cell_metrics(panel, fold, held, level, branch, k, eval_split)
                        if m is None:
                            continue
                        low = m["n_goal_groups"] < MIN_CELL_GROUPS
                        long_rows.append({"branch": branch, "level": level, "strategy": held, "layer": layer,
                                          "eval": "train_heldout" if eval_split == "train" else "select_recheck",
                                          "low_sample": low, **m})
                if eval_split == "train":
                    folds_for_selection[level]["_raw"] = folds_for_selection[level].get("_raw", [])
                    folds_for_selection[level]["_raw"].append((held, fold))
    cols = ["branch", "level", "strategy", "layer", "eval", "low_sample", "auroc", "tpr", "control_rate", "control",
            "n_J", "n_control", "n_goal_groups", "delta_mean", "delta_median", "delta_ci_low", "delta_ci_high"]
    with (out / "matrix_long.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in long_rows:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- complementary selection on train held-out folds ----
    cfg = SelectionConfig()
    selection = {}
    for level in ("strategy", "family"):
        raw = folds_for_selection[level]["_raw"]
        for branch in ("risk", "manip"):
            folds = []
            for held, fold in raw:
                strat = panel.strategy if level == "strategy" else panel.family
                ev = fold["ev"]; kind = panel.kind[ev]
                is_j = (kind == "jailbreak") & (strat[ev] == held)
                safe = kind == "safe"
                if branch == "risk":
                    scores, thr = fold["risk"], fold["thr_risk"]
                    mask = is_j | safe | (kind == "harmful")
                    labels = np.where(is_j, 1, np.where(kind == "harmful", 1, 0))
                    groups = np.where(is_j, held, "direct_harmful")
                else:
                    scores, thr = fold["manip"], fold["thr_manip"]
                    ev_ids = {panel.rows[i]["id"]: n for n, i in enumerate(ev)}
                    h_n = {ev_ids[f"{panel.group[i]}_h"] for i in ev[is_j] if f"{panel.group[i]}_h" in ev_ids}
                    h_mask = np.zeros(len(ev), dtype=bool); h_mask[list(h_n)] = True
                    mask = is_j | h_mask | safe
                    labels = np.where(is_j, 1, np.where(h_mask, 0, -1))
                    groups = np.full(len(ev), held)
                folds.append({"scores": scores[mask], "labels": labels[mask], "safe_mask": safe[mask],
                              "attack_groups": groups[mask], "thresholds": thr})
            try:
                res = select_layers(folds, cand, cfg)
                selection[f"{level}/{branch}"] = {"layers": res.layers, "layers_by_budget": res.layers_by_budget,
                                                  "diagnostics": res.diagnostics}
            except ValueError as error:
                selection[f"{level}/{branch}"] = {"error": str(error)}
    (out / "selection.json").write_text(json.dumps(selection, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print(json.dumps({k: (v.get("layers") or v.get("error")) for k, v in selection.items()}, indent=1))


if __name__ == "__main__":
    main()
