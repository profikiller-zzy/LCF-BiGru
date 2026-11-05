"""Semantic-strategy layer selection and cross-strategy fusion (development analysis).

1. Strategy-by-layer tables over 7 semantic strategies (persona_roleplay, fiction_scene, instruction_control,
   adversarial_suffix, persuasion, scenario_reframing, rule_suffix).
2. Per-branch complementary layer selection: each strategy is held out in turn (with its goal groups), concepts
   are rebuilt without it, and `select_layers` scores coverage per held-out strategy, so the worst strategy drives
   the choice and the dominant persona strategy cannot swamp it.
3. Fusion of the union of the selected layers with the project's sparse linear fusion, evaluated by leaving one
   whole strategy out of BOTH concept building and fusion fitting.

Length caveat carried over from stage 2: negatives for the manipulation branch are the length-matched padded H,
with the measured padding shift reported next to every layer. Nothing here touches confirm or calibrate.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import LAYERS, MIN_CELL_GROUPS, Panel, auroc, bootstrap_ci
from dual_concept.concepts import concept_scores
from dual_concept.evaluation import calibrate_threshold
from dual_concept.fusion import FusionConfig, build_feature_matrix, fit_sparse_fusion
from dual_concept.selection import SelectionConfig, select_layers
from stage2_length_controlled import load_controls
from stage2_matrices import build_bank
from strategy_labels import semantic_strategy

FPR = 0.05


def layer_table_cell(sj, sneg, thr):
    lab = np.r_[np.ones(len(sj)), np.zeros(len(sneg))]
    return {"auroc": auroc(lab, np.r_[sj, sneg]), "tpr": float((sj >= thr).mean()),
            "control_rate": float((sneg >= thr).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--probe-summary", required=True)
    ap.add_argument("--max-layers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out) / "stage3_strategy_fusion"; out.mkdir(parents=True, exist_ok=True)
    panel = Panel(a.data, a.cache)
    crow, cstate = load_controls(Path(a.controls), Path(a.cache), panel.protocol)
    cid = {r["id"]: i for i, r in enumerate(crow)}
    sem = np.array([semantic_strategy(r) or "" for r in panel.rows])
    cand = [int(l) for l in json.loads(Path(a.probe_summary).read_text())["C_probe"]]
    li = [l - 1 for l in cand]
    train_groups = {g for g, s, k in zip(panel.group, panel.split, panel.kind) if s == "train" and k == "harmful"}
    tj = (panel.split == "train") & (panel.kind == "jailbreak")
    strategies = sorted(set(sem[tj]))
    cfg = SelectionConfig(max_layers=a.max_layers)

    def has_pad(i):
        return f"{panel.rows[i]['id']}__pad_h" in cid

    cells, folds = [], {}
    for held in strategies:
        jm = np.array([i for i in np.flatnonzero(tj & (sem == held)) if has_pad(i)])
        held_groups = {panel.group[i] for i in jm}
        fit_groups = train_groups - held_groups
        bank, _, _ = build_bank(panel, fit_groups, li)
        fj = np.array([i for i in np.flatnonzero(tj & np.isin(panel.group, list(fit_groups))) if has_pad(i)])
        rb = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in fj])
        rh = np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in fj])
        r_ref, _ = concept_scores(cstate[np.ix_(rb, li)], bank)
        _, m_ref = concept_scores(cstate[np.ix_(rh, li)], bank)
        thr_r = np.array([calibrate_threshold(r_ref[:, k], FPR)["threshold"] for k in range(len(li))])
        thr_m = np.array([calibrate_threshold(m_ref[:, k], FPR)["threshold"] for k in range(len(li))])
        r_j, m_j = concept_scores(panel.states[np.ix_(jm, li)], bank)
        ph = np.array([cid[f"{panel.rows[i]['id']}__pad_h"] for i in jm])
        pb = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in jm])
        r_pb, m_pb = concept_scores(cstate[np.ix_(pb, li)], bank)
        _, m_ph = concept_scores(cstate[np.ix_(ph, li)], bank)
        h_plain = np.array([panel.index[f"{panel.group[i]}_h"] for i in jm])
        _, m_hp = concept_scores(panel.states[np.ix_(h_plain, li)], bank)
        shift = (m_ph - m_hp).mean(0)        # what padding alone does to the manipulation readout
        groups = panel.group[jm]
        for k, layer in enumerate(cand):
            for branch, sj, sneg, thr in (("risk", r_j[:, k], r_pb[:, k], thr_r[k]), ("manip", m_j[:, k], m_ph[:, k], thr_m[k])):
                c = layer_table_cell(sj, sneg, thr)
                row = {"branch": branch, "strategy": held, "layer": layer, "n_J": len(jm),
                       "n_goal_groups": len(set(groups)), "low_sample": len(set(groups)) < MIN_CELL_GROUPS, **c}
                if branch == "manip":
                    d = {}
                    for g, v in zip(groups, m_j[:, k] - m_ph[:, k]):
                        d.setdefault(g, []).append(v)
                    per = {g: float(np.mean(v)) for g, v in d.items()}
                    lo, hi = bootstrap_ci(per)
                    row.update(delta_vs_pad=float(np.mean(list(per.values()))), delta_ci_low=lo, delta_ci_high=hi,
                               padding_shift=float(shift[k]))
                cells.append(row)
        n = len(jm)
        folds[held] = dict(
            risk={"scores": np.vstack([r_j, r_pb]), "labels": np.r_[np.ones(n), np.zeros(n)],
                  "safe_mask": np.r_[np.zeros(n, bool), np.ones(n, bool)],
                  "attack_groups": np.full(2 * n, held), "thresholds": thr_r},
            manip={"scores": np.vstack([m_j, m_ph, m_pb]), "labels": np.r_[np.ones(n), np.zeros(n), -np.ones(n)],
                   "safe_mask": np.r_[np.zeros(2 * n, bool), np.ones(n, bool)],
                   "attack_groups": np.full(3 * n, held), "thresholds": thr_m})
        print("table", held, "done", flush=True)

    cols = ["branch", "strategy", "layer", "n_J", "n_goal_groups", "low_sample", "auroc", "tpr", "control_rate",
            "delta_vs_pad", "delta_ci_low", "delta_ci_high", "padding_shift"]
    with (out / "strategy_layer_table.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in cells:
            w.writerow({c: r.get(c, "") for c in cols})

    # ---- complementary selection per branch over semantic strategies ----
    selection = {}
    for branch in ("risk", "manip"):
        try:
            res = select_layers([folds[s][branch] for s in strategies], cand, cfg)
            selection[branch] = {"layers": res.layers, "layers_by_budget": res.layers_by_budget,
                                 "diagnostics": res.diagnostics, "strategies": strategies}
        except ValueError as e:
            selection[branch] = {"error": str(e)}
    (out / "selection.json").write_text(json.dumps(selection, indent=1,
                                                    default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    print({b: v.get("layers") or v.get("error") for b, v in selection.items()}, flush=True)
    if any("error" in v for v in selection.values()):
        return

    # ---- fusion, leave-one-strategy-out over both concept building and fusion fitting ----
    risk_layers, manip_layers = selection["risk"]["layers"], selection["manip"]["layers"]
    union = sorted(set(risk_layers) | set(manip_layers))
    arms = {"fusion_both": (risk_layers, manip_layers),
            "risk_only": (risk_layers, []), "manip_only": ([], manip_layers),
            "best_single_risk": (risk_layers[:1], []), "best_single_manip": ([], manip_layers[:1])}
    results = []
    fcfg = FusionConfig()
    for held in strategies:
        jm_all = np.flatnonzero(tj & (sem == held))
        held_groups = {panel.group[i] for i in jm_all}
        fit_groups = train_groups - held_groups
        bank, _, _ = build_bank(panel, fit_groups, [l - 1 for l in cand])
        pos = {l: k for k, l in enumerate(cand)}
        fit_rows = np.flatnonzero((panel.split == "train") & np.isin(panel.group, list(fit_groups)) & (panel.kind != "jailbreak") |
                                  (tj & np.isin(panel.group, list(fit_groups))))
        r_fit, m_fit = concept_scores(panel.states[np.ix_(fit_rows, li)], bank)
        y_fit = (panel.kind[fit_rows] != "safe").astype(float)
        ev_j = np.array([i for i in jm_all if has_pad(i)])
        ev_b = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in ev_j])
        r_ej, m_ej = concept_scores(panel.states[np.ix_(ev_j, li)], bank)
        r_eb, m_eb = concept_scores(cstate[np.ix_(ev_b, li)], bank)
        # operating point from fit-side padded benign requests so evaluation negatives share the length profile
        fb = np.array([i for i in np.flatnonzero(tj & np.isin(panel.group, list(fit_groups))) if has_pad(i)])
        fbi = np.array([cid[f"{panel.rows[i]['id']}__pad_b"] for i in fb])
        r_fb, m_fb = concept_scores(cstate[np.ix_(fbi, li)], bank)
        for arm, (rl, ml) in arms.items():
            if not rl and not ml:
                continue
            rl_ = rl or [cand[0]]; ml_ = ml or [cand[0]]
            def feats(r, m):
                X, fl_, _ = build_feature_matrix(r, m, rl_, ml_, cand)
                keep = np.r_[np.ones(len(rl_), bool) if rl else np.zeros(len(rl_), bool),
                             np.ones(len(ml_), bool) if ml else np.zeros(len(ml_), bool)]
                return X[:, keep], fl_[keep]
            X_fit, fl = feats(r_fit, m_fit)
            # fit_sparse_fusion standardizes with the training mean/std itself, and its decision_function
            # expects RAW features, so nothing is standardized by hand here.
            model = fit_sparse_fusion(X_fit, y_fit, fl, config=fcfg)
            sc = lambda r, m: model.decision_function(feats(r, m)[0])
            s_j, s_b = sc(r_ej, m_ej), sc(r_eb, m_eb)
            thr = calibrate_threshold(sc(r_fb, m_fb), FPR)["threshold"]
            lab = np.r_[np.ones(len(s_j)), np.zeros(len(s_b))]
            results.append({"held_out_strategy": held, "arm": arm, "layers_risk": rl, "layers_manip": ml,
                            "n_J": len(s_j), "n_goal_groups": len({panel.group[i] for i in ev_j}),
                            "auroc_vs_padded_benign": auroc(lab, np.r_[s_j, s_b]), "tpr": float((s_j >= thr).mean()),
                            "padded_benign_fpr": float((s_b >= thr).mean()), "converged": bool(model.diagnostics["converged"])})
        print("fusion", held, "done", flush=True)
    with (out / "fusion_leave_one_strategy_out.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    summary = {}
    for arm in arms:
        rows = [r for r in results if r["arm"] == arm]
        if rows:
            summary[arm] = {"worst_strategy_tpr": min(r["tpr"] for r in rows), "mean_tpr": float(np.mean([r["tpr"] for r in rows])),
                            "worst_strategy_auroc": min(r["auroc_vs_padded_benign"] for r in rows),
                            "mean_padded_benign_fpr": float(np.mean([r["padded_benign_fpr"] for r in rows]))}
    (out / "fusion_summary.json").write_text(json.dumps({"union_layers": union, "risk_layers": risk_layers,
                                                         "manip_layers": manip_layers, "arms": summary}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
