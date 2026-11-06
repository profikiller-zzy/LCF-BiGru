"""Stage 7: nested fusion evaluation with the rebuilt (wrapper) manipulation concept, under two W(B) policies.

Outer fold = one semantic strategy held out with all of its goal groups (as in stage 5). Inside a fold only the
remaining groups are used for: risk-layer choice (taken from stage 5's nested selection, same folds), wrapper-concept
layer choice (new, inner leave-one-strategy-out below), concept directions/anchors, fusion/probe/BoW fitting, thresholds.

Manipulation concepts compared
  orig   direction from J - H, layers from stage 5's nested selection (the previous design)
  wrapB  direction from W(B) - B_pad (goal-independent wrapper concept), layers chosen here
Wrapper-concept layer choice (inside the fold): for each inner strategy s that has W(B), hold s out, refit the wrapB
direction, score M1 = AUROC(W(B)_s vs B_pad_s) and M2 = AUROC(J_s vs W(B)_s); layer score = worst M1 - |mean M2 - 0.5|;
take the best layer and the next best that is not adjacent to it. This rule was written after stage 6 had been
seen on all strategies, so the criterion itself is not blind; its application per fold is nested.

Policies for wrapped benign W(B)
  A  W(B) counts as normal: label 0 in fusion training, and fit-group W(B) enter the 5% threshold pool (mixed_wb).
  B  W(B) counts as risk (the wrapper is the manipulation attempt): label 1 in training; threshold from ordinary benign.
  none  W(B) not used in training (reference; the earlier design).
Thresholds: short (200 calibrate), mixed (calibrate + half of the long benign pool), mixed_wb (mixed + fit-group W(B)).
Held-out rows: J_m, H and B of the held-out groups, W(B) of J_m (NaN for persuasion and scenario_reframing), the other
half of the long pool. Reported rates: TPR of J and H, FPR on B and long, W(B) flag rate, AUROC J vs W(B), paired win.
Nothing from select, confirm or diagnose is read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import common as _common
from common import Panel, auroc
from dual_concept.backbone import StateCache
from dual_concept.concepts import concept_scores
from dual_concept.evaluation import calibrate_threshold
from dual_concept.fusion import FusionConfig, fit_sparse_fusion
from stage2_length_controlled import load_controls
from stage2_matrices import build_bank
from stage5_nested_eval import load_flat, metrics
from stage6_manipulation_concept import fit_directions, unit_cos
from strategy_labels import semantic_strategy

FPR = 0.05
PROBE_C = 0.01


def choose_wrap_layers(P, S_wb, cstate, cid, wb_match, wb_group, wb_sem, sem, fit_groups, tj):
    """Inner leave-one-strategy-out choice of two non-adjacent layers for the wrapB concept."""
    inner = sorted({s for s in wb_sem[[g in fit_groups for g in wb_group]]})
    M1, M2 = {s: np.zeros(len(_common.CAND())) for s in inner}, {s: np.zeros(len(_common.CAND())) for s in inner}
    wbk = {m: k for k, m in enumerate(wb_match)}
    for s in inner:
        hg = {P.group[i] for i in np.flatnonzero(tj & (sem == s))}
        fg = fit_groups - hg
        dirs = fit_directions(P, S_wb, cstate, cid, wb_match, wb_group, fg, [l - 1 for l in _common.CAND()])
        J = np.array([i for i in np.flatnonzero(tj & (sem == s) & np.isin(P.group, list(fit_groups)))
                      if f"{P.rows[i]['id']}__pad_b" in cid and i in wbk])
        if len(J) == 0:
            continue
        wb = np.array([wbk[i] for i in J]); pb = np.array([cid[f"{P.rows[i]['id']}__pad_b"] for i in J])
        one, zero = np.ones(len(J)), np.zeros(len(J))
        for k, l in enumerate(_common.CAND()):
            v, anc = dirs[l - 1]["wrapB"]
            s_J = unit_cos(P.states[J][:, l - 1], anc, v); s_W = unit_cos(S_wb[wb][:, l - 1], anc, v)
            s_P = unit_cos(cstate[pb][:, l - 1], anc, v)
            M1[s][k] = auroc(np.r_[one, zero], np.r_[s_W, s_P]); M2[s][k] = auroc(np.r_[one, zero], np.r_[s_J, s_W])
    m1 = np.min([M1[s] for s in inner], axis=0); m2 = np.mean([M2[s] for s in inner], axis=0)
    score = m1 - np.abs(m2 - 0.5)
    order = list(np.argsort(-score))
    first = order[0]
    second = next(k for k in order[1:] if abs(k - first) > 1)
    return [_common.CAND()[first], _common.CAND()[second]], {"score": score.tolist(), "inner_strategies": inner}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--len-controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--wrap-controls", default="data/dccf_v1_wrapper_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--stage5-layers", required=True, help="nested_layers.json from stage 5 (risk and orig-manip layers per fold)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s5 = json.loads(Path(a.stage5_layers).read_text())

    P = Panel(a.data, a.cache)
    cache = StateCache(Path(a.cache) / "rows")
    crow, cstate = load_controls(Path(a.len_controls), Path(a.cache), P.protocol)
    cid = {r["id"]: i for i, r in enumerate(crow)}
    wb_rows, wb_st = load_flat(Path(a.wrap_controls) / "development", ["wrapped_benign"], cache, P.protocol)
    lb_rows, lb_st = load_flat(Path(a.wrap_controls) / "development", ["long_benign"], cache, P.protocol)
    cal_rows, cal_st = load_flat(Path(a.data) / "development", ["calibrate"], cache, P.protocol)
    N_P, N_W, N_L, N_C = len(P.rows), len(wb_rows), len(lb_rows), len(cal_rows)
    rows_all = P.rows + wb_rows + lb_rows + cal_rows
    S = np.concatenate([P.states, wb_st, lb_st, cal_st])
    wb_off, lb_off, cal_off = N_P, N_P + N_W, N_P + N_W + N_L
    tokens = np.array([r.get("full_input_token_count", len(r["text"]) // 4) for r in rows_all], dtype=float)
    texts = [r["text"] for r in rows_all]
    sem = np.array([semantic_strategy(r) or "" for r in P.rows])
    wb_group = np.array([r["group_id"] for r in wb_rows]); wb_sem = np.array([r["semantic_strategy"] for r in wb_rows])
    wb_match = np.array([P.index[r["matched_to_jailbreak_id"]] for r in wb_rows])
    long_fit = np.array([int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 2 == 0 for r in lb_rows])
    long_fit_idx = lb_off + np.flatnonzero(long_fit); long_eval_idx = lb_off + np.flatnonzero(~long_fit)
    cal_idx = cal_off + np.arange(N_C)
    train_groups = {g for g, s, k in zip(P.group, P.split, P.kind) if s == "train" and k == "harmful"}
    tj = (P.split == "train") & (P.kind == "jailbreak")
    strategies = sorted(set(sem[tj]))

    results, layer_log = [], {}
    for held in strategies:
        if a.only and held not in a.only:
            continue
        if held not in s5 or "error" in s5[held]:
            print(f"[{held}] no stage-5 layers, skipped"); continue
        held_groups = {P.group[i] for i in np.flatnonzero(tj & (sem == held))}
        fit_groups = train_groups - held_groups
        fit_p = np.flatnonzero((P.split == "train") & np.isin(P.group, list(fit_groups)))
        fit_wb = wb_off + np.flatnonzero(np.isin(wb_group, list(fit_groups)))
        rl, ml_orig = s5[held]["risk"], s5[held]["manip"]
        ml_wrap, info = choose_wrap_layers(P, wb_st, cstate, cid, wb_match, wb_group, wb_sem, sem, fit_groups, tj)
        layer_log[held] = {"risk": rl, "orig_manip": ml_orig, "wrap_manip": ml_wrap}
        print(f"[{held}] risk {rl} | orig-manip {ml_orig} | wrapB layers {ml_wrap}", flush=True)

        need = sorted(set(rl) | set(ml_orig) | set(ml_wrap))
        nidx = [l - 1 for l in need]
        bank, _, _ = build_bank(P, fit_groups, nidx)
        r_all, m_orig_all = concept_scores(S[:, nidx], bank)
        col = {l: k for k, l in enumerate(need)}
        wrap_dirs = fit_directions(P, wb_st, cstate, cid, wb_match, wb_group, fit_groups, [l - 1 for l in ml_wrap])
        wrap_all = {l: unit_cos(S[:, l - 1], wrap_dirs[l - 1]["wrapB"][1], wrap_dirs[l - 1]["wrapB"][0]) for l in ml_wrap}

        feats = {
            "risk": (np.stack([r_all[:, col[l]] for l in rl], 1), rl),
            "orig": (np.stack([m_orig_all[:, col[l]] for l in ml_orig], 1), ml_orig),
            "wrap": (np.stack([wrap_all[l] for l in ml_wrap], 1), ml_wrap)}

        def block(names):
            X = np.concatenate([feats[n][0] for n in names], 1)
            fl = np.concatenate([np.array(feats[n][1]) for n in names])
            return X, fl

        def eval_set(split):
            Jm = np.flatnonzero((P.kind == "jailbreak") & (sem == held) & (P.split == split))
            groups = sorted({P.group[i] for i in Jm})
            idx = lambda sfx: np.array([P.index[f"{g}_{sfx}"] for g in groups if f"{g}_{sfx}" in P.index])
            jset = set(Jm.tolist()); wsel = np.array([k for k in range(N_W) if wb_match[k] in jset], dtype=int)
            return {"J": Jm, "groups": groups, "B": idx("b"), "H": idx("h"), "WB": wb_off + wsel,
                    "J_wb": wb_match[wsel] if len(wsel) else np.array([], int), "WB_of": wb_off + wsel, "LONG": long_eval_idx}
        E = eval_set("train")
        y_p = (P.kind[fit_p] != "safe").astype(float)
        union_h = sorted(set(rl) | set(ml_wrap))
        Hs = S[:, [l - 1 for l in union_h]].reshape(len(S), -1)

        tags = {"none": (fit_p, y_p), "A": (np.r_[fit_p, fit_wb], np.r_[y_p, np.zeros(len(fit_wb))]),
                "B": (np.r_[fit_p, fit_wb], np.r_[y_p, np.ones(len(fit_wb))])}
        score_sets = {("length_rule", "none"): tokens.copy()}
        for tag, (idx, y) in tags.items():
            vec = CountVectorizer(lowercase=True, min_df=2)
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(vec.fit_transform([texts[i] for i in idx]), y)
            score_sets[("bow_lr", tag)] = lr.decision_function(vec.transform(texts))
            mu, sd = Hs[idx].mean(0), Hs[idx].std(0); sd = np.where(sd > 0, sd, 1.0)
            Z = (Hs - mu) / sd
            score_sets[("hidden_probe", tag)] = LogisticRegression(max_iter=500, C=PROBE_C).fit(Z[idx], y).decision_function(Z)
            for arm, names in (("dccf_risk", ["risk"]), ("dccf_wrap_only", ["wrap"]), ("dccf_orig_fusion", ["risk", "orig"]),
                               ("dccf_wrap_fusion", ["risk", "wrap"])):
                X, fl = block(names)
                model = fit_sparse_fusion(X[idx], y, fl, config=FusionConfig())
                if not model.diagnostics["converged"]:
                    print(f"  WARNING no convergence: {arm} {tag}")
                score_sets[(arm, tag)] = model.decision_function(X)
                score_sets[(arm + "__converged", tag)] = np.array([model.diagnostics["converged"]])
                score_sets[(arm + "__coef", tag)] = np.round(model.coef, 4)
        pools = {"short": cal_idx, "mixed": np.r_[cal_idx, long_fit_idx], "mixed_wb": np.r_[cal_idx, long_fit_idx, fit_wb]}
        for (arm, tag), sc in score_sets.items():
            if arm.endswith("__converged") or arm.endswith("__coef"):
                continue
            conv = bool(score_sets[(arm + "__converged", tag)][0]) if (arm + "__converged", tag) in score_sets else True
            coef = score_sets.get((arm + "__coef", tag))
            for mode, pool in pools.items():
                thr = calibrate_threshold(sc[pool], FPR)["threshold"]
                m = metrics(sc, thr, E)
                m["rate_WB"] = float((sc[E["WB"]] >= thr).mean()) if len(E["WB"]) else float("nan")
                results.append({"held_out": held, "arm": arm, "policy": tag, "threshold_mode": mode, "converged": conv,
                                "coef": "" if coef is None else json.dumps(coef.tolist()),
                                "risk_layers": rl, "orig_manip_layers": ml_orig, "wrap_layers": ml_wrap, **m})
        print(f"[{held}] done", flush=True)

    keys = ["held_out", "arm", "policy", "threshold_mode", "converged", "n_J", "n_groups", "tpr_J", "tpr_H", "fpr_B", "fpr_long",
            "rate_WB", "auroc_J_vs_B", "auroc_J_vs_WB", "paired_win_J_over_WB", "risk_layers", "orig_manip_layers", "wrap_layers", "coef"]
    with (out / "fusion_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(results)
    (out / "fold_layers.json").write_text(json.dumps(layer_log, indent=1))
    print("rows", len(results))


if __name__ == "__main__":
    main()
