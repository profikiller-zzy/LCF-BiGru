"""Stage 8: the nested evaluation on any backbone, policy A (wrapped benign counts as normal).

Reads layer count and width from the extraction protocol of the given cache, so nothing is fixed to 27 layers.
Per model it runs, for each held-out semantic strategy (the strategy's goal groups removed from every fitted step):
  * nested risk-layer selection and J-minus-H layer selection (`select_layers` over length-matched padded negatives)
  * wrapper-concept (W(B) minus length-matched B) layer choice, by inner leave-one-strategy-out
  * length rule, bag-of-words, a linear probe on the chosen layers, DCCF risk, DCCF J-minus-H fusion, DCCF wrapper fusion
Policy A: wrapped benign W(B) of the fit groups is a label-0 training negative AND enters the 5% threshold pool, as in
stage 7. Thresholds: `short` (calibrate), `mixed_wb` (calibrate + half of the long-benign pool + fit-group W(B)).
Layer positions are also reported as relative depth (layer / num_hidden_layers) so models of different depth compare.
select, confirm and diagnose are not read.
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

import common
from dual_concept.backbone import StateCache
from dual_concept.concepts import concept_scores
from dual_concept.evaluation import calibrate_threshold
from dual_concept.fusion import FusionConfig, fit_sparse_fusion
from dual_concept.selection import SelectionConfig, select_layers
from common import Panel, auroc
import stage2_length_controlled as s2lc
import stage2_matrices as s2m
import stage5_nested_eval as s5
import stage6_manipulation_concept as s6
import stage7_nested_fusion_wrapper as s7
from strategy_labels import semantic_strategy

FPR = 0.05
PROBE_C = 0.01


def configure_layers(protocol) -> list[int]:
    return list(common.set_layers(protocol["candidate_layers"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--len-controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--wrap-controls", default="data/dccf_v1_wrapper_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    proto = json.loads((Path(a.cache) / "protocol.json").read_text())
    layers = configure_layers(proto)
    depth = int(proto.get("num_hidden_layers", len(layers) + 1))   # real caches always carry it
    P = Panel(a.data, a.cache)
    cache = StateCache(Path(a.cache) / "rows")
    crow, cstate = s2lc.load_controls(Path(a.len_controls), Path(a.cache), P.protocol)
    cid = {r["id"]: i for i, r in enumerate(crow)}
    wb_rows, wb_st = s5.load_flat(Path(a.wrap_controls) / "development", ["wrapped_benign"], cache, P.protocol)
    lb_rows, lb_st = s5.load_flat(Path(a.wrap_controls) / "development", ["long_benign"], cache, P.protocol)
    cal_rows, cal_st = s5.load_flat(Path(a.data) / "development", ["calibrate"], cache, P.protocol)
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
    cfg = SelectionConfig()

    results, layer_log = [], {}
    for held in strategies:
        if a.only and held not in a.only:
            continue
        held_groups = {P.group[i] for i in np.flatnonzero(tj & (sem == held))}
        fit_groups = train_groups - held_groups
        fit_p = np.flatnonzero((P.split == "train") & np.isin(P.group, list(fit_groups)))
        fit_wb = wb_off + np.flatnonzero(np.isin(wb_group, list(fit_groups)))
        try:
            nested = s5.inner_select(P, sem, cid, cstate, fit_groups, cfg)
        except ValueError as e:
            print(f"[{held}] selection failed: {e}"); layer_log[held] = {"error": str(e)}; continue
        rl, ml_orig = nested["risk"], nested["manip"]
        ml_wrap, _ = s7.choose_wrap_layers(P, wb_st, cstate, cid, wb_match, wb_group, wb_sem, sem, fit_groups, tj)
        layer_log[held] = {"risk": rl, "orig_manip": ml_orig, "wrap_manip": ml_wrap,
                           "relative_depth": {"risk": [round(l / depth, 2) for l in rl], "orig_manip": [round(l / depth, 2) for l in ml_orig],
                                              "wrap_manip": [round(l / depth, 2) for l in ml_wrap]}}
        print(f"[{held}] risk {rl} orig {ml_orig} wrap {ml_wrap}", flush=True)

        need = sorted(set(rl) | set(ml_orig) | set(ml_wrap)); nidx = [l - 1 for l in need]
        bank, _, _ = s2m.build_bank(P, fit_groups, nidx)
        r_all, m_orig_all = concept_scores(S[:, nidx], bank)
        col = {l: k for k, l in enumerate(need)}
        wd = s6.fit_directions(P, wb_st, cstate, cid, wb_match, wb_group, fit_groups, [l - 1 for l in ml_wrap])
        wrap_all = {l: s6.unit_cos(S[:, l - 1], wd[l - 1]["wrapB"][1], wd[l - 1]["wrapB"][0]) for l in ml_wrap}
        feats = {"risk": (np.stack([r_all[:, col[l]] for l in rl], 1), rl),
                 "orig": (np.stack([m_orig_all[:, col[l]] for l in ml_orig], 1), ml_orig),
                 "wrap": (np.stack([wrap_all[l] for l in ml_wrap], 1), ml_wrap)}
        block = lambda names: (np.concatenate([feats[n][0] for n in names], 1),
                               np.concatenate([np.array(feats[n][1]) for n in names]))

        Jm = np.flatnonzero((P.kind == "jailbreak") & (sem == held) & (P.split == "train"))
        groups = sorted({P.group[i] for i in Jm})
        idx = lambda sfx: np.array([P.index[f"{g}_{sfx}"] for g in groups if f"{g}_{sfx}" in P.index])
        jset = set(Jm.tolist()); wsel = np.array([k for k in range(N_W) if wb_match[k] in jset], dtype=int)
        E = {"J": Jm, "groups": groups, "B": idx("b"), "H": idx("h"), "WB": wb_off + wsel,
             "J_wb": wb_match[wsel] if len(wsel) else np.array([], int), "WB_of": wb_off + wsel, "LONG": long_eval_idx}
        y_p = (P.kind[fit_p] != "safe").astype(float)
        idx_fit = np.r_[fit_p, fit_wb]; y_fit = np.r_[y_p, np.zeros(len(fit_wb))]          # policy A
        union_h = sorted(set(rl) | set(ml_wrap))
        Hs = S[:, [l - 1 for l in union_h]].reshape(len(S), -1)

        scores = {"length_rule": tokens.copy()}
        vec = CountVectorizer(lowercase=True, min_df=2)
        scores["bow_lr"] = LogisticRegression(max_iter=2000, C=1.0).fit(vec.fit_transform([texts[i] for i in idx_fit]), y_fit).decision_function(vec.transform(texts))
        mu, sd = Hs[idx_fit].mean(0), Hs[idx_fit].std(0); sd = np.where(sd > 0, sd, 1.0)
        Z = (Hs - mu) / sd
        scores["hidden_probe"] = LogisticRegression(max_iter=500, C=PROBE_C).fit(Z[idx_fit], y_fit).decision_function(Z)
        conv, coefs = {}, {}
        for arm, names in (("dccf_risk", ["risk"]), ("dccf_orig_fusion", ["risk", "orig"]), ("dccf_wrap_fusion", ["risk", "wrap"])):
            X, fl = block(names)
            m = fit_sparse_fusion(X[idx_fit], y_fit, fl, config=FusionConfig())
            scores[arm] = m.decision_function(X); conv[arm] = bool(m.diagnostics["converged"]); coefs[arm] = np.round(m.coef, 4).tolist()
            if not conv[arm]:
                print(f"  WARNING no convergence: {arm}")
        pools = {"short": cal_idx, "mixed_wb": np.r_[cal_idx, long_fit_idx, fit_wb]}
        for arm, sc in scores.items():
            for mode, pool in pools.items():
                thr = calibrate_threshold(sc[pool], FPR)["threshold"]
                m = s5.metrics(sc, thr, E)
                m["rate_WB"] = float((sc[E["WB"]] >= thr).mean()) if len(E["WB"]) else float("nan")
                results.append({"model": a.model_name, "held_out": held, "arm": arm, "threshold_mode": mode,
                                "converged": conv.get(arm, True), "coef": json.dumps(coefs.get(arm, "")),
                                "risk_layers": rl, "orig_manip_layers": ml_orig, "wrap_layers": ml_wrap, **m})
        print(f"[{held}] done", flush=True)

    keys = ["model", "held_out", "arm", "threshold_mode", "converged", "n_J", "n_groups", "tpr_J", "tpr_H", "fpr_B", "fpr_long",
            "rate_WB", "auroc_J_vs_B", "auroc_J_vs_WB", "paired_win_J_over_WB", "risk_layers", "orig_manip_layers", "wrap_layers", "coef"]
    with (out / "cross_model_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(results)
    (out / "fold_layers.json").write_text(json.dumps({"num_hidden_layers": depth, "hidden_size": proto["hidden_size"], "folds": layer_log}, indent=1))
    print("rows", len(results))


if __name__ == "__main__":
    main()
