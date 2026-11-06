"""Stage 5: nested leave-one-strategy-out evaluation with baselines and wrapper-matched negatives.

Outer fold = one semantic strategy m held out together with every goal group that contains an m attack.
Everything below sees only the remaining groups: concept directions, anchors, the inner layer selection,
fusion/probe/BoW fitting, and the 5% operating threshold. Evaluation rows are the held-out groups' J_m and
four kinds of negatives:
  B     the plain benign request of the same goal group
  W(B)  the same real wrapper as J with the benign request swapped in (differs from J only in the goal)
  long  long natural benign inputs (half of the pool is held out for evaluation only)
  H     direct harmful requests of the held-out groups are reported as positives (TPR_H)
Arms (all share folds, thresholds and negatives):
  length_rule, bow_lr, hidden_probe (logistic regression on the same layers), dccf_risk, dccf_manip, dccf_fusion,
  dccf_fusion_lenproj (directions made orthogonal to a padding/length direction).
Each trainable arm is run twice: without and with W(B) of the fit groups as extra negatives.
Two operating points: `short` (200 calibrate requests) and `mixed` (calibrate + the long-benign fit half).
W(B) is treated as a wrapper false-alarm diagnostic (label 0), following the architecture figure's W_m(B_i)
"wrapper false alarm and confound" control; whether a wrapped benign request should count as risk is a policy choice.
No confirm data is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

import common as _common
from common import Panel, auroc
from dual_concept.backbone import StateCache
from dual_concept.concepts import ConceptBank, concept_scores
from dual_concept.data import read_jsonl
from dual_concept.evaluation import calibrate_threshold
from dual_concept.fusion import FusionConfig, build_feature_matrix, fit_sparse_fusion
from dual_concept.selection import SelectionConfig, select_layers
from stage2_length_controlled import load_controls
from stage2_matrices import build_bank
from strategy_labels import semantic_strategy

FPR = 0.05
PROBE_C = 0.01          # fixed in advance, never tuned on evaluation data


def load_flat(root: Path, files, cache: StateCache, proto):
    rows, st = [], []
    for f in files:
        for r in read_jsonl(root / f"{f}.jsonl"):
            s = cache.load(r["id"], r["text"], proto, expected_shape=(len(_common.LAYERS), int(proto["hidden_size"])))
            if s is None:
                raise FileNotFoundError(f"Missing state for {r['id']}")
            rows.append(r); st.append(s)
    return rows, np.stack(st).astype(np.float32)


def inner_select(P: Panel, sem, cid, cstate, fit_groups, cfg):
    """Stage-3 selection (padded length-matched negatives) using only fit_groups."""
    tj = (P.split == "train") & (P.kind == "jailbreak")
    strategies = sorted(set(sem[tj & np.isin(P.group, list(fit_groups))]))
    has_pad = lambda i: f"{P.rows[i]['id']}__pad_h" in cid
    folds = {"risk": [], "manip": []}
    for held in strategies:
        jm = np.array([i for i in np.flatnonzero(tj & (sem == held) & np.isin(P.group, list(fit_groups))) if has_pad(i)])
        if len(jm) == 0:
            continue
        fg = fit_groups - {P.group[i] for i in jm}
        bank, _, _ = build_bank(P, fg, _common.LI_ALL())
        fj = np.array([i for i in np.flatnonzero(tj & np.isin(P.group, list(fg))) if has_pad(i)])
        rb = np.array([cid[f"{P.rows[i]['id']}__pad_b"] for i in fj]); rh = np.array([cid[f"{P.rows[i]['id']}__pad_h"] for i in fj])
        r_ref, _ = concept_scores(cstate[rb], bank); _, m_ref = concept_scores(cstate[rh], bank)
        thr_r = np.array([calibrate_threshold(r_ref[:, k], FPR)["threshold"] for k in range(len(_common.CAND()))])
        thr_m = np.array([calibrate_threshold(m_ref[:, k], FPR)["threshold"] for k in range(len(_common.CAND()))])
        r_j, m_j = concept_scores(P.states[jm], bank)
        pb = np.array([cid[f"{P.rows[i]['id']}__pad_b"] for i in jm]); ph = np.array([cid[f"{P.rows[i]['id']}__pad_h"] for i in jm])
        r_pb, m_pb = concept_scores(cstate[pb], bank); _, m_ph = concept_scores(cstate[ph], bank)
        n = len(jm)
        folds["risk"].append({"scores": np.vstack([r_j, r_pb]), "labels": np.r_[np.ones(n), np.zeros(n)],
                              "safe_mask": np.r_[np.zeros(n, bool), np.ones(n, bool)],
                              "attack_groups": np.full(2 * n, held), "thresholds": thr_r})
        folds["manip"].append({"scores": np.vstack([m_j, m_ph, m_pb]), "labels": np.r_[np.ones(n), np.zeros(n), -np.ones(n)],
                               "safe_mask": np.r_[np.zeros(2 * n, bool), np.ones(n, bool)],
                               "attack_groups": np.full(3 * n, held), "thresholds": thr_m})
    out = {}
    for branch in ("risk", "manip"):
        res = select_layers(folds[branch], _common.CAND(), cfg)
        out[branch] = res.layers
    return out


def project_bank(bank: ConceptBank, u: np.ndarray) -> ConceptBank:
    def proj(v):
        w = v - (v * u).sum(1, keepdims=True) * u
        return w / np.linalg.norm(w, axis=1, keepdims=True)
    return ConceptBank(layer_ids=bank.layer_ids, risk_direction=proj(bank.risk_direction),
                       manipulation_direction=proj(bank.manipulation_direction),
                       safe_anchor=bank.safe_anchor, harmful_anchor=bank.harmful_anchor)


def metrics(sc, thr, E):
    out = {}
    J = E["J"]
    out["tpr_J"] = float((sc[J] >= thr).mean()); out["n_J"] = len(J); out["n_groups"] = len(E["groups"])
    out["tpr_H"] = float((sc[E["H"]] >= thr).mean()) if len(E["H"]) else float("nan")
    for name, idx, jsub in (("B", E["B"], J), ("WB", E["WB"], E["J_wb"]), ("long", E["LONG"], J)):
        if len(idx) == 0 or len(jsub) == 0:
            out[f"fpr_{name}"] = out[f"auroc_J_vs_{name}"] = float("nan"); continue
        out[f"fpr_{name}"] = float((sc[idx] >= thr).mean())
        out[f"auroc_J_vs_{name}"] = auroc(np.r_[np.ones(len(jsub)), np.zeros(len(idx))], np.r_[sc[jsub], sc[idx]])
    if len(E["J_wb"]):
        d = sc[E["J_wb"]] - sc[E["WB_of"]]
        out["paired_win_J_over_WB"] = float(((d > 0) + 0.5 * (d == 0)).mean())
    else:
        out["paired_win_J_over_WB"] = float("nan")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--len-controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--wrap-controls", default="data/dccf_v1_wrapper_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="restrict outer folds to these strategies (debug)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

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
    src = np.array(["panel"] * N_P + ["wb"] * N_W + ["long"] * N_L + ["cal"] * N_C)
    wb_off, lb_off, cal_off = N_P, N_P + N_W, N_P + N_W + N_L
    tokens = np.array([r.get("full_input_token_count", len(r["text"]) // 4) for r in rows_all], dtype=float)
    texts = [r["text"] for r in rows_all]
    sem = np.array([semantic_strategy(r) or "" for r in P.rows])
    sem_all = np.concatenate([sem, np.array([r.get("semantic_strategy") or "" for r in wb_rows]), np.array([""] * (N_L + N_C))])
    wb_group = np.array([r["group_id"] for r in wb_rows])
    wb_match = np.array([P.index[r["matched_to_jailbreak_id"]] for r in wb_rows])
    long_fit = np.array([int(hashlib.sha256(r["id"].encode()).hexdigest(), 16) % 2 == 0 for r in lb_rows])
    long_idx_fit = lb_off + np.flatnonzero(long_fit); long_idx_eval = lb_off + np.flatnonzero(~long_fit)
    cal_idx = cal_off + np.arange(N_C)
    train_groups = {g for g, s, k in zip(P.group, P.split, P.kind) if s == "train" and k == "harmful"}
    tj = (P.split == "train") & (P.kind == "jailbreak")
    strategies = sorted(set(sem[tj]))
    cfg = SelectionConfig()

    all_rows, layer_log = [], {}
    for held in strategies:
        if a.only and held not in a.only:
            continue
        held_groups = {P.group[i] for i in np.flatnonzero(tj & (sem == held))}
        fit_groups = train_groups - held_groups
        fit_p = np.flatnonzero((P.split == "train") & np.isin(P.group, list(fit_groups)))
        fit_wb = wb_off + np.flatnonzero(np.isin(wb_group, list(fit_groups)))
        print(f"[{held}] fit groups {len(fit_groups)}, held-out groups {len(held_groups)}", flush=True)

        # ---- nested layer selection on the fit groups only ----
        try:
            layers = inner_select(P, sem, cid, cstate, fit_groups, cfg)
        except ValueError as e:
            print("  selection failed:", e); layer_log[held] = {"error": str(e)}; continue
        rl, ml = layers["risk"], layers["manip"]
        union = sorted(set(rl) | set(ml)); uidx = [l - 1 for l in union]
        layer_log[held] = {"risk": rl, "manip": ml}
        print(f"  nested selection: risk {rl} manip {ml}", flush=True)

        # ---- evaluation sets ----
        def eval_set(split):
            Jm = np.flatnonzero((P.kind == "jailbreak") & (sem == held) & (P.split == split))
            groups = sorted({P.group[i] for i in Jm})
            idx = lambda suffix: np.array([P.index[f"{g}_{suffix}"] for g in groups if f"{g}_{suffix}" in P.index])
            jset = set(Jm.tolist())
            wb_sel = np.array([k for k in range(N_W) if wb_match[k] in jset], dtype=int)
            return {"J": Jm, "groups": groups, "B": idx("b"), "H": idx("h"),
                    "WB": wb_off + wb_sel, "J_wb": wb_match[wb_sel] if len(wb_sel) else np.array([], int),
                    "WB_of": wb_off + wb_sel, "LONG": long_idx_eval}
        evals = {"train_heldout": eval_set("train"), "select": eval_set("select")}

        # ---- concept bank on fit groups (union layers), plus length-projected copy ----
        bank, _, _ = build_bank(P, fit_groups, uidx)
        fj = [i for i in np.flatnonzero(tj & np.isin(P.group, list(fit_groups))) if f"{P.rows[i]['id']}__pad_h" in cid]
        pad_h = np.array([cid[f"{P.rows[i]['id']}__pad_h"] for i in fj]); h_plain = np.array([P.index[f"{P.group[i]}_h"] for i in fj])
        d = (cstate[pad_h][:, uidx] - P.states[h_plain][:, uidx]).mean(0)
        u = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        bank_lp = project_bank(bank, u)
        y_p = (P.kind[fit_p] != "safe").astype(float)

        def dccf_scores(bk, rl_, ml_, with_wb):
            r_all, m_all = concept_scores(S[:, uidx], bk)
            cols = {l: k for k, l in enumerate(union)}
            use_r, use_m = bool(rl_), bool(ml_)
            rr = rl_ or [union[0]]; mm = ml_ or [union[0]]
            X, fl, _ = build_feature_matrix(r_all, m_all, rr, mm, union)
            keep = np.r_[np.ones(len(rr), bool) if use_r else np.zeros(len(rr), bool),
                         np.ones(len(mm), bool) if use_m else np.zeros(len(mm), bool)]
            X, fl = X[:, keep], fl[keep]
            idx = np.r_[fit_p, fit_wb] if with_wb else fit_p
            y = np.r_[y_p, np.zeros(len(fit_wb))] if with_wb else y_p
            model = fit_sparse_fusion(X[idx], y, fl, config=FusionConfig())
            return model.decision_function(X), bool(model.diagnostics["converged"])

        score_sets = {}
        score_sets[("length_rule", "none")] = tokens.copy()
        for with_wb in (False, True):
            tag = "wb" if with_wb else "none"
            idx = np.r_[fit_p, fit_wb] if with_wb else fit_p
            y = np.r_[y_p, np.zeros(len(fit_wb))] if with_wb else y_p
            vec = CountVectorizer(lowercase=True, min_df=2)
            Xb = vec.fit_transform([texts[i] for i in idx])
            lr = LogisticRegression(max_iter=2000, C=1.0).fit(Xb, y)
            score_sets[("bow_lr", tag)] = lr.decision_function(vec.transform(texts))
            feats = S[:, uidx].reshape(len(S), -1)
            mu, sd = feats[idx].mean(0), feats[idx].std(0); sd = np.where(sd > 0, sd, 1.0)
            Z = (feats - mu) / sd
            pr = LogisticRegression(max_iter=500, C=PROBE_C).fit(Z[idx], y)
            score_sets[("hidden_probe", tag)] = pr.decision_function(Z)
            for name, bk, r_, m_ in (("dccf_risk", bank, rl, []), ("dccf_manip", bank, [], ml),
                                     ("dccf_fusion", bank, rl, ml), ("dccf_fusion_lenproj", bank_lp, rl, ml)):
                sc, conv = dccf_scores(bk, r_, m_, with_wb)
                if not conv:
                    print("  WARNING fusion did not converge:", name, tag)
                score_sets[(name, tag)] = sc

        thr_pool = {"short": cal_idx, "mixed": np.r_[cal_idx, long_idx_fit]}
        for (arm, tag), sc in score_sets.items():
            for mode, pool in thr_pool.items():
                thr = calibrate_threshold(sc[pool], FPR)["threshold"]
                for ev_name, E in evals.items():
                    if len(E["J"]) == 0:
                        continue
                    m = metrics(sc, thr, E)
                    all_rows.append({"held_out": held, "eval": ev_name, "arm": arm, "train_negatives": tag,
                                     "threshold_mode": mode, "risk_layers": rl, "manip_layers": ml, **m})
        print(f"  [{held}] done", flush=True)

    import csv
    keys = ["held_out", "eval", "arm", "train_negatives", "threshold_mode", "n_J", "n_groups", "tpr_J", "tpr_H", "fpr_B",
            "fpr_WB", "fpr_long", "auroc_J_vs_B", "auroc_J_vs_WB", "auroc_J_vs_long", "paired_win_J_over_WB",
            "risk_layers", "manip_layers"]
    with (out / "nested_results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(all_rows)
    (out / "nested_layers.json").write_text(json.dumps(layer_log, indent=1))
    print("rows", len(all_rows))


if __name__ == "__main__":
    main()
