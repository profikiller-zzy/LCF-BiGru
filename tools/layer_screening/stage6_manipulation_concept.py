"""Stage 6: rebuild the manipulation concept so it measures the wrapper, not the goal.

Why. In stage 5 the manipulation readout (direction from J minus H) could not separate a jailbreak J from the same
wrapper around a benign request W(B) (AUROC 0.44 to 0.49): J minus H mixes "a wrapper is present" with "the goal is
harmful". A manipulation concept should be goal-independent: high for wrapped inputs whatever sits inside, and tell
nothing about the goal. This stage builds candidate directions and checks exactly that.

Constructions (all rank-one on uncentered differences, same code as `dual_concept.concepts`):
  orig      J - H                        (current; wrapper + harmful goal, length-confounded)
  unmatched W(B) - B                     (wrapper only, but wrapped inputs are longer than B)
  wrapB     W(B) - B_pad                 (wrapper only, length matched; the proposed manipulation concept)
  wrapBH    stack of W(B) - B_pad and J - H_pad (wrapper from both goals, length matched)
Anchor for the cosine readout: the mean of the corresponding reference group (B_pad for wrapB, H for orig, ...).

Measurements on a held-out strategy m (its goal groups are removed from direction fitting):
  M1 wrapper_vs_plain  AUROC(W(B)_m vs B_pad_m)   wrapper detected regardless of goal      want high
  M2 goal_in_wrapper   AUROC(J_m vs W(B)_m)        does the readout also see the goal?      want ~0.5 for a pure wrapper concept
  M3 wrapped_vs_padH   AUROC(J_m vs H_pad_m)       the earlier stage-2 test                   want high
  M4 plain_goal        AUROC(H_m vs B_m)           risk of an unwrapped request              reported for contrast
Only strategies that have W(B) (persona, fiction, instruction-control, adversarial-suffix, rule-suffix) are scored.
persuasion and scenario_reframing have no extractable wrapper, so they cannot test a wrapper concept; they are listed
as untested rather than dropped silently.
Nothing here uses select, confirm or calibrate rows to fit a direction.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import LAYERS, Panel, auroc
from dual_concept.backbone import StateCache
from dual_concept.concepts import anchored_cosine, rank_one_direction
from stage2_length_controlled import load_controls
from stage5_nested_eval import load_flat
from strategy_labels import semantic_strategy

CONSTRUCTIONS = ("orig", "unmatched", "wrapB", "wrapBH")


def unit_cos(states, anchor, direction):
    return anchored_cosine(states[:, None, :], anchor[None, :], direction[None, :])[:, 0]


def fit_directions(P, S_wb, cstate, cid, wb_match, wb_group, fit_groups, layer_idx):
    """Directions/anchors for every construction on the given fit goal groups, per requested layer."""
    gset = set(fit_groups)
    tj = (P.split == "train") & (P.kind == "jailbreak")
    J = [i for i in np.flatnonzero(tj & np.isin(P.group, list(gset))) if f"{P.rows[i]['id']}__pad_h" in cid]
    h_plain = np.array([P.index[f"{P.group[i]}_h"] for i in J])
    b_plain = np.array([P.index[f"{P.group[i]}_b"] for i in J])
    pad_h = np.array([cid[f"{P.rows[i]['id']}__pad_h"] for i in J])
    pad_b = np.array([cid[f"{P.rows[i]['id']}__pad_b"] for i in J])
    wbk = {m: k for k, m in enumerate(wb_match)}          # jailbreak panel index -> W(B) row
    keep = [n for n, i in enumerate(J) if i in wbk]
    J = np.array(J)[keep]; h_plain, b_plain, pad_h, pad_b = h_plain[keep], b_plain[keep], pad_h[keep], pad_b[keep]
    wb = np.array([wbk[i] for i in J])
    out = {}
    for k, l in enumerate(layer_idx):
        Jh = P.states[J][:, l].astype(np.float64); Hh = P.states[h_plain][:, l].astype(np.float64)
        Bh = P.states[b_plain][:, l].astype(np.float64)
        Wb = S_wb[wb][:, l].astype(np.float64); PB = cstate[pad_b][:, l].astype(np.float64); PH = cstate[pad_h][:, l].astype(np.float64)
        d = {"orig": Jh - Hh, "unmatched": Wb - Bh, "wrapB": Wb - PB, "wrapBH": np.vstack([Wb - PB, Jh - PH])}
        anchors = {"orig": Hh.mean(0), "unmatched": Bh.mean(0), "wrapB": PB.mean(0), "wrapBH": np.vstack([PB, PH]).mean(0)}
        out[l] = {c: (rank_one_direction(d[c])[0], anchors[c]) for c in CONSTRUCTIONS}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--len-controls", default="data/dccf_v1_length_controls")
    ap.add_argument("--wrap-controls", default="data/dccf_v1_wrapper_controls")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    P = Panel(a.data, a.cache)
    cache = StateCache(Path(a.cache) / "rows")
    crow, cstate = load_controls(Path(a.len_controls), Path(a.cache), P.protocol)
    cid = {r["id"]: i for i, r in enumerate(crow)}
    wb_rows, S_wb = load_flat(Path(a.wrap_controls) / "development", ["wrapped_benign"], cache, P.protocol)
    wb_match = np.array([P.index[r["matched_to_jailbreak_id"]] for r in wb_rows])
    wb_group = np.array([r["group_id"] for r in wb_rows])
    wb_sem = np.array([r["semantic_strategy"] for r in wb_rows])
    sem = np.array([semantic_strategy(r) or "" for r in P.rows])
    train_groups = {g for g, s, k in zip(P.group, P.split, P.kind) if s == "train" and k == "harmful"}
    tj = (P.split == "train") & (P.kind == "jailbreak")
    testable = sorted(set(wb_sem[[r["split"] == "train" for r in wb_rows]]))
    layer_idx = [l - 1 for l in a.layers]
    rows = []
    for held in testable:
        hg = {P.group[i] for i in np.flatnonzero(tj & (sem == held))}
        fit = train_groups - hg
        dirs = fit_directions(P, S_wb, cstate, cid, wb_match, wb_group, fit, layer_idx)
        J = np.array([i for i in np.flatnonzero(tj & (sem == held)) if f"{P.rows[i]['id']}__pad_h" in cid])
        wbk = {m: k for k, m in enumerate(wb_match)}
        J = np.array([i for i in J if i in wbk])
        if len(J) == 0:
            continue
        wb = np.array([wbk[i] for i in J])
        h_plain = np.array([P.index[f"{P.group[i]}_h"] for i in J]); b_plain = np.array([P.index[f"{P.group[i]}_b"] for i in J])
        pad_h = np.array([cid[f"{P.rows[i]['id']}__pad_h"] for i in J]); pad_b = np.array([cid[f"{P.rows[i]['id']}__pad_b"] for i in J])
        n = len(J); one, zero = np.ones(n), np.zeros(n)
        for l in layer_idx:
            for c in CONSTRUCTIONS:
                v, anc = dirs[l][c]
                s_J = unit_cos(P.states[J][:, l], anc, v); s_WB = unit_cos(S_wb[wb][:, l], anc, v)
                s_PB = unit_cos(cstate[pad_b][:, l], anc, v); s_PH = unit_cos(cstate[pad_h][:, l], anc, v)
                s_H = unit_cos(P.states[h_plain][:, l], anc, v); s_B = unit_cos(P.states[b_plain][:, l], anc, v)
                rows.append({"held_out": held, "construction": c, "layer": l + 1, "n": n,
                             "n_groups": len(set(P.group[J])),
                             "M1_wrapper_vs_plain": auroc(np.r_[one, zero], np.r_[s_WB, s_PB]),
                             "M2_goal_in_wrapper": auroc(np.r_[one, zero], np.r_[s_J, s_WB]),
                             "M3_wrapped_vs_padH": auroc(np.r_[one, zero], np.r_[s_J, s_PH]),
                             "M4_plain_goal": auroc(np.r_[one, zero], np.r_[s_H, s_B]),
                             "mean_J": float(s_J.mean()), "mean_WB": float(s_WB.mean()),
                             "mean_PB": float(s_PB.mean()), "mean_PH": float(s_PH.mean())})
        print(f"[{held}] done", flush=True)
    with (out / "concept_profile.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "columns.json").write_text(json.dumps({"testable": testable,
                                                  "untestable_no_wrapper": ["persuasion", "scenario_reframing"]}))
    print("rows", len(rows))


if __name__ == "__main__":
    main()
