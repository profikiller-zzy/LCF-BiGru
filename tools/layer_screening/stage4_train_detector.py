"""Stage 4: train and freeze the cross-layer detector from the layers chosen in stage 3.

Roles
  train      builds the final concept directions/anchors and fits the sparse linear fusion head.
  calibrate  (200 ordinary benign requests, independent of every goal group) fixes the threshold only.
  select     is used only for reporting how the frozen detector behaves; it fits nothing here.
  confirm    is never read.

The selected layers come from `semantic_strategy/selection.json` (stage 3), which already used only train.
Also reported, because they decide whether the fusion earns its keep:
  * the same pipeline with risk layers only and with manipulation layers only,
  * a leave-one-semantic-strategy-out run in which the held-out strategy's goal groups are removed from
    concept building, fusion fitting and (for the threshold) nothing else changes.
Outputs are written to a new directory; an existing one is never overwritten. The frozen artifact uses the
project's own `Detector` format so `python -m dual_concept evaluate` can load it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import LAYERS, Panel, auroc, bootstrap_ci, sha256_file
from dual_concept.backbone import StateCache
from dual_concept.concepts import ConceptBank, concept_scores, fit_concepts
from dual_concept.data import load_dataset, read_jsonl
from dual_concept.evaluation import binary_metrics, calibrate_threshold
from dual_concept.fusion import FusionConfig, build_feature_matrix, fit_sparse_fusion
from dual_concept.pipeline import Detector, json_bytes
from strategy_labels import semantic_strategy

TARGET_FPR = 0.05


def bank_from_train(rows, states, groups_keep, layer_ids):
    """Concept bank from the train pairs whose goal group is in groups_keep."""
    idx = {r["id"]: i for i, r in enumerate(rows)}
    risk, manip, safe, orig = [], [], [], set()
    for g in sorted(groups_keep):
        h, b = f"{g}_h", f"{g}_b"
        if h not in idx or b not in idx:
            continue
        risk.append([idx[b], idx[h]]); safe.append(idx[b])
        for r in rows:
            if r["group_id"] == g and r["kind"] == "jailbreak":
                manip.append([idx[h], idx[r["id"]]]); orig.add(idx[h])
    return fit_concepts(states, risk, manip, sorted(set(safe)), sorted(orig), layer_ids), len(risk), len(manip)


def fit_arm(rows, states, groups_keep, layer_ids, risk_layers, manip_layers, calib_states):
    """Fit one arm on the kept train groups and calibrate it on the calibrate split."""
    keep_rows = [i for i, r in enumerate(rows) if r["group_id"] in groups_keep]
    sub_rows = [rows[i] for i in keep_rows]
    sub_states = states[keep_rows]
    bank, n_risk, n_manip = bank_from_train(sub_rows, sub_states, groups_keep, layer_ids)
    risk, manip = concept_scores(sub_states, bank)
    rl = risk_layers or [manip_layers[0]]
    ml = manip_layers or [risk_layers[0]]
    X, fl, br = build_feature_matrix(risk, manip, rl, ml, layer_ids)
    keep = np.r_[np.ones(len(rl), bool) if risk_layers else np.zeros(len(rl), bool),
                 np.ones(len(ml), bool) if manip_layers else np.zeros(len(ml), bool)]
    X, fl = X[:, keep], fl[keep]
    y = np.array([r["label"] for r in sub_rows], dtype=float)
    fusion = fit_sparse_fusion(X, y, fl, config=FusionConfig())
    if not fusion.diagnostics["converged"]:
        raise RuntimeError("Fusion did not converge; nothing was frozen")

    def score(st):
        r, m = concept_scores(st, bank)
        Xs, _, _ = build_feature_matrix(r, m, rl, ml, layer_ids)
        return fusion.decision_function(Xs[:, keep])

    cal = calibrate_threshold(score(calib_states), TARGET_FPR)
    return {"bank": bank, "fusion": fusion, "score": score, "threshold": cal["threshold"], "calibration": cal,
            "risk_layers": risk_layers, "manip_layers": manip_layers, "n_pairs": (n_risk, n_manip),
            "feature_layers": fl.tolist(), "keep": keep, "rl": rl, "ml": ml}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--selection", required=True, help="semantic_strategy/selection.json from stage 3")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    panel = Panel(a.data, a.cache)           # train + select states
    cache = StateCache(Path(a.cache) / "rows")
    cal_rows = read_jsonl(Path(a.data) / "development" / "calibrate.jsonl")
    cal_states = np.stack([cache.load(r["id"], r["text"], panel.protocol,
                                      expected_shape=(len(LAYERS), int(panel.protocol["hidden_size"])))
                           for r in cal_rows]).astype(np.float32)
    sel = json.loads(Path(a.selection).read_text())
    risk_layers, manip_layers = sel["risk"]["layers"], sel["manip"]["layers"]
    union = sorted(set(risk_layers) | set(manip_layers))
    layer_ids = list(LAYERS)
    train_idx = np.flatnonzero(panel.split == "train")
    rows = [panel.rows[i] for i in train_idx]
    states = panel.states[train_idx]
    groups = {r["group_id"] for r in rows if r["kind"] == "harmful"}

    # ---------- final detector (all train) ----------
    final = fit_arm(rows, states, groups, layer_ids, risk_layers, manip_layers, cal_states)
    pos = [layer_ids.index(l) for l in union]
    sub = final["bank"]
    deploy = ConceptBank(layer_ids=np.asarray(union), risk_direction=sub.risk_direction[pos],
                         manipulation_direction=sub.manipulation_direction[pos],
                         safe_anchor=sub.safe_anchor[pos], harmful_anchor=sub.harmful_anchor[pos])
    model = Detector(deploy, risk_layers, manip_layers, final["fusion"], final["threshold"])
    model.metadata = {
        "method": "dual_concept_complementary_cross_layer", "stage": "stage4_train_detector",
        "layer_selection_source": str(Path(a.selection).name),
        "layer_selection_sha256": sha256_file(Path(a.selection)),
        "selection_note": "Layers chosen in stage 3 by held-out semantic-strategy folds on train; see REPORT.md caveats.",
        "fit_roles": {"concepts_and_fusion": "train", "threshold": "calibrate"},
        "n_train_rows": len(rows), "n_train_goal_groups": len(groups), "n_calibrate": len(cal_rows),
        "fusion": final["fusion"].diagnostics, "calibration": final["calibration"],
        "extraction_protocol": panel.protocol,
        "data_manifest_sha256": sha256_file(Path(a.data) / "manifest.json"),
        "label_status": "two_agent_reviewed_adjudicated_not_human_verified", "confirmation_claim": False,
        "confirm_split_used": False}
    model.save(out / "detector")
    # reload through the project's own loader and confirm identical scores on the calibrate split
    reloaded = Detector.load(out / "detector")
    cal_deploy = cal_states[:, pos]
    same = np.allclose(reloaded.decision_function(cal_deploy), final["score"](cal_states), atol=1e-9)
    if not same:
        raise RuntimeError("Reloaded detector disagrees with the in-memory detector")

    # ---------- report on select (nothing fitted here) ----------
    sel_idx = np.flatnonzero(panel.split == "select")
    sel_states = panel.states[sel_idx]
    sel_rows = [panel.rows[i] for i in sel_idx]
    thr = final["threshold"]
    report = {"threshold": thr, "calibration": final["calibration"], "select": {}}
    for arm_name, (rl, ml) in {"fusion_both": (risk_layers, manip_layers), "risk_only": (risk_layers, []),
                               "manip_only": ([], manip_layers)}.items():
        arm = final if arm_name == "fusion_both" else fit_arm(rows, states, groups, layer_ids, rl, ml, cal_states)
        s = arm["score"](sel_states)
        t = arm["threshold"]
        labels = np.array([r["label"] for r in sel_rows])
        kinds = np.array([r["kind"] for r in sel_rows])
        strat = np.array([semantic_strategy(r) or "" for r in sel_rows])
        m = {"overall": binary_metrics(labels, s, t),
             "by_kind": {k: binary_metrics(labels[kinds == k], s[kinds == k], t) for k in ("safe", "harmful", "jailbreak")},
             "by_strategy": {}}
        for name in sorted(set(strat) - {""}):
            mask = strat == name
            groups_n = len({sel_rows[i]["group_id"] for i in np.flatnonzero(mask)})
            tp = int((s[mask] >= t).sum())
            m["by_strategy"][name] = {"n": int(mask.sum()), "n_goal_groups": groups_n, "tpr": tp / int(mask.sum()),
                                      "low_sample": groups_n < 15}
        m["select_ordinary_B_fpr"] = float((s[kinds == "safe"] >= t).mean())
        m["threshold"] = t
        m["layers"] = {"risk": rl, "manip": ml}
        report["select"][arm_name] = m

    # ---------- leave-one-strategy-out (train only; threshold from calibrate) ----------
    sem_train = np.array([semantic_strategy(r) or "" for r in rows])
    loso = {}
    for held in sorted(set(sem_train) - {""}):
        held_groups = {rows[i]["group_id"] for i in np.flatnonzero(sem_train == held)}
        keep_groups = groups - held_groups
        res = {}
        for arm_name, (rl, ml) in {"fusion_both": (risk_layers, manip_layers), "risk_only": (risk_layers, []),
                                   "manip_only": ([], manip_layers)}.items():
            arm = fit_arm(rows, states, keep_groups, layer_ids, rl, ml, cal_states)
            # held-out strategy J and, as in-distribution benign controls, B of the same held-out groups
            ji = [i for i in np.flatnonzero(sem_train == held)]
            bi = [i for i, r in enumerate(rows) if r["group_id"] in held_groups and r["kind"] == "safe"]
            sj, sb = arm["score"](states[ji]), arm["score"](states[bi])
            t = arm["threshold"]
            lab = np.r_[np.ones(len(sj)), np.zeros(len(sb))]
            res[arm_name] = {"n_J": len(ji), "n_goal_groups": len(held_groups), "tpr": float((sj >= t).mean()),
                             "ordinary_B_fpr": float((sb >= t).mean()), "auroc_J_vs_B": auroc(lab, np.r_[sj, sb])}
        loso[held] = res
    report["leave_one_strategy_out"] = loso
    report["leave_one_strategy_out_note"] = (
        "Concepts and fusion refit without the held-out strategy's goal groups; the layer set was chosen with "
        "all strategies visible, so this is not independent of layer selection. Controls are the held-out groups' "
        "own short B requests and the threshold comes from calibrate (short ordinary requests), so wrapper length "
        "can inflate TPR and AUROC here; see the padded-control analysis in stage 3.")
    (out / "report.json").write_bytes(json_bytes(report))
    summary = {"risk_layers": risk_layers, "manip_layers": manip_layers, "union": union,
               "coef": dict(zip([f"{b}{l}" for b, l in zip(np.array(["risk"] * len(risk_layers) + ["manip"] * len(manip_layers)),
                                                          final["feature_layers"])], np.round(final["fusion"].coef, 4).tolist())),
               "threshold": thr, "calibration": final["calibration"], "reload_matches": bool(same)}
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
