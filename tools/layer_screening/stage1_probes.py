"""Stage 1: per-layer MLP probes (B=0 vs H=1), candidate pool, J transfer and confound checks.

Probes only read frozen cached last-token states. J rows are never used to fit a probe;
they are scored by models that did not see their goal group. select rows are scored by
fold-averaged probes and never influence admission.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import LAYERS, Panel, auroc, grouped_folds, sha256_file

CFG = {
    "hidden": 100, "max_epochs": 500, "patience": 20, "lr": 1e-3, "weight_decay": 1e-2, "batch_size": 64,
    "early_stop_group_fraction": 0.15, "n_folds": 5, "seeds": [0, 1, 2, 3, 4],
    "tau_probe": 0.70, "worst_run_floor": 0.55, "min_positive_seed_fraction": 0.8,
}


def make_probe(width: int, hidden: int) -> nn.Module:
    return nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 2))


def fit_probe(x_fit, y_fit, x_stop, y_stop, seed: int, device) -> nn.Module:
    torch.manual_seed(seed)
    model = make_probe(x_fit.shape[1], CFG["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, wait = float("inf"), None, 0
    n = len(x_fit)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(CFG["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        for start in range(0, n, CFG["batch_size"]):
            idx = perm[start:start + CFG["batch_size"]]
            opt.zero_grad(set_to_none=True)
            loss_fn(model(x_fit[idx]), y_fit[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            stop_loss = float(loss_fn(model(x_stop), y_stop))
        if stop_loss < best - 1e-6:
            best, wait = stop_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["patience"]:
                break
    model.load_state_dict(best_state)
    return model.eval()


def prob_positive(model, x) -> np.ndarray:
    with torch.no_grad():
        return torch.softmax(model(x), dim=1)[:, 1].cpu().numpy()


def standardize(fit: torch.Tensor):
    mean = fit.mean(0, keepdim=True)
    std = fit.std(0, unbiased=False, keepdim=True)
    std = torch.where(std > 0, std, torch.ones_like(std))
    return mean, std


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=list(LAYERS))
    ap.add_argument("--seeds", type=int, nargs="*", default=CFG["seeds"])
    args = ap.parse_args()
    out = Path(args.out)
    (out / "stage1").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    panel = Panel(args.data, args.cache)
    train_m = panel.mask("train")
    probe_rows = np.flatnonzero(train_m & np.isin(panel.kind, ["safe", "harmful"]))
    y_all = (panel.kind[probe_rows] == "harmful").astype(np.int64)
    j_train = np.flatnonzero(train_m & (panel.kind == "jailbreak"))
    sel_b = np.flatnonzero(panel.mask("select", "safe"))
    sel_j = np.flatnonzero(panel.mask("select", "jailbreak"))
    sel_h = np.flatnonzero(panel.mask("select", "harmful"))
    groups = panel.group
    config = {**CFG, "seeds": args.seeds, "layers": args.layers,
              "n_probe_rows": int(len(probe_rows)), "n_train_goal_groups": len(set(groups[probe_rows])),
              "data_manifest_sha256": sha256_file(Path(args.data) / "manifest.json"),
              "protocol_sha256": panel.protocol.get("model_config_sha256"),
              "device": str(device), "torch": torch.__version__}
    (out / "stage1" / "config.json").write_text(json.dumps(config, indent=1))

    metrics_rows = []
    oof_probe = {}      # (layer, seed) -> prob for every train row (probe rows + J train), out-of-fold
    sel_scores = {}     # (layer, seed) -> fold-averaged prob on select rows
    select_rows = np.concatenate([sel_b, sel_h, sel_j])
    t0 = time.time()
    for layer in args.layers:
        li = layer - 1
        x_layer = torch.from_numpy(panel.states[:, li, :]).to(device)
        for seed in args.seeds:
            fold_of = grouped_folds(groups[probe_rows], CFG["n_folds"], seed)
            oof = np.full(len(panel.rows), np.nan)
            sel_acc = np.zeros(len(select_rows))
            for fold in range(CFG["n_folds"]):
                test_groups = {g for g, f in fold_of.items() if f == fold}
                train_groups = sorted(set(fold_of) - test_groups)
                stop_n = max(1, int(round(CFG["early_stop_group_fraction"] * len(train_groups))))
                rng = np.random.default_rng(1000 * seed + fold)
                stop_groups = set(rng.choice(train_groups, size=stop_n, replace=False))
                fit_groups = set(train_groups) - stop_groups
                pr_groups = groups[probe_rows]
                fit_idx = probe_rows[np.isin(pr_groups, list(fit_groups))]
                stop_idx = probe_rows[np.isin(pr_groups, list(stop_groups))]
                test_idx = probe_rows[np.isin(pr_groups, list(test_groups))]
                j_test_idx = j_train[np.isin(groups[j_train], list(test_groups))]
                yfit = torch.from_numpy((panel.kind[fit_idx] == "harmful").astype(np.int64)).to(device)
                ystop = torch.from_numpy((panel.kind[stop_idx] == "harmful").astype(np.int64)).to(device)
                mean, std = standardize(x_layer[fit_idx])
                z = lambda idx: (x_layer[idx] - mean) / std
                model = fit_probe(z(fit_idx), yfit, z(stop_idx), ystop, seed * 100 + fold, device)
                oof[test_idx] = prob_positive(model, z(test_idx))
                if len(j_test_idx):
                    oof[j_test_idx] = prob_positive(model, z(j_test_idx))
                sel_acc += prob_positive(model, z(select_rows)) / CFG["n_folds"]
                yt = (panel.kind[test_idx] == "harmful").astype(int)
                metrics_rows.append({"layer": layer, "seed": seed, "fold": fold,
                                     "auroc": auroc(yt, oof[test_idx]), "n_test": len(test_idx),
                                     "n_test_groups": len(test_groups)})
            oof_probe[(layer, seed)] = oof
            sel_scores[(layer, seed)] = sel_acc
        print(f"layer {layer:2d} done  elapsed {time.time() - t0:6.0f}s", flush=True)
        del x_layer
        torch.cuda.empty_cache()

    with (out / "stage1" / "probe_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0]))
        w.writeheader(); w.writerows(metrics_rows)

    # ---- per-layer summary and admission ----
    summary = {}
    for layer in args.layers:
        runs = [r for r in metrics_rows if r["layer"] == layer]
        a = np.array([r["auroc"] for r in runs])
        seed_means = np.array([np.nanmean([r["auroc"] for r in runs if r["seed"] == s]) for s in args.seeds])
        pooled = []
        for s in args.seeds:      # pooled out-of-fold AUROC per seed (all B vs all H)
            o = oof_probe[(layer, s)][probe_rows]
            pooled.append(auroc(y_all, o))
        summary[layer] = {
            "median_auroc": float(np.nanmedian(a)), "mean_auroc": float(np.nanmean(a)), "std_auroc": float(np.nanstd(a)),
            "worst_run_auroc": float(np.nanmin(a)), "seed_mean_auroc": seed_means.tolist(),
            "pooled_oof_auroc_by_seed": pooled,
            "seeds_with_mean_above_half": int((seed_means > 0.5).sum()),
        }
        summary[layer]["admitted"] = bool(
            summary[layer]["median_auroc"] >= CFG["tau_probe"]
            and summary[layer]["worst_run_auroc"] >= CFG["worst_run_floor"]
            and summary[layer]["seeds_with_mean_above_half"] >= CFG["min_positive_seed_fraction"] * len(args.seeds))
    best = max(v["median_auroc"] for v in summary.values())
    for v in summary.values():
        v["within_0.02_of_best"] = bool(v["median_auroc"] >= best - 0.02)
    admitted = [l for l, v in summary.items() if v["admitted"]]

    # ---- J transfer: oof on train J, fold-averaged on select J ----
    b_train = np.flatnonzero(train_m & (panel.kind == "safe"))
    rows = []
    strategies_fam = sorted(set(panel.family[j_train]))
    strategies_lin = sorted(set(panel.strategy[j_train]))
    for layer in args.layers:
        for level, names, arr in (("family", strategies_fam, panel.family), ("strategy", strategies_lin, panel.strategy)):
            for name in names:
                ja = j_train[arr[j_train] == name]
                vals, tprs = [], []
                for s in args.seeds:
                    o = oof_probe[(layer, s)]
                    lab = np.r_[np.zeros(len(b_train)), np.ones(len(ja))]
                    sc = np.r_[o[b_train], o[ja]]
                    vals.append(auroc(lab, sc))
                    thr = np.quantile(o[b_train], 0.95)
                    tprs.append(float((o[ja] > thr).mean()))
                rows.append({"split": "train_oof", "level": level, "strategy": name, "layer": layer,
                             "auroc_B_vs_J": float(np.mean(vals)), "tpr_at_5pct_fpr": float(np.mean(tprs)),
                             "n_J": len(ja), "n_goal_groups": len(set(groups[ja])), "n_B": len(b_train)})
    pos = {r: i for i, r in enumerate(select_rows)}
    for layer in args.layers:
        for level, names, arr in (("family", sorted(set(panel.family[sel_j])), panel.family),
                                  ("strategy", sorted(set(panel.strategy[sel_j])), panel.strategy)):
            for name in names:
                ja = sel_j[arr[sel_j] == name]
                vals, tprs = [], []
                for s in args.seeds:
                    sc_all = sel_scores[(layer, s)]
                    b = sc_all[[pos[i] for i in sel_b]]; j = sc_all[[pos[i] for i in ja]]
                    vals.append(auroc(np.r_[np.zeros(len(b)), np.ones(len(j))], np.r_[b, j]))
                    tprs.append(float((j > np.quantile(b, 0.95)).mean()))
                rows.append({"split": "select", "level": level, "strategy": name, "layer": layer,
                             "auroc_B_vs_J": float(np.mean(vals)), "tpr_at_5pct_fpr": float(np.mean(tprs)),
                             "n_J": len(ja), "n_goal_groups": len(set(groups[ja])), "n_B": len(sel_b)})
    with (out / "stage1" / "j_transfer.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    # ---- confound checks: length and bag-of-words baselines on the same folds ----
    confound = lexical_baselines(panel, probe_rows, y_all, args.seeds)
    probe_cache = {"layers": args.layers, "seeds": args.seeds}
    np.savez_compressed(out / "stage1" / "oof_scores.npz",
                        **{f"L{l}_S{s}": oof_probe[(l, s)].astype(np.float32) for (l, s) in oof_probe},
                        ids=np.array([r["id"] for r in panel.rows]))
    result = {"config": config, "layers": summary, "C_probe": admitted, "n_admitted": len(admitted),
              "best_median_auroc": best, "confound": confound, "status":
              "STOP_C_probe_below_2" if len(admitted) < 2 else "ok"}
    (out / "stage1" / "probe_summary.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"C_probe": admitted, "best": best, "confound": confound}, indent=1))


def lexical_baselines(panel: Panel, probe_rows, y_all, seeds) -> dict:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    texts = [panel.rows[i]["text"] for i in probe_rows]
    char_len = np.array([len(t) for t in texts], dtype=float)
    tok_len = np.array([panel.rows[i]["full_input_token_count"] for i in probe_rows], dtype=float)
    out = {"length_auroc": auroc(y_all, char_len), "token_length_auroc": auroc(y_all, tok_len)}
    aucs = []
    for s in seeds:
        fold_of = grouped_folds(panel.group[probe_rows], CFG["n_folds"], s)
        fold = np.array([fold_of[g] for g in panel.group[probe_rows]])
        for f in range(CFG["n_folds"]):
            tr, te = fold != f, fold == f
            vec = CountVectorizer(lowercase=True, min_df=2)
            xtr = vec.fit_transform([t for t, m in zip(texts, tr) if m])
            xte = vec.transform([t for t, m in zip(texts, te) if m])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(xtr, y_all[tr])
            aucs.append(auroc(y_all[te], clf.decision_function(xte)))
    out["bag_of_words_auroc_median"] = float(np.nanmedian(aucs))
    out["bag_of_words_auroc_mean"] = float(np.nanmean(aucs))
    out["bag_of_words_auroc_worst"] = float(np.nanmin(aucs))
    official = [i for i in probe_rows if panel.rows[i].get("source", {}).get("origin") == "jbb_benign_behavior"]
    out["n_official_jbb_benign_in_probe_rows"] = len(official)
    return out


if __name__ == "__main__":
    main()
