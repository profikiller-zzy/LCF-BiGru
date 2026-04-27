"""
Generate ROC curves for LCF across all attack-task-model combinations.

Reads telemetry JSONL files to extract per-example max z-scores (steps 0-1),
then sweeps thresholds to compute TPR/FPR. Outputs ROC data as JSON and
optionally generates PDF plots.

Usage: python scripts/roc_curves.py [--plot]
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

EVAL_BASE = Path(__file__).resolve().parent.parent / "attack" / "DPA" / "eval_result"
MODELS = [
    ("Meta-Llama-3-8B-Instruct", "llama3", "Llama-3-8B"),
    ("Qwen2.5-7B-Instruct-qwentpl", "qwen", "Qwen2.5-7B"),
    ("Gemma-2-9b-it", "gemma", "Gemma-2-9B-IT"),
]
TASKS = ["negsentiment", "refusal"]
ATTACKS = ["badnet", "sleeper", "vpi", "ctba", "mtba", "stylebkd", "synbkd"]
LCF_TAG = "lcf_v3"


def extract_max_z_per_example(telemetry_path, max_step=1):
    """Extract per-example max z-score from steps 0..max_step."""
    example_zs = defaultdict(list)
    with open(telemetry_path) as f:
        for line in f:
            d = json.loads(line)
            if d["step_index"] <= max_step:
                example_zs[d["example_id"]].append(d["z_score"])
    # Max z across monitored steps per example
    return {eid: max(zs) for eid, zs in example_zs.items()}


def compute_roc(clean_scores, triggered_scores, n_thresholds=500):
    """Compute ROC curve from clean and triggered max z-scores."""
    all_scores = np.concatenate([clean_scores, triggered_scores])
    # Use actual score values as thresholds (plus endpoints) for accurate ROC
    unique_scores = np.unique(all_scores)
    # Add a point below min and above max to ensure we reach (0,0) and (1,1)
    thresholds = np.concatenate([
        [unique_scores.max() + 1],  # no positives → (0, 0)
        np.linspace(unique_scores.max(), unique_scores.min() - 0.01, n_thresholds),
        [unique_scores.min() - 1],  # all positive → (1, 1)
    ])

    fprs, tprs = [], []
    for tau in thresholds:
        fpr = np.mean(clean_scores >= tau)  # clean flagged = false positive
        tpr = np.mean(triggered_scores >= tau)  # triggered flagged = true positive
        fprs.append(float(fpr))
        tprs.append(float(tpr))

    # AUC (trapezoidal, curve goes from (0,0) to (1,1))
    fprs_arr = np.array(fprs)
    tprs_arr = np.array(tprs)
    # Sort by FPR for proper integration
    sorted_idx = np.argsort(fprs_arr)
    auc = float(np.trapz(tprs_arr[sorted_idx], fprs_arr[sorted_idx]))

    return {
        "thresholds": [float(t) for t in thresholds],
        "fpr": fprs,
        "tpr": tprs,
        "auc": abs(auc),
        "n_clean": len(clean_scores),
        "n_triggered": len(triggered_scores),
    }


def main():
    do_plot = "--plot" in sys.argv
    output_dir = Path(__file__).resolve().parent.parent / "roc_results"
    output_dir.mkdir(exist_ok=True)

    all_results = {}

    for model, model_short, model_label in MODELS:
        for task in TASKS:
            for attack in ATTACKS:
                lcf_dir = EVAL_BASE / task / attack / f"eval_{model}" / "kgc" / LCF_TAG
                trig_telem = lcf_dir / "telemetry_triggered.jsonl"
                clean_telem = lcf_dir / "telemetry_clean.jsonl"

                if not trig_telem.exists() or not clean_telem.exists():
                    print(f"SKIP {model_short}/{task}/{attack}: telemetry not found")
                    continue

                clean_z = extract_max_z_per_example(clean_telem)
                trig_z = extract_max_z_per_example(trig_telem)

                if not clean_z or not trig_z:
                    print(f"SKIP {model_short}/{task}/{attack}: empty z-scores")
                    continue

                clean_arr = np.array(list(clean_z.values()))
                trig_arr = np.array(list(trig_z.values()))

                roc = compute_roc(clean_arr, trig_arr)
                key = f"{model_short}_{task}_{attack}"
                all_results[key] = roc

                print(f"{key}: AUC={roc['auc']:.4f} (n_clean={roc['n_clean']}, n_trig={roc['n_triggered']})")

    # Save raw ROC data
    with open(output_dir / "roc_data.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nROC data saved to {output_dir / 'roc_data.json'}")

    if do_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # --- Per-model aggregate plots ---
            for _, model_key, model_label in MODELS:
                fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
                for ti, task in enumerate(TASKS):
                    ax = axes[ti]
                    for attack in ATTACKS:
                        key = f"{model_key}_{task}_{attack}"
                        if key not in all_results:
                            continue
                        roc = all_results[key]
                        ax.plot(roc["fpr"], roc["tpr"],
                                label=f"{attack} (AUC={roc['auc']:.3f})", linewidth=1.2)
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.set_title(f"{model_label} — {task}")
                    ax.legend(fontsize=7, loc="lower right")
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_aspect("equal")
                    ax.grid(True, alpha=0.2)

                fig.tight_layout()
                fig_path = output_dir / f"roc_{model_key}.pdf"
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved {fig_path}")

            # --- Combined summary: per-model mean ROC ---
            fig, ax = plt.subplots(1, 1, figsize=(5, 4.5))
            for model_key, model_label, color in [
                ("llama3", "Llama-3-8B", "#2563eb"),
                ("qwen", "Qwen2.5-7B", "#dc2626"),
                ("gemma", "Gemma-2-9B-IT", "#059669"),
            ]:
                # Collect all ROCs for this model
                model_rocs = [v for k, v in all_results.items() if k.startswith(model_key)]
                if not model_rocs:
                    continue
                # Interpolate all to common FPR grid
                fpr_grid = np.linspace(0, 1, 200)
                tpr_interps = []
                for roc in model_rocs:
                    fpr_arr = np.array(roc["fpr"])
                    tpr_arr = np.array(roc["tpr"])
                    # Sort by FPR for interpolation
                    idx = np.argsort(fpr_arr)
                    tpr_interp = np.interp(fpr_grid, fpr_arr[idx], tpr_arr[idx])
                    tpr_interps.append(tpr_interp)
                mean_tpr = np.mean(tpr_interps, axis=0)
                std_tpr = np.std(tpr_interps, axis=0)
                mean_auc = np.mean([r["auc"] for r in model_rocs])

                ax.plot(fpr_grid, mean_tpr, color=color, linewidth=1.5,
                        label=f"{model_label} (mean AUC={mean_auc:.3f})")
                ax.fill_between(fpr_grid, mean_tpr - std_tpr, mean_tpr + std_tpr,
                                alpha=0.15, color=color)

            ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("LCF Detection — Mean ROC by Architecture")
            ax.legend(fontsize=9)
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 1.02)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.2)
            fig.tight_layout()
            fig_path = output_dir / "roc_summary.pdf"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fig_path}")

            # --- Combined paper figure: all models by task ---
            fig, axes = plt.subplots(len(TASKS), len(MODELS), figsize=(10.8, 6.9))
            for ti, task in enumerate(TASKS):
                for mi, (_, model_key, model_label) in enumerate(MODELS):
                    ax = axes[ti, mi]
                    for attack in ATTACKS:
                        key = f"{model_key}_{task}_{attack}"
                        if key not in all_results:
                            continue
                        roc = all_results[key]
                        ax.plot(
                            roc["fpr"],
                            roc["tpr"],
                            label=f"{attack} ({roc['auc']:.3f})",
                            linewidth=1.1,
                        )
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.02, 1.02)
                    ax.grid(True, alpha=0.2)
                    if ti == 0:
                        ax.set_title(model_label, fontsize=10)
                    if mi == 0:
                        ax.set_ylabel(f"{task.capitalize()}\nTrue Positive Rate", fontsize=9)
                    else:
                        ax.set_ylabel("")
                    if ti == len(TASKS) - 1:
                        ax.set_xlabel("False Positive Rate", fontsize=9)
                    else:
                        ax.set_xlabel("")
                    ax.tick_params(labelsize=8)
                    ax.legend(fontsize=5.5, loc="lower right", framealpha=0.9)

            fig.tight_layout(h_pad=1.0, w_pad=0.8)
            fig_path = output_dir / "roc_all_models.pdf"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {fig_path}")

        except ImportError:
            print("matplotlib not available, skipping plots")

    # --- Print AUC summary table ---
    print(f"\n{'='*60}")
    print("AUC SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<10} {'Task':<15} {'Attack':<10} {'AUC':>6}")
    print("-" * 45)
    aucs_by_model = defaultdict(list)
    for key, roc in sorted(all_results.items()):
        parts = key.split("_", 2)
        model_key, task, attack = parts[0], parts[1], parts[2]
        print(f"{model_key:<10} {task:<15} {attack:<10} {roc['auc']:>6.4f}")
        aucs_by_model[model_key].append(roc["auc"])
    print("-" * 45)
    for mk, aucs in aucs_by_model.items():
        print(f"{mk:<10} {'MEAN':<15} {'':10} {np.mean(aucs):>6.4f}")


if __name__ == "__main__":
    main()
