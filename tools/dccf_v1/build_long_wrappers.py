"""Build long composite wrappers: one template stacked inside another, with a paired harmful and benign goal.

Why. In the five-backbone run, the held-out rule_suffix attacks are about 512 tokens while the longest wrapped
benign request the fit ever sees is 436 tokens, so that fold tests length extrapolation. These composites supply
longer, still-real wrapper text without padding. Each composite is built twice from the same wrapper stack:
  * harmful goal H of a goal group  -> label 1 (a long jailbreak)
  * benign request B of the same group -> label 0 (the same wrapper around a benign goal, like W(B))
so a model cannot learn "long wrapper" as a proxy for either class.

Construction. A wrapper is the text before and after the goal in a real template attack. A composite is
  outer_prefix + inner_prefix + GOAL + inner_suffix + outer_suffix
using two different template lineages. Only the 28 template lineages with an extractable wrapper are used; rule_suffix,
PAIR/TAP, persuasion and the GCG/DSN families are not stacked (their wrappers are goal-dependent or free-form).

Leakage rules, enforced here and recorded per row:
  * a composite is tagged with the semantic strategy of EVERY lineage it contains (`strategies`);
  * when a strategy m is held out, every composite whose `strategies` contains m must be dropped (done in stage 9);
  * composites are assigned to goal groups of the TRAIN split only, so held-out groups never contribute composites;
  * the held-out group's own goals are never used (a composite carries its group's goal and is dropped with that group).
Token length is targeted to 250..600 tokens by stratified sampling over length buckets.
Output is a separate panel `data/dccf_v1_long_wrappers`, never mixed into train/select/calibrate/confirm roles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_jsonl, sha256, write_jsonl  # noqa: E402

_spec = importlib.util.spec_from_file_location("strategy_labels", Path(__file__).resolve().parents[1] / "layer_screening" / "strategy_labels.py")
_mod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_mod)
semantic_strategy = _mod.semantic_strategy

BUCKETS = [(250, 350), (350, 450), (450, 550), (550, 620)]
PER_BUCKET = 150           # composites per bucket (each yields a harmful and a benign row)
MAX_PER_GROUP = 3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--build", default="artifacts/dccf_v1_build")
    ap.add_argument("--out", default="data/dccf_v1_long_wrappers")
    ap.add_argument("--seed", type=int, default=20261106)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer
    from dual_concept.backbone import render_chat_input
    tok = AutoTokenizer.from_pretrained(str(Path(a.build) / "qwen_tokenizer"))
    ntok = lambda t: len(tok(render_chat_input(tok, t), add_special_tokens=False)["input_ids"])

    rows = {}
    for split in ("train", "select"):
        for r in read_jsonl(Path(a.data) / "development" / f"{split}.jsonl"):
            rows[r["id"]] = r
    # one wrapper per lineage (the template text is identical across goals)
    wrappers = {}
    for r in rows.values():
        if r["kind"] == "jailbreak" and r["attack_family"] == "template" and r["template_lineage_id"] not in wrappers:
            h = rows[r["original_id"]]["text"]; t = r["text"]; i = t.lower().find(h.lower())
            if i >= 0 and t.lower().count(h.lower()) == 1:
                wrappers[r["template_lineage_id"]] = {"pre": t[:i], "suf": t[i + len(h):], "strategy": semantic_strategy(r)}
    lineages = sorted(wrappers)
    train_groups = sorted(g for g in {r["group_id"] for r in rows.values() if r["split"] == "train" and r["kind"] == "harmful"})
    goal = {g: (rows[f"{g}_h"]["text"], rows[f"{g}_b"]["text"]) for g in train_groups if f"{g}_h" in rows and f"{g}_b" in rows}
    rng = random.Random(a.seed)
    pairs = [(o, i) for o in lineages for i in lineages if o != i]
    # token cost of a wrapper stack with a placeholder goal
    cost = {}
    base = ntok("x")
    for o, i in pairs:
        cost[(o, i)] = ntok(wrappers[o]["pre"] + wrappers[i]["pre"] + "x" + wrappers[i]["suf"] + wrappers[o]["suf"]) - base

    chosen, used_group = [], Counter()
    for lo, hi in BUCKETS:
        # goals are ~43 tokens (template adds a little); aim the stack so that the total lands in the bucket
        cand = [(p, c) for p, c in cost.items() if lo - 70 <= c + 45 <= hi - 20]
        rng.shuffle(cand)
        n = 0
        groups = train_groups[:]
        rng.shuffle(groups)
        gi = 0
        for (o, i), _ in cand * 6:
            if n >= PER_BUCKET:
                break
            g = groups[gi % len(groups)]; gi += 1
            if used_group[g] >= MAX_PER_GROUP or g not in goal:
                continue
            h, b = goal[g]
            hb = []
            ok = True
            for kind, goal_text in (("harmful", h), ("safe", b)):
                text = wrappers[o]["pre"] + wrappers[i]["pre"] + goal_text + wrappers[i]["suf"] + wrappers[o]["suf"]
                hb.append((kind, text, ntok(text)))
            if not all(lo <= n_ < hi for _, _, n_ in hb):
                continue
            used_group[g] += 1
            n += 1
            stamp = hashlib.sha256(f"{o}|{i}|{g}".encode()).hexdigest()[:10]
            strategies = sorted({wrappers[o]["strategy"], wrappers[i]["strategy"]})
            for kind, text, n_tok in hb:
                chosen.append({
                    "id": f"lw_{stamp}_{'h' if kind == 'harmful' else 'b'}", "text": text, "kind": kind,
                    "label": 1 if kind == "harmful" else 0, "core_group": "long_wrapper", "group_id": g, "split": "train",
                    "attack_family": None, "original_id": None, "exposure": "dccf_v1_long_wrapper",
                    "label_status": "constructed_control_not_reviewed", "outer_lineage": o, "inner_lineage": i,
                    "strategies": strategies, "lineages": [o, i], "full_input_token_count": n_tok, "length_bucket": f"{lo}-{hi}"})
    chosen.sort(key=lambda r: r["id"])
    write_jsonl(out / "development" / "long_wrappers.jsonl", chosen)
    toks = [r["full_input_token_count"] for r in chosen]
    manifest = {
        "scope": "constructed_long_wrapper_panel_not_part_of_any_train_select_calibrate_confirm_role",
        "construction": "outer template prefix + inner template prefix + goal + inner suffix + outer suffix; each composite built with H (label 1) and B (label 0)",
        "n_rows": len(chosen), "n_composites": len(chosen) // 2,
        "by_label": dict(Counter(r["label"] for r in chosen)),
        "by_bucket": dict(Counter(r["length_bucket"] for r in chosen if r["kind"] == "harmful")),
        "token_min": min(toks), "token_median": sorted(toks)[len(toks) // 2], "token_max": max(toks),
        "strategy_pairs": {"+".join(k): v for k, v in Counter(tuple(r["strategies"]) for r in chosen if r["kind"] == "harmful").items()},
        "lineages_used": len({l for r in chosen for l in r["lineages"]}), "goal_groups_used": len({r["group_id"] for r in chosen}),
        "leakage_rule": "when strategy m is held out, drop every composite whose `strategies` contains m; composites use train goal groups only",
        "limitations": ["Stacked templates are artificial: no real attack nests two role-play prompts.",
                        "rule_suffix, PAIR/TAP, persuasion and GCG/DSN wrappers are not stacked, so the rule_suffix fold remains a structurally unseen wrapper.",
                        "Composites are constructed and not reviewed; harmful rows still contain the harmful goal and are labelled 1 by construction."],
        "seed": a.seed,
        "output_sha256": {str(p.relative_to(out)): sha256(p.read_bytes()) for p in sorted(out.rglob("*.jsonl"))},
        "source_data_manifest_sha256": sha256((Path(a.data) / "manifest.json").read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("output_sha256", "limitations", "strategy_pairs")}, indent=1))
    print("strategy pairs:", manifest["strategy_pairs"])


if __name__ == "__main__":
    main()
