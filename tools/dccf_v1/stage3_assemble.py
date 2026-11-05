"""Stage 3: assemble the reviewed DCCF v1 dataset in the dual_concept loader layout.

Inputs are the stage-1 goal pools, the authored benign counterparts and the
reviewer/adjudicator decision files. Nothing is auto-approved here: a goal is kept only
when its H-B pair and at least one J-H pair were accepted by the two-reviewer +
adjudicator process. Run with a Python that has `transformers` and `jinja2`
(token-length audit uses the project's own chat-template renderer).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import canonical, jaccard, read_jsonl, sha256, stable_key, word_set, write_jsonl  # noqa: E402
from stage1_select import assign_roles  # noqa: E402

CAL_TARGET = 200
MAX_TOKENS = 2048
SPLIT_OF_ROLE = {"train": "train", "select": "select", "confirm": "confirm"}


def load_decisions(build: Path) -> dict[str, str]:
    decisions = {k: v for k, v in json.loads((build / "round1_final.json").read_text()).items()}
    # round 2 (item ids hb2_/jh_ for new goals)
    adjudicated = {x["item_id"]: x["decision"] for x in json.loads((build / "review2/adj.json").read_text())}
    names = sorted(Path(p).name for p in glob.glob(str(build / "review2/*_*.json")) if "disputes" not in p and "adj" not in p)
    for name in names:
        a = {x["item_id"]: x["decision"] for x in json.loads((build / "r1b" / name).read_text())}
        b = {x["item_id"]: x["decision"] for x in json.loads((build / "r2b" / name).read_text())}
        inputs = [x["item_id"] for x in json.loads((build / "review2" / name).read_text())]
        assert set(a) == set(b) == set(inputs), name
        for item in inputs:
            if a[item] == b[item]:
                decisions[item] = a[item]
            else:
                decisions[item] = adjudicated[item]
    return decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="artifacts/dccf_v1_build")
    parser.add_argument("--output", default="data/dccf_v1")
    args = parser.parse_args()
    build, out = Path(args.build), Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    decisions = load_decisions(build)

    round1 = {g["goal_id"]: g for g in read_jsonl(build / "stage1/goals.jsonl")}
    round2 = {g["goal_id"]: g for g in read_jsonl(build / "stage1/goals_round2_new.jsonl")}
    benign1 = {}
    for i in range(6):
        for y in json.loads((build / f"tasks/author_{i}.out.json").read_text()):
            benign1[y["goal_id"]] = y["benign_request"].strip()
    benign2 = {}
    for path in sorted(glob.glob(str(build / "tasks2/author_*.out.json"))):
        for y in json.loads(Path(path).read_text()):
            benign2[y["goal_id"]] = y["benign_request"].strip()

    kept = []
    for gid, g in {**round1, **round2}.items():
        j_prefix = f"jh_{gid}_"
        accepted_j = [k for k in range(len(g["J"])) if decisions.get(j_prefix + str(k)) == "accept"]
        b_text = b_source = None
        if gid in round2:
            if decisions.get("hb2_" + gid) == "accept":
                b_text, b_source = benign2[gid], "authored_round2"
        else:
            if decisions.get("hb_" + gid) == "accept":
                b_text = g["B_candidate"]["text"] if g["B_candidate"] else benign1[gid]
                b_source = "jbb_benign_behavior" if g["B_candidate"] else "authored_round1"
            elif decisions.get("hb2_" + gid) == "accept":
                b_text, b_source = benign2[gid], "authored_round2"
        if b_text and accepted_j:
            kept.append({**g, "B": b_text, "B_source": b_source, "J_accepted": accepted_j})
    print("goals kept", len(kept), "of", len(round1) + len(round2))

    # cross-goal duplicate audit on every text that will be written
    seen, duplicates = {}, []
    for g in kept:
        for label, text in [("H", g["H"]["text"]), ("B", g["B"])] + [("J", g["J"][k]["text"]) for k in g["J_accepted"]]:
            key = canonical(text)
            if key in seen and seen[key] != g["goal_id"]:
                duplicates.append((label, g["goal_id"], seen[key]))
            seen[key] = g["goal_id"]
    if duplicates:
        raise ValueError(f"Cross-goal duplicate texts: {duplicates[:5]}")
    words = [(g["goal_id"], word_set(g["H"]["text"])) for g in kept]
    near = [(a, b) for i, (a, wa) in enumerate(words) for b, wb in words[i + 1:] if jaccard(wa, wb) >= 0.8]
    if near:
        raise ValueError(f"Near-duplicate harmful goals across groups: {near[:5]}")
    # B must not equal any H/J of other goals nor be a copy of its own H
    for g in kept:
        if canonical(g["B"]) == canonical(g["H"]["text"]):
            raise ValueError(f"Benign equals harmful for {g['goal_id']}")

    assign_roles(kept)
    kept.sort(key=lambda g: g["goal_id"])

    # calibration pool: accepted dolly requests, deterministic subset
    cal_all = {"cal_" + r["source_row_id"]: r for r in read_jsonl(build / "stage1/calibration_candidates.jsonl")}
    accepted_cal = sorted((i for i in cal_all if decisions.get(i) == "accept"), key=lambda i: stable_key("calsel", cal_all[i]["text"]))
    cal_ids = accepted_cal[:CAL_TARGET]
    taken_texts = {canonical(t) for t in seen}
    cal_ids = [i for i in cal_ids if canonical(cal_all[i]["text"]) not in taken_texts]

    # tokenizer audit with the project's own renderer
    from transformers import AutoTokenizer
    from dual_concept.backbone import render_chat_input
    tokenizer = AutoTokenizer.from_pretrained(str(build / "qwen_tokenizer"))
    def tokens(text: str) -> int:
        return len(tokenizer(render_chat_input(tokenizer, text), add_special_tokens=False)["input_ids"])

    records: dict[str, list[dict]] = defaultdict(list)
    risk_all, manip_all, dropped_long = [], [], []
    common = {"language": "en", "turn_count": 1, "label_status": "two_agent_reviewed_adjudicated_not_human_verified",
              "attack_success": "not_evaluated"}
    for g in kept:
        split = SPLIT_OF_ROLE[g["role"]]
        exposure = "independent_confirmation" if split == "confirm" else "dccf_v1_construction"
        gid = g["goal_id"]
        h_id, b_id = f"{gid}_h", f"{gid}_b"
        h_src = {k: v for k, v in g["H"].items() if k != "text"}
        base = {"group_id": gid, "split": split, "exposure": exposure, "goal_category": g["category"],
                "goal_source": g["source"], **common}
        h_rec = {**base, "id": h_id, "text": g["H"]["text"], "kind": "harmful", "label": 1, "core_group": "harmful",
                 "attack_family": None, "original_id": None, "source": h_src}
        b_rec = {**base, "id": b_id, "text": g["B"], "kind": "safe", "label": 0, "core_group": "benign",
                 "attack_family": None, "original_id": None, "source": {"origin": g["B_source"]}}
        js = []
        for k in g["J_accepted"]:
            j = g["J"][k]
            js.append({**base, "id": f"{gid}_j{k}", "text": j["text"], "kind": "jailbreak", "label": 1, "core_group": "jailbreak",
                       "attack_family": j["family"], "template_lineage_id": j["lineage"], "original_id": h_id,
                       "source": {"sources": j["sources"]}})
        group_records = [h_rec, b_rec] + js
        for r in group_records:
            r["full_input_token_count"] = tokens(r["text"])
        too_long = [r["id"] for r in group_records if r["full_input_token_count"] > MAX_TOKENS]
        if too_long:
            # drop only the over-long jailbreaks; an over-long H or B removes the whole goal
            if any(i in too_long for i in (h_id, b_id)) or all(r["id"] in too_long for r in js):
                dropped_long.append(gid)
                continue
            group_records = [r for r in group_records if r["id"] not in too_long]
            js = [r for r in js if r["id"] not in too_long]
            dropped_long.extend(too_long)
        records[split].extend(group_records)
        risk_all.append({"pair_id": f"risk_{gid}", "positive_id": h_id, "negative_id": b_id, "split": split,
                         "goal_cluster_id": gid, "review": "accepted_two_reviewers_or_adjudicated"})
        for r in js:
            manip_all.append({"pair_id": f"manip_{r['id']}", "positive_id": r["id"], "negative_id": h_id, "split": split,
                              "goal_cluster_id": gid, "attack_family": r["attack_family"],
                              "template_lineage_id": r["template_lineage_id"],
                              "review": "accepted_two_reviewers_or_adjudicated"})
    for cid in cal_ids:
        text = cal_all[cid]["text"]
        records["calibrate"].append({
            "id": cid, "text": text, "kind": "safe", "label": 0, "core_group": "benign", "group_id": cid, "split": "calibrate",
            "attack_family": None, "original_id": None, "exposure": "dccf_v1_construction", "goal_category": "ordinary_instruction",
            "goal_source": "dolly", "source": {"dataset": "databricks-dolly-15k", "revision": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
                                                "source_row_id": cal_all[cid]["source_row_id"], "category": cal_all[cid]["dolly_category"],
                                                "license": "CC-BY-SA-3.0"},
            "full_input_token_count": tokens(text), **common})

    # loader layout: risk pairs are training-only
    risk_loader = [p for p in risk_all if p["split"] == "train"]
    for name in ("train", "select", "calibrate", "diagnose", "confirm"):
        rows = sorted(records.get(name, []), key=lambda r: r["id"])
        if name == "diagnose":
            write_jsonl(out / "development" / "diagnose.jsonl", [])
        elif rows or name in ("train", "select", "calibrate"):
            write_jsonl(out / "development" / f"{name}.jsonl", rows)
    write_jsonl(out / "pairs" / "risk.jsonl", sorted(risk_loader, key=lambda p: p["pair_id"]))
    write_jsonl(out / "pairs" / "manipulation.jsonl", sorted(manip_all, key=lambda p: p["pair_id"]))
    write_jsonl(out / "annotation" / "pairs_risk_all_roles.jsonl", sorted(risk_all, key=lambda p: p["pair_id"]))

    # statistics and manifest
    counts = {s: Counter(r["kind"] for r in rows) for s, rows in records.items()}
    family_by_split = Counter(f"{r['split']}/{r['attack_family']}" for rows in records.values() for r in rows if r["kind"] == "jailbreak")
    lineages = defaultdict(set)
    for rows in records.values():
        for r in rows:
            if r["kind"] == "jailbreak":
                lineages[r["template_lineage_id"]].add(r["split"])
    shared = sorted(l for l, s in lineages.items() if len(s) > 1)
    j_per_goal = Counter(sum(1 for r in manip_all if r["goal_cluster_id"] == g["goal_id"]) for g in kept)
    manifest = {
        "schema_version": "dual_concept_development_v1",
        "dataset_name": "dccf_v1",
        "scope": "dccf_v1_construction_set_with_unexposed_confirm_candidate_not_human_verified",
        "label_status": "two_agent_reviewed_adjudicated_not_human_verified",
        "label_semantics": {"0": "safe candidate", "1": "harmful or jailbreak risk candidate"},
        "counts": {"goals": len({r["goal_cluster_id"] for r in risk_all}), "records_by_split": {s: len(r) for s, r in records.items()},
                   "kinds_by_split": {s: dict(c) for s, c in counts.items()},
                   "risk_pairs_all_roles": len(risk_all), "risk_pairs_loader_train_only": len(risk_loader),
                   "manipulation_pairs": len(manip_all),
                   "manipulation_pairs_by_split": dict(Counter(p["split"] for p in manip_all)),
                   "goals_by_source": dict(Counter(g["source"] for g in kept)),
                   "jailbreaks_per_goal": {str(k): v for k, v in sorted(j_per_goal.items())},
                   "jailbreaks_by_family_and_split": dict(sorted(family_by_split.items())),
                   "goals_dropped_or_items_dropped_for_length": dropped_long},
        "review_process": {"reviewers": "two independent LLM subagents per item (different file groupings), disagreements adjudicated by a third LLM subagent; no 'uncertain' label permitted",
                           "human_verified": False},
        "limitations": [
            "Labels come from two LLM reviewers plus an LLM adjudicator, not from human annotators; do not describe them as human ground truth.",
            "B requests for 300+ goals were authored by an LLM as risk-contrast counterparts; style regularities between authored B and source H may remain.",
            "The 'confirm' split was never used for any parameter and was built after the construction rules were frozen, but it shares attack template lineages with train (see lineages_shared_across_splits); it does not test unseen lineages. Use --holdout for family-level generalisation.",
            "J sources are public benchmark artifacts (JailbreakBench artifacts, JailBreakV-28K); Qwen2.5 pretraining overlap is unknown.",
            "Calibration requests come from Dolly and are ordinary instructions; they are not matched to jailbreak formatting. Calibration FPR does not bound deployment FPR.",
            "The three-group design cannot rule out wrapper-style confounding in the manipulation concept; no wrapper-matched benign control set is included.",
            "Attack success on any model was never evaluated and does not influence labels or inclusion."],
        "lineages_shared_across_splits": shared,
        "lineage_count": len(lineages),
        "max_formatted_tokens": max(r["full_input_token_count"] for rows in records.values() for r in rows),
        "tokenizer": {"model": "Qwen/Qwen2.5-7B-Instruct", "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                      "files_sha256": {p.name: sha256(p.read_bytes()) for p in sorted((build / "qwen_tokenizer").iterdir())}},
        "source_revisions": {"JailbreakBench/JBB-Behaviors": "886acc352a31533ffbcf4ef22c744658688086fc",
                             "JailbreakBench/artifacts": "909e68c01d94222b8ad2e397a017e2e12e2adb73",
                             "JailbreakV-28K/JailBreakV-28k": "f949ca582fff13d396ac8fce59596afafb2b78d3",
                             "databricks/databricks-dolly-15k": "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"},
        "builder": "tools/dccf_v1/stage1_select.py + stage3_assemble.py",
    }
    data_files = sorted(str(p.relative_to(out)) for p in out.rglob("*.jsonl"))
    manifest["output_sha256"] = {name: sha256((out / name).read_bytes()) for name in data_files}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=1))
    print("shared lineages:", len(shared), "of", len(lineages))


if __name__ == "__main__":
    main()
