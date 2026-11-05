"""Build an auditable, unassigned DCCF pilot from pinned JBB snapshots.

This script only reuses published prompts. It does not generate attacks or
model responses. All pairings remain candidates until human review.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import unicodedata
from typing import Any, Mapping
from urllib.request import Request, urlopen


JBB_REVISION = "886acc352a31533ffbcf4ef22c744658688086fc"
ARTIFACTS_REVISION = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
USER_AGENT = "DCCF-dataset-builder/1.0"

SOURCES: dict[str, dict[str, Any]] = {
    "harmful_behaviors": {
        "url": f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REVISION}/data/harmful-behaviors.csv",
        "sha256": "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
        "license": "MIT",
    },
    "benign_behaviors": {
        "url": f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REVISION}/data/benign-behaviors.csv",
        "sha256": "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
        "license": "MIT",
    },
    "jbb_dataset_license": {
        "url": f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REVISION}/LICENSE",
        "sha256": "7e1ae8f3676f39ff6e62af1743385c176a9476b1157f35bb9d05b4e6c65b7d94",
        "license": "MIT",
    },
    "pair_artifact": {
        "url": f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ARTIFACTS_REVISION}/attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json",
        "sha256": "18b101386419a98fbdf45cb987069f83af3a40d5e0f7f0d32c4e9dc5ecc049b4",
        "license": "MIT",
        "method": "PAIR",
        "attack_type": "black_box",
        "source_model": "vicuna-13b-v1.5",
    },
    "gcg_artifact": {
        "url": f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ARTIFACTS_REVISION}/attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json",
        "sha256": "9eff8b6f78cae002737868ca8cdd886be5afcd1cbbbe4cf51ff837bc688b01b1",
        "license": "MIT",
        "method": "GCG",
        "attack_type": "white_box",
        "source_model": "vicuna-13b-v1.5",
    },
    "artifacts_license": {
        "url": f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ARTIFACTS_REVISION}/LICENSE",
        "sha256": "09017abee723a7b715866bbcd1bd2569e0a5fa7e996dd1d9d1ec4b36fb5ca51c",
        "license": "MIT",
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _fetch(source: Mapping[str, Any], *, timeout: int = 45) -> bytes:
    request = Request(source["url"], headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
    digest = sha256_bytes(content)
    if digest != source["sha256"]:
        raise ValueError(f"Pinned source hash mismatch: expected {source['sha256']}, got {digest}")
    return content


def _csv_rows(payload: bytes, name: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))
    required = {"Index", "Goal", "Behavior", "Category", "Source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Unexpected CSV schema for {name}")
    indexes = [int(row["Index"]) for row in rows]
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"Duplicate source Index in {name}")
    return rows


def _json_bytes(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid JSON source {name}") from error
    if not isinstance(value, dict) or not isinstance(value.get("jailbreaks"), list):
        raise ValueError(f"Unexpected attack artifact schema for {name}")
    return value


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    return content.encode("utf-8")


def _annotation_csv(rows: list[dict[str, Any]], records_by_id: Mapping[str, dict[str, Any]], branch: str) -> bytes:
    if branch == "risk":
        columns = ["pair_id", "candidate_rank", "goal_cluster_id_candidate", "harmful_id", "harmful_text",
                   "benign_id", "benign_text", "category", "character_length_difference",
                   "reviewer1_id", "reviewer1_decision", "reviewer1_reason",
                   "reviewer2_id", "reviewer2_decision", "reviewer2_reason",
                   "adjudicator_id", "final_decision", "final_goal_cluster_id", "final_reason"]
        mapped = []
        for row in rows:
            h = records_by_id[row["positive_id"]]
            b = records_by_id[row["negative_id"]]
            mapped.append({
                "pair_id": row["pair_id"], "candidate_rank": row["candidate_rank"],
                "goal_cluster_id_candidate": row["proposed_harmful_goal_cluster_id"],
                "harmful_id": row["positive_id"], "harmful_text": h["text"],
                "benign_id": row["negative_id"], "benign_text": b["text"],
                "category": h["category"], "character_length_difference": row["character_length_difference"],
            })
    elif branch == "manipulation":
        columns = ["pair_id", "attack_family", "goal_cluster_id", "jailbreak_id", "jailbreak_text",
                   "harmful_id", "harmful_text", "source_mapping_evidence",
                   "reviewer1_id", "reviewer1_goal_preserved", "reviewer1_reason",
                   "reviewer2_id", "reviewer2_goal_preserved", "reviewer2_reason",
                   "adjudicator_id", "final_goal_preserved", "final_reason"]
        mapped = []
        for row in rows:
            j = records_by_id[row["positive_id"]]
            h = records_by_id[row["negative_id"]]
            mapped.append({
                "pair_id": row["pair_id"], "attack_family": row["attack_family"],
                "goal_cluster_id": row["goal_cluster_id"], "jailbreak_id": row["positive_id"],
                "jailbreak_text": j["text"], "harmful_id": row["negative_id"], "harmful_text": h["text"],
                "source_mapping_evidence": row["mapping_evidence"],
            })
    else:
        raise ValueError(f"Unknown annotation branch: {branch}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(mapped)
    return stream.getvalue().encode("utf-8")


def _unpaired_annotation_csv(rows: list[dict[str, Any]], harmful_rows: Mapping[int, dict[str, str]]) -> bytes:
    columns = ["record_id", "source", "source_row_id", "candidate_harmful_id", "candidate_harmful_text",
               "jailbreak_text", "goal_text_matches", "behavior_matches", "category_matches",
               "reviewer1_decision", "reviewer1_reason", "reviewer2_decision", "reviewer2_reason",
               "adjudicator_id", "final_decision", "final_harmful_id", "final_reason"]
    mapped = []
    for row in rows:
        index = int(row["source_row_id"])
        harmful = harmful_rows[index]
        evidence = row["mapping_evidence"]
        mapped.append({
            "record_id": row["record_id"], "source": row["source"], "source_row_id": row["source_row_id"],
            "candidate_harmful_id": f"jbb_h_{index:03d}", "candidate_harmful_text": harmful["Goal"],
            "jailbreak_text": row["text"], "goal_text_matches": evidence["goal_text_matches"],
            "behavior_matches": evidence["behavior_matches"], "category_matches": evidence["category_matches"],
        })
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(mapped)
    return stream.getvalue().encode("utf-8")


def _candidate_benign_matches(harmful: list[dict[str, str]], benign: list[dict[str, str]]) -> list[tuple[int, int, int, int]]:
    """Return the three nearest same-category character-length suggestions.

    Character length is only a cheap review-queue heuristic. It is not token
    matching, semantic validation, or a final risk pair.
    """
    by_category_h: dict[str, list[dict[str, str]]] = {}
    by_category_b: dict[str, list[dict[str, str]]] = {}
    for row in harmful:
        by_category_h.setdefault(row["Category"], []).append(row)
    for row in benign:
        by_category_b.setdefault(row["Category"], []).append(row)

    matches: list[tuple[int, int, int, int]] = []
    if set(by_category_h) != set(by_category_b):
        raise ValueError("Harmful and benign source categories do not match")
    for category in sorted(by_category_h):
        left = sorted(by_category_h[category], key=lambda row: (len(row["Goal"]), int(row["Index"])))
        right = sorted(by_category_b[category], key=lambda row: (len(row["Goal"]), int(row["Index"])))
        for h_row in left:
            candidates = sorted(right, key=lambda row: (abs(len(row["Goal"]) - len(h_row["Goal"])), int(row["Index"])))[:3]
            for rank, b_row in enumerate(candidates, 1):
                matches.append((int(h_row["Index"]), int(b_row["Index"]), rank,
                                abs(len(b_row["Goal"]) - len(h_row["Goal"]))))
    return sorted(matches)


def build_payloads(source_payloads: Mapping[str, bytes]) -> dict[str, bytes]:
    """Transform pinned input bytes into deterministic review-stage files."""
    missing = set(SOURCES) - set(source_payloads)
    if missing:
        raise ValueError(f"Missing source payloads: {sorted(missing)}")
    for name, source in SOURCES.items():
        if sha256_bytes(source_payloads[name]) != source["sha256"]:
            raise ValueError(f"Pinned source hash mismatch for {name}")
    for name in ("jbb_dataset_license", "artifacts_license"):
        if not source_payloads[name].lstrip().startswith(b"MIT License"):
            raise ValueError(f"Expected pinned MIT license text for {name}")

    harmful = _csv_rows(source_payloads["harmful_behaviors"], "harmful_behaviors")
    benign = _csv_rows(source_payloads["benign_behaviors"], "benign_behaviors")
    if len(harmful) != 100 or len(benign) != 100:
        raise ValueError("Pilot source must contain the pinned 100 harmful and 100 benign rows")
    harmful_by_index = {int(row["Index"]): row for row in harmful}
    benign_by_index = {int(row["Index"]): row for row in benign}
    if set(harmful_by_index) != set(range(100)) or set(benign_by_index) != set(range(100)):
        raise ValueError("Source Index values must be exactly 0..99")

    records: list[dict[str, Any]] = []
    used_text: dict[str, str] = {}
    excluded: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    manipulation_rows: list[dict[str, Any]] = []
    external_unpaired: list[dict[str, Any]] = []

    def add_record(record: dict[str, Any]) -> None:
        if not isinstance(record.get("text"), str) or not record["text"].strip():
            raise ValueError(f"Empty text for {record.get('record_id')}")
        record["text_sha256"] = sha256_bytes(record["text"].encode("utf-8"))
        key = canonical_text(record["text"])
        previous = used_text.get(key)
        if previous is not None:
            raise ValueError(f"Exact normalized text collision between {previous} and {record['record_id']}")
        used_text[key] = record["record_id"]
        records.append(record)

    def base_record(record_id: str, text: str, core_group: str, label: int, goal_id: str,
                    source_name: str, source_row_id: str, category: str, source_tag: str) -> dict[str, Any]:
        return {
            "record_id": record_id,
            "text": text,
            "core_group": core_group,
            # Runtime-compatible aliases; "safe" means benign request, not a model response.
            "kind": "safe" if core_group == "benign" else core_group,
            "label": label,
            "risk_label": label,
            "goal_cluster_id": goal_id,
            "original_id": None,
            "original_goal_id": source_row_id if core_group != "benign" else None,
            "split": None,
            "source_dataset": "JBB-Behaviors",
            "source_revision": JBB_REVISION,
            "source_file": source_name,
            "source_row_id": source_row_id,
            "source_url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
            "source_license": "MIT",
            "source_tag": source_tag,
            "category": category,
            "language": "en",
            "turn_count": 1,
            "full_input_token_count": None,
            "harmful_goal": "yes" if core_group in {"harmful", "jailbreak"} else "no",
            "harmful_goal_status": "source_candidate_pending_human_review",
            "manipulation_intent": "yes" if core_group == "jailbreak" else "no",
            "manipulation_intent_status": "source_candidate_pending_human_review",
            "goal_preserved": "not_applicable",
            "attack_success": "not_evaluated",
            "attack_success_model": None,
            "attack_method": None,
            "attack_family": None,
            "template_lineage_id": None,
            "label_status": "source_candidate_pending_human_review",
            "reviewer_ids": [],
            "exposure": "external_source_overlap_audit_pending",
            "source_provenance": {
                "dataset_url": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
                "dataset_revision": JBB_REVISION,
                "source_row_id": source_row_id,
                "source_tag": source_tag,
            },
        }

    for index in range(100):
        h_row = harmful_by_index[index]
        b_row = benign_by_index[index]
        goal_id = f"jbb_goal_{index:03d}"
        h_id = f"jbb_h_{index:03d}"
        b_id = f"jbb_b_{index:03d}"
        add_record(base_record(h_id, h_row["Goal"], "harmful", 1, goal_id, "harmful-behaviors.csv",
                               str(index), h_row["Category"], h_row["Source"]))
        add_record(base_record(b_id, b_row["Goal"], "benign", 0, b_id, "benign-behaviors.csv",
                               str(index), b_row["Category"], b_row["Source"]))

    for method_key in ("pair_artifact", "gcg_artifact"):
        spec = SOURCES[method_key]
        artifact = _json_bytes(source_payloads[method_key], method_key)
        rows = artifact["jailbreaks"]
        if len(rows) != 100 or {row.get("index") for row in rows} != set(range(100)):
            raise ValueError(f"Unexpected behavior index coverage in {method_key}")
        method = spec["method"]
        for row in sorted(rows, key=lambda item: item["index"]):
            index = row["index"]
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                excluded.append({"source": method_key, "source_row_id": str(index), "reason": "published_artifact_has_no_prompt"})
                continue
            source_h = harmful_by_index[index]
            goal_match = canonical_text(str(row.get("goal") or "")) == canonical_text(source_h["Goal"])
            behavior_match = row.get("behavior") == source_h["Behavior"]
            category_match = row.get("category") == source_h["Category"]
            mapping_exact = goal_match and behavior_match and category_match
            record_id = f"jbb_j_{method.lower()}_{index:03d}"
            record = base_record(record_id, prompt, "jailbreak", 1, f"jbb_goal_{index:03d}",
                                 f"{method.lower()}-artifact.json", str(index), source_h["Category"], source_h["Source"])
            success = row.get("jailbroken")
            record.update({
                "source_dataset": "JailbreakBench-artifacts",
                "source_revision": ARTIFACTS_REVISION,
                "source_license": "MIT",
                "harmful_goal": "yes",
                "original_id": f"jbb_h_{index:03d}",
                "original_goal_id": str(index),
                "manipulation_intent": "yes",
                "manipulation_intent_status": "published_attack_candidate_pending_human_review",
                "goal_preserved": "uncertain",
                "attack_success": "success" if success is True else "failure" if success is False else "unknown",
                "attack_success_model": spec["source_model"],
                "attack_method": method,
                "attack_family": "gcg" if method == "GCG" else "pair_tap",
                "template_lineage_id": f"jbb:{method}:{spec['attack_type']}:{spec['source_model']}",
                "label_status": "published_attack_attempt_pending_human_goal_review",
                "source_provenance": {
                    "artifact_url": spec["url"],
                    "artifact_revision": ARTIFACTS_REVISION,
                    "artifact_index": index,
                    "source_model": spec["source_model"],
                    "source_attack_type": spec["attack_type"],
                    "published_attack_success": success,
                    "index_matches_harmful_source": True,
                    "goal_text_matches": goal_match,
                    "behavior_matches": behavior_match,
                    "category_matches": category_match,
                },
            })
            record["text_sha256"] = sha256_bytes(prompt.encode("utf-8"))
            if mapping_exact:
                add_record(record)
                manipulation_rows.append({
                    "pair_id": f"jbb_jh_{method.lower()}_{index:03d}",
                    "positive_id": record_id,
                    "negative_id": f"jbb_h_{index:03d}",
                    "goal_cluster_id": f"jbb_goal_{index:03d}",
                    "split": None,
                    "attack_family": record["attack_family"],
                    "goal_preserved": "uncertain",
                    "mapping_status": "source_fields_match_pending_human_goal_preservation_review",
                    "mapping_evidence": "artifact index, normalized goal, behavior identifier, and category match the pinned JBB harmful row",
                })
            else:
                external_unpaired.append({
                    "record_id": record_id,
                    "source": method_key,
                    "source_row_id": str(index),
                    "text": prompt,
                    "text_sha256": record["text_sha256"],
                    "source_provenance": record["source_provenance"],
                    "reason": "published artifact fields do not all match the linked harmful row; manual mapping review required",
                    "mapping_evidence": {"goal_text_matches": goal_match, "behavior_matches": behavior_match,
                                         "category_matches": category_match, "index_matches": True},
                })

    for h_index, b_index, rank, length_delta in _candidate_benign_matches(harmful, benign):
        pair_rows.append({
            "pair_id": f"jbb_hb_{h_index:03d}_candidate_{rank}",
            "positive_id": f"jbb_h_{h_index:03d}",
            "negative_id": f"jbb_b_{b_index:03d}",
            "goal_cluster_id": None,
            "proposed_harmful_goal_cluster_id": f"jbb_goal_{h_index:03d}",
            "split": None,
            "match_status": "heuristic_candidate_pending_dual_human_review",
            "match_method": "same broad category and nearest character-length shortlist",
            "candidate_rank": rank,
            "requires_unique_benign_endpoint_in_final_match": True,
            "character_length_difference": length_delta,
            "warning": "Not a semantic match; do not use for fitting until reviewed and adjudicated.",
        })

    records.sort(key=lambda row: row["record_id"])
    source_lock = {
        "schema_version": "dccf_source_lock_v1",
        "sources": {
            name: {key: value for key, value in spec.items() if key in ("url", "sha256", "license", "method", "attack_type", "source_model")}
            for name, spec in SOURCES.items()
        },
        "repository_revisions": {"JailbreakBench/JBB-Behaviors": JBB_REVISION, "JailbreakBench/artifacts": ARTIFACTS_REVISION},
        "license_texts_verified": True,
    }
    records_by_id = {row["record_id"]: row for row in records}
    files = {
        "records.jsonl": _jsonl(records),
        "pairs/risk_candidates.jsonl": _jsonl(pair_rows),
        "pairs/manipulation_candidates.jsonl": _jsonl(manipulation_rows),
        "review/external_unpaired_jailbreaks.jsonl": _jsonl(external_unpaired),
        "review/excluded_artifact_rows.jsonl": _jsonl(excluded),
        "review/risk_pair_review.csv": _annotation_csv(pair_rows, records_by_id, "risk"),
        "review/manipulation_pair_review.csv": _annotation_csv(manipulation_rows, records_by_id, "manipulation"),
        "review/unpaired_mapping_review.csv": _unpaired_annotation_csv(external_unpaired, harmful_by_index),
        "licenses/JBB-Behaviors-MIT.txt": source_payloads["jbb_dataset_license"],
        "licenses/JailbreakBench-artifacts-MIT.txt": source_payloads["artifacts_license"],
        "source_lock.json": (json.dumps(source_lock, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "README.md": ("""# DCCF JBB pilot v1 — review stage only

This directory contains an unassigned 100-goal construction pilot from pinned
JailbreakBench (JBB-Behaviors) and its published PAIR/GCG artifacts. It is a
candidate pool for human review, not a model-ready train/evaluation dataset.

- `records.jsonl` contains 100 harmful candidates, 100 benign candidates, and
  only jailbreak prompts whose pinned artifact index, normalized goal,
  behavior identifier, and category all match the corresponding harmful row.
- `pairs/manipulation_candidates.jsonl` indexes those source-linked J-H pairs.
- `pairs/risk_candidates.jsonl` contains three same-category, nearest-character-
  length B suggestions for each H. These are not semantic matches.
- `review/*.csv` are independent-review worksheets. Reviewers should not see
  model scores or the artifact success outcome when judging input semantics.
- `review/external_unpaired_jailbreaks.jsonl` preserves prompts whose source
  fields do not all match the linked H; they are excluded from the core records.
- `review/excluded_artifact_rows.jsonl` logs source artifact entries with no
  prompt field. Source model responses and judge outputs are not retained.
- `licenses/` stores the MIT license texts retrieved from the pinned sources.

`review/ANNOTATION_GUIDE.md` defines the double-review and adjudication rules.

Every record remains a source candidate. Published attack success is metadata
for the named source model only; it does not affect the global input risk label
or inclusion. No train/select/calibrate/diagnose/confirmation role is assigned.
Before fitting, complete two-person review, adjudicate conflicts, check external
overlap with historical DCCF material, tokenize the full chat-template inputs
for the target backbone without truncation, and assign groups/splits under the
protocol in `docs/DATASET_CONSTRUCTION_PLAN_V1.md` and `docs/METHOD.md`.

Rebuild from the hash-pinned sources (standard library only):

```bash
python tools/build_jbb_pilot.py --output data/dccf_v1_jbb_pilot
```

The builder refuses to overwrite an existing output directory.
""").encode("utf-8"),
        "review/ANNOTATION_GUIDE.md": ("""# DCCF pilot annotation guide

Two reviewers independently assess each row, without model scores, detector
predictions, or source-model responses. Use `accepted`, `rejected`, or
`uncertain`; record a short evidence-based reason. An adjudicator resolves
conflicts or leaves the item uncertain. Keep reviewer IDs and decisions.

For H-B candidates, confirm that H is a direct request for a harmful goal and
that B is a genuinely benign, executable request with sufficiently similar
topic and task form. Same broad category and similar character count are only
candidate-generation heuristics. Do not treat safe discussion, prevention, or
critique of the harmful goal as an automatic match. A benign record can appear
in several candidate rows, but the final accepted pair map should not reuse a
benign endpoint unless that design is explicitly frozen.

For J-H candidates, confirm that the complete J input attempts to bypass or
override safeguards and preserves the H goal. Attack success on the artifact's
source model is descriptive metadata only; it is not an inclusion criterion.
If source fields disagree or intent/goal preservation is unclear, leave the J
outside the core paired set and retain the reason in the unpaired review file.

After adjudication, audit exact and near-duplicate text, goal overlap, attack
template lineage, and overlap with historical DCCF development material.
Assign complete goal groups to train/select/independent evaluation only after
this audit. Create a separate benign-only calibration pool from non-overlapping
goal groups; never repurpose exposed `diagnose` material as confirmation.
""").encode("utf-8"),
    }
    source_hashes = {name: sha256_bytes(payload) for name, payload in source_payloads.items()}
    summary = {
        "schema_version": "dccf_jbb_pilot_candidates_v1",
        "scope": "pilot_candidate_material_only_not_training_ready",
        "split_assignment": "none_pending_review_and_leakage_audit",
        "source_hashes_verified": True,
        "source_hashes": source_hashes,
        "source_revisions": {"JailbreakBench/JBB-Behaviors": JBB_REVISION, "JailbreakBench/artifacts": ARTIFACTS_REVISION},
        "label_status": "source_candidate_pending_human_review",
        "counts": {
            "goal_clusters": 100,
            "records": len(records),
            "benign_records": sum(row["core_group"] == "benign" for row in records),
            "harmful_records": sum(row["core_group"] == "harmful" for row in records),
            "jailbreak_records": sum(row["core_group"] == "jailbreak" for row in records),
            "risk_pair_candidates": len(pair_rows),
            "manipulation_pair_candidates": len(manipulation_rows),
            "unpaired_jailbreaks_pending_mapping_review": len(external_unpaired),
            "artifact_rows_without_prompt": len(excluded),
            "human_reviewed_records": 0,
            "assigned_splits": 0,
        },
        "limitations": [
            "Harmful and jailbreak labels are source candidates, not independently adjudicated labels.",
            "Benign pairing is only category and character-length based; it is a review suggestion, not a semantic match.",
            "Jailbreak success is source-model metadata and never determines the input risk label or pair inclusion.",
            "Only published prompt text is retained; model responses and judge outputs are omitted.",
            "Text-level and goal-level overlap with prior DCCF development data has not yet been audited.",
            "Unassigned pilot data are not train, select, calibrate, diagnose, or confirmation data.",
            "A three-group dataset cannot by itself rule out wrapper-style confounding in the manipulation concept.",
        ],
        "output_sha256": {name: sha256_bytes(payload) for name, payload in files.items()},
    }
    files["manifest.json"] = (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return files


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists; refusing to overwrite: {output}")
    payloads = {name: _fetch(spec) for name, spec in SOURCES.items()}
    files = build_payloads(payloads)
    output.mkdir(parents=True)
    try:
        for relative, content in files.items():
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
    except Exception:
        # Do not leave a seemingly complete pilot after a write failure.
        import shutil
        shutil.rmtree(output, ignore_errors=True)
        raise
    return json.loads(files["manifest.json"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/dccf_v1_jbb_pilot"))
    args = parser.parse_args()
    manifest = build(args.output)
    print(json.dumps({"output": str(args.output), "counts": manifest["counts"],
                      "manifest_sha256": sha256_bytes((args.output / "manifest.json").read_bytes())},
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
