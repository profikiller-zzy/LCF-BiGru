#!/usr/bin/env python3
"""Migrate the recorded phase13 development panel without importing legacy code.

This command is deliberately offline. It preserves the historical partitions and
candidate labels; it does not turn previously examined material into a fresh test.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


LEGACY_NAME = "phase13_dual_concept_qwen7b_v1"
LEGACY_RELATIVE_ROOT = f"attack/DPA/lcf_bigru_results/{LEGACY_NAME}"
SOURCE_FILES = (
    "panel_PRIVATE.jsonl",
    "risk_pairs.jsonl",
    "manipulation_pairs.jsonl",
    "panel_summary.json",
    "preparation_protocol.json",
    "protocol.json",
    "original_goal_ids.json",
)
SPLITS = ("train", "select", "calibrate", "diagnose")
LABEL_STATUS = "candidate_not_independently_validated"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path.name}:{line_number}: invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number}: expected a JSON object")
        records.append((line_number, row))
    return records


def _encode_jsonl(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def resolve_legacy_root(root: Path) -> Path:
    """Accept the panel directory, a results directory, a DPA root, or a repo root."""
    candidates = (
        root,
        root / LEGACY_NAME,
        root / "lcf_bigru_results" / LEGACY_NAME,
        root / LEGACY_RELATIVE_ROOT,
    )
    for candidate in candidates:
        if all((candidate / name).is_file() for name in SOURCE_FILES):
            return candidate
    raise FileNotFoundError(
        f"No complete {LEGACY_NAME} source below {root}; required: {', '.join(SOURCE_FILES)}"
    )


def _verify_historical_hashes(root: Path) -> None:
    registration = json.loads((root / "preparation_protocol.json").read_text(encoding="utf-8"))
    for name, expected in registration.get("artifacts_sha256", {}).items():
        if name in SOURCE_FILES and _sha256(root / name) != expected:
            raise ValueError(f"Source integrity check failed for {name}")


def _validate_panel(records: list[tuple[int, dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    group_splits: dict[str, str] = {}
    seen_text: set[str] = set()
    for line_number, row in records:
        required = ("id", "instruction", "label", "kind", "stage", "cluster_id", "provenance")
        missing = [key for key in required if key not in row]
        if missing:
            raise ValueError(f"panel_PRIVATE.jsonl:{line_number}: missing {missing}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in by_id:
            raise ValueError(f"panel_PRIVATE.jsonl:{line_number}: missing or duplicate id")
        if row["stage"] not in SPLITS:
            raise ValueError(f"Unexpected historical stage: {row['stage']}")
        if type(row["label"]) is not int or row["label"] not in (0, 1):
            raise ValueError(f"Invalid candidate label for {row['id']}")
        if not isinstance(row["instruction"], str) or not row["instruction"].strip():
            raise ValueError(f"Empty instruction for {row['id']}")
        if row["instruction"] in seen_text:
            raise ValueError(f"Duplicate instruction in historical panel: {row['id']}")
        seen_text.add(row["instruction"])
        if not isinstance(row["provenance"], dict):
            raise ValueError(f"Invalid provenance for {row['id']}")
        if row.get("independent_human_validated") is not False:
            raise ValueError(f"Expected recorded unvalidated candidate status for {row['id']}")
        group = row["cluster_id"]
        if not isinstance(group, str) or not group:
            raise ValueError(f"Missing historical group for {row['id']}")
        if group in group_splits and group_splits[group] != row["stage"]:
            raise ValueError(f"Historical group crosses partitions: {group}")
        group_splits[group] = row["stage"]
        by_id[row["id"]] = row
    return by_id


def _kind(row: dict[str, Any]) -> str:
    if row["label"] == 0:
        return "safe"
    if row["kind"] == "public_attack":
        return "jailbreak"
    if row["kind"] in {"natural_pair", "original_goal"}:
        return "harmful"
    raise ValueError(f"Unsupported positive candidate kind: {row['kind']}")


def _pair_rows(
    entries: list[tuple[int, dict[str, Any]]],
    family: str,
    by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    keys = ("harmful_id", "safe_id") if family == "risk" else ("attack_id", "original_id")
    expected_kinds = ("harmful", "safe") if family == "risk" else ("jailbreak", "harmful")
    output = []
    seen_pairs = set()
    source_file = "risk_pairs.jsonl" if family == "risk" else "manipulation_pairs.jsonl"
    for line_number, row in entries:
        positive_id, negative_id = row.get(keys[0]), row.get(keys[1])
        if positive_id not in by_id or negative_id not in by_id:
            raise ValueError(f"{source_file}:{line_number}: pair endpoint missing from panel")
        endpoints = (by_id[positive_id], by_id[negative_id])
        if any(endpoint["stage"] != "train" for endpoint in endpoints):
            raise ValueError(f"{source_file}:{line_number}: concept pairs must remain in train")
        if tuple(_kind(endpoint) for endpoint in endpoints) != expected_kinds:
            raise ValueError(f"{source_file}:{line_number}: incorrect concept-pair orientation")
        if (positive_id, negative_id) in seen_pairs:
            raise ValueError(f"{source_file}:{line_number}: duplicate concept pair")
        seen_pairs.add((positive_id, negative_id))
        output.append({
            "positive_id": positive_id,
            "negative_id": negative_id,
            "split": "train",
            "source": {
                "file": f"{LEGACY_RELATIVE_ROOT}/{source_file}",
                "line": line_number,
                "legacy_fields": row,
            },
            "exposure": "development_exposed",
            "label_status": LABEL_STATUS,
        })
    return output


def _complete_manipulation_pairs(
    panel: list[tuple[int, dict[str, Any]]],
    pairs: list[dict[str, Any]],
    by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Use exact recorded goals, never a merely related request, as J negatives."""
    originals = {pair["positive_id"]: pair["negative_id"] for pair in pairs}
    by_text = {row["instruction"]: row for _, row in panel if row["stage"] != "calibrate"}
    derived = []
    reused = set()
    completed = 0
    for line_number, attack in list(panel):
        if attack["stage"] == "calibrate" or _kind(attack) != "jailbreak":
            continue
        goal = attack["provenance"].get("original_goal")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"No recorded exact original_goal for {attack['id']}")
        if attack["id"] in originals:
            original = by_id[originals[attack["id"]]]
            if original["instruction"] != goal or original["cluster_id"] != attack["cluster_id"]:
                raise ValueError(f"Recorded manipulation original does not match metadata for {attack['id']}")
            continue
        original = by_text.get(goal)
        if original is not None:
            if original["stage"] != attack["stage"]:
                raise ValueError(f"Exact original text crosses partitions for {attack['id']}")
            if _kind(original) != "harmful" or original["cluster_id"] != attack["cluster_id"]:
                raise ValueError(f"Exact original has conflicting label/group for {attack['id']}")
            reused.add(original["id"])
        else:
            original_id = "metadata_original_" + hashlib.sha256(goal.encode("utf-8")).hexdigest()[:20]
            if original_id in by_id:
                raise ValueError(f"Derived original id collision for {attack['id']}")
            original = {
                "id": original_id,
                "instruction": goal,
                "label": 1,
                "kind": "original_goal",
                "stage": attack["stage"],
                "cluster_id": attack["cluster_id"],
                "independent_human_validated": False,
                "label_status": "metadata_derived_original_candidate",
                "provenance": {
                    "source_dataset": "metadata-derived-original",
                    "derived_from_attack_id": attack["id"],
                    "original_goal": goal,
                    "source_goal_metadata": attack["provenance"],
                },
                "metadata_derived_original": True,
            }
            panel.append((line_number, original))
            derived.append(original_id)
            by_id[original_id] = original
            by_text[goal] = original
        originals[attack["id"]] = original["id"]
        pairs.append({
            "positive_id": attack["id"],
            "negative_id": original["id"],
            "split": attack["stage"],
            "source": {
                "file": f"{LEGACY_RELATIVE_ROOT}/panel_PRIVATE.jsonl",
                "line": line_number,
                "derivation": "metadata-derived-original",
                "original_goal_field": "provenance.original_goal",
                "original_goal_sha256": hashlib.sha256(goal.encode("utf-8")).hexdigest(),
                "original_record_reused": not original.get("metadata_derived_original", False),
                "source_metadata_match": True,
                "semantic_preservation_independently_validated": False,
            },
            "exposure": "development_exposed",
            "label_status": LABEL_STATUS,
        })
        completed += 1
    return {
        "added_pairs": completed,
        "added_original_records": len(derived),
        "added_original_ids": derived,
        "reused_existing_original_records": len(reused),
        "rule": "Exact provenance.original_goal text; same partition and group; cross-partition conflicts stop migration.",
    }


def migrate(legacy_root: Path, output: Path) -> dict[str, Any]:
    root = resolve_legacy_root(Path(legacy_root))
    output = Path(output)
    if output.resolve() == root.resolve() or root.resolve() in output.resolve().parents:
        raise ValueError("Migration output must not be inside the historical source directory")
    _verify_historical_hashes(root)
    panel = _read_jsonl(root / "panel_PRIVATE.jsonl")
    source_record_count = len(panel)
    by_id = _validate_panel(panel)
    pairs = {
        "risk": _pair_rows(_read_jsonl(root / "risk_pairs.jsonl"), "risk", by_id),
        "manipulation": _pair_rows(
            _read_jsonl(root / "manipulation_pairs.jsonl"), "manipulation", by_id
        ),
    }
    derived_originals = _complete_manipulation_pairs(panel, pairs["manipulation"], by_id)
    original_by_attack = {}
    for pair in pairs["manipulation"]:
        attack_id, original_id = pair["positive_id"], pair["negative_id"]
        if attack_id in original_by_attack and original_by_attack[attack_id] != original_id:
            raise ValueError(f"Ambiguous recorded original for {attack_id}")
        original_by_attack[attack_id] = original_id

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    excluded = []
    extracted_fields = {
        "id", "instruction", "label", "kind", "stage", "cluster_id", "label_status", "provenance"
    }
    for line_number, row in panel:
        if row["stage"] == "calibrate" and row["label"] != 0:
            excluded.append(row["id"])
            continue
        kind = _kind(row)
        split_rows[row["stage"]].append({
            "id": row["id"],
            "text": row["instruction"],
            "label": row["label"],
            "kind": kind,
            "group_id": row["cluster_id"],
            "attack_family": row["provenance"].get("source_method") if kind == "jailbreak" else None,
            "original_id": original_by_attack.get(row["id"]) if kind == "jailbreak" else None,
            "source": {
                "file": f"{LEGACY_RELATIVE_ROOT}/panel_PRIVATE.jsonl",
                "line": line_number,
                "text_field": "provenance.original_goal" if row.get("metadata_derived_original") else "instruction",
                "derivation": "metadata-derived-original" if row.get("metadata_derived_original") else "legacy-record",
                "legacy_id": None if row.get("metadata_derived_original") else row["id"],
                "legacy_stage": row["stage"],
                "legacy_kind": row["kind"],
                "legacy_label": row["label"],
                "legacy_label_status": row.get("label_status"),
                "provenance": row["provenance"],
                "legacy_metadata": {
                    key: value for key, value in row.items() if key not in extracted_fields
                },
            },
            "exposure": "development_exposed",
            "split": row["stage"],
            "label_status": LABEL_STATUS,
        })

    # Validate the complete migration before writing any output.
    retained = {row["id"]: row for rows in split_rows.values() for row in rows}
    for row in retained.values():
        original_id = row["original_id"]
        if original_id is not None:
            original = retained[original_id]
            if original["split"] != row["split"] or original["kind"] != "harmful":
                raise ValueError(f"Invalid same-partition original endpoint for {row['id']}")
        elif row["kind"] == "jailbreak":
            raise ValueError(f"Missing exact original for jailbreak {row['id']}")
    for split in SPLITS:
        (output / "development").mkdir(parents=True, exist_ok=True)
        (output / "development" / f"{split}.jsonl").write_text(
            _encode_jsonl(split_rows[split]), encoding="utf-8"
        )
    for family, rows in pairs.items():
        (output / "pairs").mkdir(parents=True, exist_ok=True)
        (output / "pairs" / f"{family}.jsonl").write_text(_encode_jsonl(rows), encoding="utf-8")

    counts = {
        split: {
            "records": len(rows),
            "labels": dict(sorted(Counter(str(row["label"]) for row in rows).items())),
            "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        }
        for split, rows in split_rows.items()
    }
    source_summary = json.loads((root / "panel_summary.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "dual_concept_development_v1",
        "migration": "tools/migrate_legacy_data.py",
        "legacy_relative_root": LEGACY_RELATIVE_ROOT,
        "scope": "development_only_not_fresh_confirmation",
        "exposure": "development_exposed",
        "label_status": LABEL_STATUS,
        "label_semantics": {"0": "safe candidate", "1": "harmful or jailbreak risk candidate"},
        "historical_label_limitations": [
            "Source or constructed candidate labels are preserved, not independently validated truth.",
            "Original harmful and safe risk pairs are existing recorded pairs; no new pairing is invented.",
            "Manipulation pairs are matched by recorded source metadata; semantic preservation was not independently validated.",
            "All material has historical development exposure and cannot serve as a fresh confirmatory test.",
            "Safety calibration uses only historical calibrate rows labelled zero; no attack row tunes its threshold.",
        ],
        "source_sha256": {name: _sha256(root / name) for name in SOURCE_FILES},
        "historical_panel_summary": source_summary,
        "source_records": source_record_count,
        "retained_records": len(retained),
        "splits": counts,
        "pairs": {family: len(rows) for family, rows in pairs.items()},
        "pair_splits": {
            family: dict(sorted(Counter(row["split"] for row in rows).items()))
            for family, rows in pairs.items()
        },
        "metadata_derived_originals": derived_originals,
        "unpaired_jailbreaks": sum(
            row["kind"] == "jailbreak" and row["original_id"] is None for row in retained.values()
        ),
        "excluded": {
            "historical_calibrate_positive": {
                "count": len(excluded),
                "ids": excluded,
                "reason": "Safety calibration must contain only safe candidates; original rows remain in the legacy backup.",
            }
        },
    }
    data_files = [f"development/{split}.jsonl" for split in SPLITS] + [
        f"pairs/{family}.jsonl" for family in pairs
    ]
    manifest["output_sha256"] = {name: _sha256(output / name) for name in data_files}
    (output / "manifest.json").write_text(_json_text(manifest), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-root", type=Path,
        default=Path(__file__).resolve().parents[1] / LEGACY_RELATIVE_ROOT,
        help="Historical phase13 directory, results directory, DPA directory, or repository root.",
    )
    parser.add_argument("--output", type=Path, default=Path("data"), help="Destination directory.")
    args = parser.parse_args()
    try:
        manifest = migrate(args.legacy_root, args.output)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Migration failed: {error}\n")
    print(json.dumps({
        "retained_records": manifest["retained_records"],
        "splits": {split: counts["records"] for split, counts in manifest["splits"].items()},
        "pairs": manifest["pairs"],
        "excluded_calibration_positives": manifest["excluded"]["historical_calibrate_positive"]["count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
