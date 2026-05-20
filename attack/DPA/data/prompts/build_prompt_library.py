#!/usr/bin/env python3
"""
Build the directory-managed prompt library used by BiGRU experiments.

Output layout:
  data/prompts/benign/benign_prompts.json
  data/prompts/jailbreak/<attack_type>/<attack_type>_prompts.json

Each final prompt record has:
  instruction, input, output, label, prompt_type, attack_type, source_dataset

This script integrates:
  - prompts_final_dedup.json for benign/generic and in-the-wild typed prompts
  - existing LCF test_data jailbreak prompts
  - downloaded JailbreakBench artifacts under _raw_sources/

It does not fabricate AutoDAN/GCG/PAIR outputs from seed goals. Seed goals are
kept under _raw_sources and reported as pending generation.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


DPA_ROOT = Path(__file__).resolve().parents[2]
PROMPT_ROOT = DPA_ROOT / "data" / "prompts"
TEST_DATA = DPA_ROOT / "data" / "test_data"
RAW = PROMPT_ROOT / "_raw_sources"

SCHEMA_KEYS = [
    "instruction",
    "input",
    "output",
    "label",
    "prompt_type",
    "attack_type",
    "source_dataset",
]


def clean(value: str) -> str:
    return (value or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").strip()


def load_json(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def record(instruction: str, *, label: int, prompt_type: str, attack_type: str, source_dataset: str, user_input: str = "", output: str = "") -> dict:
    return {
        "instruction": clean(instruction),
        "input": clean(user_input),
        "output": clean(output),
        "label": int(label),
        "prompt_type": prompt_type,
        "attack_type": attack_type,
        "source_dataset": source_dataset,
    }


def dedupe(records: list[dict]) -> list[dict]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict] = []
    for item in records:
        instruction = clean(item.get("instruction", ""))
        if not instruction:
            continue
        normalized = {k: item[k] for k in SCHEMA_KEYS}
        normalized["instruction"] = instruction
        key = (normalized["prompt_type"], normalized["attack_type"], instruction)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)


def load_prompts_final() -> dict[str, list[dict]]:
    data = load_json(PROMPT_ROOT / "prompts_final_dedup.json")
    buckets: dict[str, list[dict]] = {
        "benign": [],
        "generic": [],
        "dan": [],
        "roleplay": [],
        "pair": [],
    }
    for row in data:
        attack_type = row.get("attack_type")
        if row.get("label") == 0:
            buckets["benign"].append(
                record(
                    row.get("instruction", ""),
                    label=0,
                    prompt_type="benign",
                    attack_type="benign",
                    source_dataset=row.get("source_dataset", "prompts_regular"),
                )
            )
        elif attack_type == "generic_jailbreak":
            buckets["generic"].append(
                record(
                    row.get("instruction", ""),
                    label=1,
                    prompt_type="jailbreak",
                    attack_type="generic",
                    source_dataset=row.get("source_dataset", "prompts_jailbreak"),
                )
            )
        elif attack_type in {"dan", "roleplay", "pair"}:
            buckets[attack_type].append(
                record(
                    row.get("instruction", ""),
                    label=1,
                    prompt_type="jailbreak",
                    attack_type=attack_type,
                    source_dataset=row.get("source_dataset", f"prompts_{attack_type}"),
                )
            )
    return {k: dedupe(v) for k, v in buckets.items()}


def load_existing_lcf_jailbreaks() -> dict[str, list[dict]]:
    buckets = {"gcg": [], "pair": [], "dan": [], "roleplay": []}
    for row in load_json(TEST_DATA / "poison" / "jailbreak" / "gcg" / "backdoor200_jailbreak_gcg.json"):
        buckets["gcg"].append(
            record(
                row.get("instruction", ""),
                label=1,
                prompt_type="jailbreak",
                attack_type="gcg",
                source_dataset=row.get("source") or "jailbreakbench_gcg",
            )
        )
    for row in load_json(TEST_DATA / "poison" / "jailbreak" / "pair" / "pair_jailbreaks.json"):
        buckets["pair"].append(
            record(
                row.get("instruction", ""),
                label=1,
                prompt_type="jailbreak",
                attack_type="pair",
                source_dataset=row.get("source") or "pair",
            )
        )
    for row in load_json(TEST_DATA / "poison" / "jailbreak" / "dan" / "dan_jailbreaks.json"):
        source = row.get("source") or ""
        attack_type = "roleplay" if source == "roleplay_template" else "dan"
        buckets[attack_type].append(
            record(
                row.get("instruction", ""),
                label=1,
                prompt_type="jailbreak",
                attack_type=attack_type,
                source_dataset=source or attack_type,
            )
        )
    return {k: dedupe(v) for k, v in buckets.items()}


def load_jailbreakbench_artifacts(attack_type: str) -> list[dict]:
    base = RAW / "jailbreakbench_artifacts" / attack_type.upper()
    items: list[dict] = []
    if not base.exists():
        return items
    for path in sorted(base.rglob("*.json")):
        payload = load_json(path)
        if not isinstance(payload, dict):
            continue
        params = payload.get("parameters", {})
        method = str(params.get("method") or attack_type).lower()
        model = str(params.get("model") or path.stem).replace("/", "_")
        setting = str(params.get("attack_type") or path.parent.name)
        source_dataset = f"jailbreakbench_{method}_{setting}_{model}"
        for row in payload.get("jailbreaks", []):
            items.append(
                record(
                    row.get("prompt") or row.get("instruction") or "",
                    label=1,
                    prompt_type="jailbreak",
                    attack_type=attack_type.lower(),
                    source_dataset=source_dataset,
                )
            )
    return dedupe(items)


def count_seed_goals(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return sum(1 for row in reader if clean(row.get("goal", "")))


def validate(records: list[dict], prompt_type: str, attack_type: str) -> None:
    for idx, item in enumerate(records):
        if list(item.keys()) != SCHEMA_KEYS:
            raise ValueError(f"Bad schema {attack_type}[{idx}]: {list(item.keys())}")
        if not item["instruction"]:
            raise ValueError(f"Empty instruction {attack_type}[{idx}]")
        if item["prompt_type"] != prompt_type or item["attack_type"] != attack_type:
            raise ValueError(f"Bad type {attack_type}[{idx}]: {item}")
        if prompt_type == "benign" and item["label"] != 0:
            raise ValueError(f"Bad benign label {idx}")
        if prompt_type == "jailbreak" and item["label"] != 1:
            raise ValueError(f"Bad jailbreak label {idx}")


def main() -> None:
    final = load_prompts_final()
    lcf = load_existing_lcf_jailbreaks()

    outputs = {
        "benign": final["benign"],
        "generic": final["generic"],
        "gcg": dedupe(lcf["gcg"] + load_jailbreakbench_artifacts("gcg")),
        "pair": dedupe(lcf["pair"] + load_jailbreakbench_artifacts("pair") + final["pair"]),
        "dan": dedupe(lcf["dan"] + final["dan"]),
        "roleplay": dedupe(lcf["roleplay"] + final["roleplay"]),
    }

    validate(outputs["benign"], "benign", "benign")
    for attack_type in ["generic", "gcg", "pair", "dan", "roleplay"]:
        validate(outputs[attack_type], "jailbreak", attack_type)

    write_json(PROMPT_ROOT / "benign" / "benign_prompts.json", outputs["benign"])
    for attack_type in ["generic", "gcg", "pair", "dan", "roleplay"]:
        write_json(PROMPT_ROOT / "jailbreak" / attack_type / f"{attack_type}_prompts.json", outputs[attack_type])

    summary = {
        "schema": SCHEMA_KEYS,
        "target_per_attack_type": 500,
        "counts": {k: len(v) for k, v in outputs.items()},
        "remaining_to_500": {
            "generic": max(0, 500 - len(outputs["generic"])),
            "gcg": max(0, 500 - len(outputs["gcg"])),
            "pair": max(0, 500 - len(outputs["pair"])),
            "dan": max(0, 500 - len(outputs["dan"])),
            "roleplay": max(0, 500 - len(outputs["roleplay"])),
            "autodan": 500,
        },
        "raw_seed_goal_counts": {
            "advbench_llm_attacks": count_seed_goals(RAW / "advbench" / "llm_attacks_harmful_behaviors.csv"),
            "autodan_official_advbench": count_seed_goals(RAW / "autodan" / "official_advbench_harmful_behaviors.csv"),
        },
        "source_dataset_counts": {
            k: dict(Counter(item["source_dataset"] for item in v))
            for k, v in outputs.items()
        },
        "note": "AutoDAN and additional GCG/PAIR prompts require running generation algorithms over the downloaded AdvBench/JailbreakBench seed goals; they are not fabricated here.",
    }
    write_json(PROMPT_ROOT / "library_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
