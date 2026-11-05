"""Validated input records, paired concepts and development split isolation."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import unicodedata

SPLITS = ("train", "select", "calibrate", "diagnose")
KINDS = {"safe", "harmful", "jailbreak"}


def canonical_family(value: str | None) -> str:
    name = (value or "").strip().lower().replace("-", "").replace("_", "")
    return {"nanogcg": "gcg", "gcg": "gcg", "pair": "pair_tap", "tap": "pair_tap",
            "pair/tap": "pair_tap", "pairtap": "pair_tap", "gcg/nanogcg": "gcg"}.get(name, name)


def normalized_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path}:{number}: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object at {path}:{number}")
        rows.append(value)
    return rows


def validate_records(rows: Sequence[Mapping[str, Any]], *, split: str | None = None) -> None:
    seen = set()
    for row in rows:
        for name in ("id", "text", "group_id", "kind", "exposure", "label_status"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"Every record needs nonempty {name}")
        if row["id"] in seen:
            raise ValueError(f"Duplicate record id: {row['id']}")
        seen.add(row["id"])
        if row["kind"] not in KINDS or type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError(f"Invalid kind/label for {row['id']}")
        if (row["kind"] == "safe") != (row["label"] == 0):
            raise ValueError(f"Risk label conflicts with kind: {row['id']}")
        if "source" in row and not isinstance(row["source"], dict):
            raise ValueError(f"Record source must be a provenance object: {row['id']}")
        if row.get("attack_family") is not None and not isinstance(row["attack_family"], str):
            raise ValueError(f"attack_family must be a string or null: {row['id']}")
        if row["kind"] == "jailbreak" and not canonical_family(row.get("attack_family")):
            raise ValueError(f"Jailbreak row needs an attack_family: {row['id']}")
        if split is not None and row.get("split") != split:
            raise ValueError(f"Record split disagrees with file: {row['id']}")
        if split == "calibrate" and row["kind"] != "safe":
            raise ValueError("Deployment threshold calibration accepts safety-only records")


@dataclass
class Dataset:
    splits: dict[str, list[dict[str, Any]]]
    risk_pairs: list[dict[str, Any]]
    manipulation_pairs: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def records(self) -> list[dict[str, Any]]:
        return [row for rows in self.splits.values() for row in rows]

    def validate(self) -> None:
        ids: dict[str, dict[str, Any]] = {}
        groups: dict[str, str] = {}
        texts: dict[str, str] = {}
        for split, rows in self.splits.items():
            if split not in (*SPLITS, "confirm"):
                raise ValueError(f"Unsupported split: {split}")
            validate_records(rows, split=split)
            for row in rows:
                if row["id"] in ids:
                    raise ValueError(f"Record reused across splits: {row['id']}")
                ids[row["id"]] = row
                for lookup, key, label in ((groups, row["group_id"], "goal component"),
                                            (texts, normalized_text(row["text"]), "normalized text")):
                    if key in lookup and lookup[key] != split:
                        raise ValueError(f"Cross-split {label} leakage involving {row['id']}")
                    lookup[key] = split
                if split == "confirm" and row["exposure"] != "independent_confirmation":
                    raise ValueError("Confirmation rows must explicitly declare independent_confirmation")
        for branch, pairs in (("risk", self.risk_pairs), ("manipulation", self.manipulation_pairs)):
            seen = set()
            for pair in pairs:
                key = (pair.get("negative_id"), pair.get("positive_id"))
                if key in seen or any(item not in ids for item in key):
                    raise ValueError(f"Duplicate or unresolved {branch} pair: {key}")
                seen.add(key)
                negative, positive = (ids[item] for item in key)
                expected = ("safe", "harmful") if branch == "risk" else ("harmful", "jailbreak")
                if (negative["kind"], positive["kind"]) != expected:
                    raise ValueError(f"Wrong endpoint kinds in {branch} pair")
                if negative["split"] != positive["split"] or pair.get("split") != positive["split"]:
                    raise ValueError("Concept pairs must remain inside one split")
                if negative["group_id"] != positive["group_id"]:
                    raise ValueError("Concept pairs must share one goal component")
                if branch == "risk" and pair["split"] != "train":
                    raise ValueError("Risk construction pairs are training-only")
                if branch == "manipulation" and positive.get("original_id") != negative["id"]:
                    raise ValueError("Manipulation pair disagrees with corresponding original_id")
        for row in ids.values():
            if row["kind"] == "jailbreak":
                original = ids.get(row.get("original_id"))
                if original is None or original["kind"] != "harmful" or original["split"] != row["split"]:
                    raise ValueError(f"Missing same-split harmful original for {row['id']}")
                if original["group_id"] != row["group_id"]:
                    raise ValueError("A jailbreak and its original must share one goal component")
        if not any(pair["split"] == "train" for pair in self.risk_pairs) or not any(
            pair["split"] == "train" for pair in self.manipulation_pairs
        ):
            raise ValueError("Both concept branches need explicit training pairs")

    def without_attacks(self, holdout: str | None) -> "Dataset":
        """Remove a complete attack family from all fitting/selection/calibration roles.

        Direct harmful goals stay available as task-risk supervision; a withheld
        attack prompt and its manipulation pair never contribute to the bank.
        Diagnostic and confirmation rows are never used to fit parameters.
        """
        if not holdout:
            return self
        family = canonical_family(holdout)
        known = {canonical_family(row.get("attack_family")) for row in self.records if row["kind"] == "jailbreak"}
        if family not in known:
            raise ValueError(f"Unknown held-out attack family: {holdout}; available: {sorted(known)}")
        splits = {name: [row for row in rows if name in ("diagnose", "confirm") or row["kind"] != "jailbreak" or
                         canonical_family(row.get("attack_family")) != family]
                  for name, rows in self.splits.items()}
        ids = {row["id"] for rows in splits.values() for row in rows}
        keep = lambda pairs: [pair for pair in pairs if pair["positive_id"] in ids and pair["negative_id"] in ids]
        result = Dataset(splits, keep(self.risk_pairs), keep(self.manipulation_pairs),
                         {**self.manifest, "outer_holdout": family})
        result.validate()
        return result


def load_dataset(root: str | Path, *, verify_hashes: bool = True) -> Dataset:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if verify_hashes:
        hashes = manifest.get("output_sha256", manifest.get("outputs_sha256", {}))
        for relative, expected in hashes.items():
            path = root / relative
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Manifest hash path escapes the dataset directory")
            if sha256_file(path) != expected:
                raise ValueError(f"Dataset fingerprint changed: {relative}")
    splits = {name: read_jsonl(root / "development" / f"{name}.jsonl") for name in SPLITS}
    if (root / "development" / "confirm.jsonl").exists():
        splits["confirm"] = read_jsonl(root / "development" / "confirm.jsonl")
    data = Dataset(splits, read_jsonl(root / "pairs" / "risk.jsonl"),
                   read_jsonl(root / "pairs" / "manipulation.jsonl"), manifest)
    data.validate()
    return data
