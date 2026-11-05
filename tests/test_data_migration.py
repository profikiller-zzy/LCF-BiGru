"""Research-integrity checks for the standalone historical data migration."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("migrate_legacy_data", ROOT / "tools/migrate_legacy_data.py")
MIGRATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MIGRATION)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class MigrationTests(unittest.TestCase):
    def create_source(self, directory):
        root = Path(directory) / MIGRATION.LEGACY_NAME
        root.mkdir()
        rows = []
        for id_, kind, label, stage, group in [
            ("safe", "natural_pair", 0, "train", "g1"),
            ("harmful", "natural_pair", 1, "train", "g1"),
            ("attack", "public_attack", 1, "train", "g1"),
            ("select", "public_attack", 1, "select", "g2"),
            ("cal_safe", "source_hard_safe", 0, "calibrate", "g3"),
            ("cal_attack", "public_attack", 1, "calibrate", "g3"),
            ("diagnose", "source_ordinary_safe", 0, "diagnose", "g4"),
        ]:
            rows.append({
                "id": id_, "instruction": f"record {id_}", "kind": kind, "label": label,
                "stage": stage, "cluster_id": group, "independent_human_validated": False,
                "label_status": "source_or_construction_candidate",
                "provenance": {
                    "source_dataset": "fixture", "source_method": "pair",
                    "original_goal": "record harmful" if id_ == "attack" else f"original {id_}",
                },
                "target_success": None,
            })
        files = {
            "panel_PRIVATE.jsonl": rows,
            "risk_pairs.jsonl": [{"harmful_id": "harmful", "safe_id": "safe", "pair_id": "r1"}],
            "manipulation_pairs.jsonl": [{
                "attack_id": "attack", "original_id": "harmful", "source_metadata_match": True,
                "semantic_preservation_independently_validated": False,
            }],
        }
        for name, records in files.items():
            (root / name).write_text(MIGRATION._encode_jsonl(records), encoding="utf-8")
        for name in ["panel_summary.json", "protocol.json", "original_goal_ids.json"]:
            (root / name).write_text("{}\n", encoding="utf-8")
        (root / "preparation_protocol.json").write_text(json.dumps({
            "artifacts_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}
        }), encoding="utf-8")
        return root

    def test_migration_preserves_pairing_partitions_and_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.create_source(temporary)
            output = Path(temporary) / "result"
            manifest = MIGRATION.migrate(source, output)
            train = {row["id"]: row for row in read_jsonl(output / "development/train.jsonl")}
            self.assertEqual(train["attack"]["original_id"], "harmful")
            select = read_jsonl(output / "development/select.jsonl")
            self.assertEqual(select[0]["original_id"], select[1]["id"])
            self.assertEqual(select[1]["source"]["provenance"]["source_dataset"], "metadata-derived-original")
            self.assertEqual(train["attack"]["source"]["legacy_id"], "attack")
            self.assertEqual(train["attack"]["source"]["legacy_metadata"]["target_success"], None)
            self.assertEqual(train["attack"]["exposure"], "development_exposed")
            self.assertEqual(train["attack"]["label_status"], "candidate_not_independently_validated")
            self.assertEqual([row["label"] for row in read_jsonl(output / "development/calibrate.jsonl")], [0])
            self.assertEqual(manifest["excluded"]["historical_calibrate_positive"]["ids"], ["cal_attack"])
            risk = read_jsonl(output / "pairs/risk.jsonl")[0]
            self.assertEqual((risk["positive_id"], risk["negative_id"]), ("harmful", "safe"))
            self.assertIn("not_fresh_confirmation", manifest["scope"])
            self.assertFalse((output / "development/test.jsonl").exists())

    def test_outputs_are_identical_when_source_is_copied_to_a_different_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.create_source(temporary)
            first, second = Path(temporary) / "one", Path(temporary) / "two"
            MIGRATION.migrate(source, first)
            copied = Path(temporary) / "different_source" / MIGRATION.LEGACY_NAME
            shutil.copytree(source, copied)
            MIGRATION.migrate(copied, second)
            for path in first.rglob("*"):
                if path.is_file():
                    self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())

    def test_source_tampering_is_rejected_before_output_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.create_source(temporary)
            with (source / "panel_PRIVATE.jsonl").open("a", encoding="utf-8") as stream:
                stream.write("{}\n")
            output = Path(temporary) / "result"
            with self.assertRaisesRegex(ValueError, "integrity"):
                MIGRATION.migrate(source, output)
            self.assertFalse(output.exists())

    def test_missing_pair_endpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.create_source(temporary)
            (source / "risk_pairs.jsonl").write_text('{"harmful_id":"absent","safe_id":"safe"}\n', encoding="utf-8")
            (source / "preparation_protocol.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "endpoint missing"):
                MIGRATION.migrate(source, Path(temporary) / "result")

    def test_exact_original_text_crossing_partitions_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = self.create_source(temporary)
            rows = read_jsonl(source / "panel_PRIVATE.jsonl")
            next(row for row in rows if row["id"] == "select")["provenance"]["original_goal"] = "record harmful"
            (source / "panel_PRIVATE.jsonl").write_text(MIGRATION._encode_jsonl(rows), encoding="utf-8")
            (source / "preparation_protocol.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "crosses partitions"):
                MIGRATION.migrate(source, Path(temporary) / "result")

    def test_shipped_dataset_hashes_and_pair_endpoints(self):
        data = ROOT / "data"
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((data / name).read_bytes()).hexdigest(), expected)
        rows = [row for split in MIGRATION.SPLITS for row in read_jsonl(data / "development" / f"{split}.jsonl")]
        self.assertEqual(len(rows), 1665)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(len(by_id), len(rows))
        for family, kinds in [("risk", ("harmful", "safe")), ("manipulation", ("jailbreak", "harmful"))]:
            pairs = read_jsonl(data / "pairs" / f"{family}.jsonl")
            self.assertEqual(len(pairs), 200 if family == "risk" else 260)
            for pair in pairs:
                endpoints = [by_id[pair[key]] for key in ["positive_id", "negative_id"]]
                self.assertEqual(tuple(row["kind"] for row in endpoints), kinds)
                self.assertTrue(all(row["split"] == pair["split"] for row in endpoints))
                if family == "risk":
                    self.assertEqual(pair["split"], "train")
        self.assertTrue(all(row["original_id"] in by_id for row in rows if row["kind"] == "jailbreak"))
        self.assertTrue(all(row["kind"] == "safe" for row in rows if row["split"] == "calibrate"))
        seen_groups = {}
        for row in rows:
            if row["group_id"] in seen_groups:
                self.assertEqual(seen_groups[row["group_id"]], row["split"])
            seen_groups[row["group_id"]] = row["split"]


if __name__ == "__main__":
    unittest.main()
