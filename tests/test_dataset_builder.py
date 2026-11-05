"""Tests for deterministic, review-only construction of the JBB pilot."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("build_jbb_pilot", ROOT / "tools/build_jbb_pilot.py")
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def encode_csv(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["Index", "Goal", "Target", "Behavior", "Category", "Source"])
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class JBBPilotBuilderTests(unittest.TestCase):
    def fixture_payloads(self):
        harmful, benign = [], []
        pair_rows, gcg_rows = [], []
        for index in range(100):
            category = f"category_{index % 5}"
            behavior = f"behavior_{index:03d}"
            harmful_goal = f"harmful request {index:03d}"
            harmful.append({"Index": index, "Goal": harmful_goal, "Target": "omitted", "Behavior": behavior,
                            "Category": category, "Source": "fixture"})
            benign.append({"Index": index, "Goal": f"ordinary request {index:03d}", "Target": "omitted",
                           "Behavior": f"benign_{index:03d}", "Category": category, "Source": "fixture"})
            pair_rows.append({"index": index, "goal": harmful_goal, "behavior": behavior, "category": category,
                              "prompt": "" if index == 0 else f"published PAIR candidate {index:03d}",
                              "response": "response_must_not_be_exported", "jailbroken": index % 2 == 0})
            gcg_rows.append({"index": index, "goal": "different goal" if index == 1 else harmful_goal,
                             "behavior": behavior, "category": category,
                             "prompt": f"published GCG candidate {index:03d}",
                             "response": "response_must_not_be_exported", "jailbroken": index % 2 == 1})
        values = {
            "harmful_behaviors": encode_csv(harmful),
            "benign_behaviors": encode_csv(benign),
            "jbb_dataset_license": b"MIT License\nfixture\n",
            "pair_artifact": json.dumps({"jailbreaks": pair_rows}).encode(),
            "gcg_artifact": json.dumps({"jailbreaks": gcg_rows}).encode(),
            "artifacts_license": b"MIT License\nfixture\n",
        }
        return values

    def run_fixture(self):
        payloads = self.fixture_payloads()
        specs = {
            name: {"url": f"https://example.invalid/{name}", "sha256": hashlib.sha256(payload).hexdigest(), "license": "MIT",
                   **({"method": "PAIR", "attack_type": "black_box", "source_model": "fixture"} if name == "pair_artifact" else {}),
                   **({"method": "GCG", "attack_type": "white_box", "source_model": "fixture"} if name == "gcg_artifact" else {})}
            for name, payload in payloads.items()
        }
        with patch.object(BUILDER, "SOURCES", specs):
            return payloads, BUILDER.build_payloads(payloads)

    def test_pairing_and_labels_are_review_only_and_source_outcome_independent(self):
        _, outputs = self.run_fixture()
        manifest = json.loads(outputs["manifest.json"])
        records = [json.loads(line) for line in outputs["records.jsonl"].decode().splitlines()]
        risk = [json.loads(line) for line in outputs["pairs/risk_candidates.jsonl"].decode().splitlines()]
        manipulation = [json.loads(line) for line in outputs["pairs/manipulation_candidates.jsonl"].decode().splitlines()]
        unpaired = [json.loads(line) for line in outputs["review/external_unpaired_jailbreaks.jsonl"].decode().splitlines()]
        excluded = [json.loads(line) for line in outputs["review/excluded_artifact_rows.jsonl"].decode().splitlines()]
        self.assertEqual(manifest["scope"], "pilot_candidate_material_only_not_training_ready")
        self.assertEqual(manifest["counts"]["assigned_splits"], 0)
        self.assertEqual(manifest["counts"]["risk_pair_candidates"], 300)
        self.assertEqual(manifest["counts"]["manipulation_pair_candidates"], 198)
        self.assertEqual(len(unpaired), 1)
        self.assertEqual(len(excluded), 1)
        self.assertTrue(all(pair["goal_cluster_id"] is None for pair in risk))
        self.assertTrue(all(pair["match_status"].startswith("heuristic_candidate") for pair in risk))
        self.assertTrue(all(pair["goal_preserved"] == "uncertain" for pair in manipulation))
        self.assertTrue(all(row["split"] is None for row in records))
        self.assertTrue(all(row["kind"] in {"safe", "harmful", "jailbreak"} for row in records))
        self.assertTrue(all(row["risk_label"] == row["label"] for row in records))
        self.assertTrue(all(row["risk_label"] == 1 for row in records if row["core_group"] == "jailbreak"))
        self.assertTrue(all(row["attack_success"] in {"success", "failure", "unknown"}
                            for row in records if row["core_group"] == "jailbreak"))
        self.assertNotIn(b"response_must_not_be_exported", b"\n".join(outputs.values()))

    def test_outputs_are_deterministic_and_hash_indexed(self):
        _, first = self.run_fixture()
        _, second = self.run_fixture()
        self.assertEqual(first, second)
        manifest = json.loads(first["manifest.json"])
        for path, expected in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256(first[path]).hexdigest(), expected)

    def test_hash_mismatch_is_rejected(self):
        payloads = self.fixture_payloads()
        payloads["harmful_behaviors"] += b"tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            BUILDER.build_payloads(payloads)

    def test_shipped_pilot_hashes_and_links(self):
        root = ROOT / "data/dccf_v1_jbb_pilot"
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        for relative, expected in manifest["output_sha256"].items():
            self.assertEqual(hashlib.sha256((root / relative).read_bytes()).hexdigest(), expected)
        records = [json.loads(line) for line in (root / "records.jsonl").read_text(encoding="utf-8").splitlines()]
        by_id = {row["record_id"]: row for row in records}
        pairs = [json.loads(line) for line in (root / "pairs/manipulation_candidates.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(manifest["counts"]["goal_clusters"], 100)
        self.assertEqual(manifest["counts"]["records"], len(records))
        self.assertEqual(manifest["counts"]["risk_pair_candidates"], 300)
        self.assertEqual(manifest["counts"]["manipulation_pair_candidates"], len(pairs))
        self.assertEqual(manifest["counts"]["human_reviewed_records"], 0)
        self.assertEqual(manifest["counts"]["assigned_splits"], 0)
        self.assertEqual(manifest["counts"]["unpaired_jailbreaks_pending_mapping_review"], 10)
        self.assertEqual(manifest["counts"]["artifact_rows_without_prompt"], 18)
        self.assertTrue(all(row["original_id"] == next(pair["negative_id"] for pair in pairs if pair["positive_id"] == row["record_id"])
                            for row in records if row["core_group"] == "jailbreak"))
        self.assertTrue(all(by_id[pair["positive_id"]]["core_group"] == "jailbreak" and
                            by_id[pair["negative_id"]]["core_group"] == "harmful" for pair in pairs))


if __name__ == "__main__":
    unittest.main()
