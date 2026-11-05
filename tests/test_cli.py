"""Exercise real files and public command parsing without downloading weights."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from dual_concept import cli
from dual_concept.backbone import FrozenBackboneConfig, StateCache
from dual_concept.pipeline import Detector
from test_pipeline import LAYERS, PROTOCOL, synthetic_dataset


class CommandTests(unittest.TestCase):
    def test_real_file_cli_validation_training_and_frozen_evaluation(self):
        data, states = synthetic_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, cache = root / "data", root / "cache"
            (dataset / "development").mkdir(parents=True)
            (dataset / "pairs").mkdir()
            cache.mkdir()
            (dataset / "manifest.json").write_text('{"output_sha256":{}}')
            for split, rows in data.splits.items():
                (dataset / "development" / f"{split}.jsonl").write_text(
                    "".join(json.dumps(row) + "\n" for row in rows))
            for branch, pairs in (("risk", data.risk_pairs), ("manipulation", data.manipulation_pairs)):
                (dataset / "pairs" / f"{branch}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in pairs))
            (cache / "protocol.json").write_text(json.dumps(PROTOCOL))
            rows_cache = StateCache(cache / "rows")
            for split, rows in data.splits.items():
                for row, state in zip(rows, states[split]):
                    rows_cache.save(row["id"], row["text"], PROTOCOL, state)
            config = root / "config.json"
            config.write_text(json.dumps({"backbone": {"candidate_layers": LAYERS}, "method": {}}))
            output = root / "frozen"
            commands = [
                ["validate", "--data", str(dataset)],
                ["train", "--data", str(dataset), "--cache", str(cache), "--config", str(config),
                 "--output", str(output), "--holdout", "gcg"],
                ["evaluate", "--data", str(dataset), "--cache", str(cache), "--model", str(output),
                 "--split", "diagnose", "--output", str(root / "evaluation.json")],
            ]
            results = []
            for command in commands:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cli.main(command), 0)
                results.append(json.loads(stdout.getvalue()))
            self.assertTrue(results[0]["valid"])
            self.assertEqual(results[1]["outer_holdout"], "gcg")
            self.assertEqual(results[2]["held_out_family_metrics"]["overall"]["tpr"], 1.0)
            self.assertFalse(results[2]["independent_confirmation"])
            self.assertEqual(json.loads((root / "evaluation.json").read_text())["threshold"],
                             Detector.load(output).threshold)

    def test_remote_prediction_reuses_frozen_commit_instead_of_mutable_branch(self):
        from argparse import Namespace
        from dataclasses import asdict
        expected = {"parameters": asdict(FrozenBackboneConfig()),
                    "checkpoint": {"kind": "hub_commit", "commit": "frozen-immutable-commit"}}
        model = mock.Mock()
        model.metadata = {"extraction_protocol": expected}
        model.bank.layer_ids = np.asarray([1, 7])
        model.threshold = 0.5
        model.decision_function.return_value = np.asarray([1.])
        backbone = mock.Mock()
        backbone.get_protocol.return_value = expected
        args = Namespace(model="unused", model_path=None, device=None, batch_size=1,
                         input=None, text="A new request", output=None)
        with mock.patch("dual_concept.cli.Detector.load", return_value=model), \
             mock.patch("dual_concept.cli.FrozenBackbone", return_value=backbone) as factory, \
             mock.patch("dual_concept.backbone.assert_compatible_protocol"):
            result = cli._predict(args)
        config = factory.call_args.args[0]
        self.assertEqual(config.revision, "frozen-immutable-commit")
        self.assertEqual(config.candidate_layers, (1, 7))
        self.assertTrue(result["predictions"][0]["risk"])


if __name__ == "__main__":
    unittest.main()
