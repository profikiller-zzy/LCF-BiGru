"""Runner orchestration checks; fake CLI processes never load model weights."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

from tools.run_server_experiments import ExperimentRunner, FAMILIES, main


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeCLI:
    def __init__(self, failed_training=None, failed_evaluation=None, fail_extract=False):
        self.calls = []
        self.models = {}
        self.failed_training = dict(failed_training or {})
        self.failed_evaluation = dict(failed_evaluation or {})
        self.fail_extract = fail_extract

    def __call__(self, command, **kwargs):
        if "shell" in kwargs:
            raise AssertionError("Runner must use argv without shell=True")
        if kwargs["env"]["HF_HUB_OFFLINE"] != "1" or kwargs["env"]["TRANSFORMERS_OFFLINE"] != "1":
            raise AssertionError("Subprocess must remain offline")
        phase = command[3]
        values = command[4:]

        def argument(name):
            return values[values.index(name) + 1]

        family = argument("--holdout") if "--holdout" in values else "full"
        if phase == "evaluate":
            family = self.models[argument("--model")]
        self.calls.append((phase, family, command))
        kwargs["stdout"].write(f"{phase} stdout\n".encode())
        kwargs["stderr"].write(f"{phase} stderr\n".encode())
        if phase == "extract":
            if self.fail_extract:
                return subprocess.CompletedProcess(command, 8)
            write_json(Path(argument("--cache")) / "protocol.json", {"synthetic_cache": True})
        elif phase == "train":
            destination = Path(argument("--output"))
            if self.failed_training.get(family, 0):
                self.failed_training[family] -= 1
                destination.mkdir()
                (destination / "partial.txt").write_text("Failed attempt retained for audit")
                return subprocess.CompletedProcess(command, 7)
            destination.mkdir()
            parameters = b"Synthetic numerical artifact; no real weights"
            (destination / "parameters.npz").write_bytes(parameters)
            write_json(destination / "model.json", {"parameters_sha256": hashlib.sha256(parameters).hexdigest()})
            self.models[str(destination)] = family
        elif phase == "evaluate":
            if self.failed_evaluation.get(family, 0):
                self.failed_evaluation[family] -= 1
                return subprocess.CompletedProcess(command, 6)
            if argument("--split") != "diagnose":
                raise AssertionError("Frozen experiment must evaluate diagnostics only")
            result = {"split": "diagnose", "independent_confirmation": False,
                      "metrics": {"overall": {"n": 17, "tpr": .75, "fpr": .1}}}
            if family != "full":
                result["held_out_family_metrics"] = {"overall": {"n": 3, "tpr": 2 / 3}}
            write_json(Path(argument("--output")), result)
        else:
            raise AssertionError(f"Unexpected command: {phase}")
        return subprocess.CompletedProcess(command, 0)


class ServerRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.config = self.root / "config.json"
        self.data, self.cache, self.output = [self.root / name for name in ("data", "cache", "experiment")]
        write_json(self.config, {"backbone": {"model_name_or_path": "/models/existing", "local_files_only": True}})
        write_json(self.data / "manifest.json", {"output_sha256": {}})
        for split in ("train", "select", "calibrate", "diagnose"):
            path = self.data / "development" / f"{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")
        for branch in ("risk", "manipulation"):
            path = self.data / "pairs" / f"{branch}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n")

    def tearDown(self):
        self.temporary.cleanup()

    def runner(self, families=None, **kwargs):
        return ExperimentRunner(self.config, self.data, self.cache, self.output, families, **kwargs)

    def test_default_frozen_plan_extracts_once_then_runs_six_independent_models(self):
        child = FakeCLI()
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            result = self.runner().run()
        self.assertEqual(result["failed_runs"], [])
        self.assertEqual([phase for phase, _, _ in child.calls].count("extract"), 1)
        self.assertEqual([family for phase, family, _ in child.calls if phase == "train"], list(FAMILIES))
        self.assertEqual([phase for phase, _, _ in child.calls].count("evaluate"), 6)
        self.assertEqual(result["runs"]["full"]["metrics"]["overall"]["tpr"], .75)
        self.assertFalse(result["independent_confirmation"])
        checkpoint = json.loads((self.output / "checkpoint.json").read_text())
        for step in checkpoint["steps"].values():
            attempt = step["attempts"][0]
            self.assertEqual(attempt["exit_code"], 0)
            self.assertEqual(attempt["status"], "succeeded")
            self.assertIn("stdout", Path(attempt["stdout"]).read_text())
            self.assertIn("stderr", Path(attempt["stderr"]).read_text())
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertIn(str(self.config), plan["specification"]["input_sha256"])
        self.assertIn(str(self.data / "development/diagnose.jsonl"), plan["specification"]["input_sha256"])

    def test_failed_training_has_no_result_and_resume_retries_only_failed_steps(self):
        child = FakeCLI(failed_training={"autodan": 1})
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            first = self.runner(["full", "autodan"]).run()
            first_model = Path(first["runs"]["full"]["model"])
            original_bytes = (first_model / "parameters.npz").read_bytes()
            self.assertEqual(first["failed_runs"], ["autodan"])
            self.assertNotIn("metrics", first["runs"]["autodan"])
            self.assertNotIn("model", first["runs"]["autodan"])
            self.assertEqual(sum(phase == "evaluate" for phase, family, _ in child.calls if family == "autodan"), 0)
            call_count = len(child.calls)
            resumed = self.runner(["full", "autodan"]).run()
        self.assertEqual(resumed["failed_runs"], [])
        self.assertEqual([(phase, family) for phase, family, _ in child.calls[call_count:]],
                         [("train", "autodan"), ("evaluate", "autodan")])
        self.assertEqual(Path(resumed["runs"]["full"]["model"]), first_model)
        self.assertEqual((first_model / "parameters.npz").read_bytes(), original_bytes)
        attempts = self.output / "steps/autodan.train"
        self.assertTrue((attempts / "attempt-0001/model/partial.txt").exists())
        self.assertTrue((attempts / "attempt-0002/model/model.json").exists())

    def test_evaluation_retry_reuses_successful_model(self):
        child = FakeCLI(failed_evaluation={"full": 1})
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            first = self.runner(["full"]).run()
            self.assertEqual(first["failed_runs"], ["full"])
            self.assertIn("model", first["runs"]["full"])
            second = self.runner(["full"]).run()
        self.assertEqual(second["failed_runs"], [])
        self.assertEqual([phase for phase, _, _ in child.calls], ["extract", "train", "evaluate", "evaluate"])

    def test_extraction_failure_blocks_training_and_is_retryable(self):
        child = FakeCLI(fail_extract=True)
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            failed = self.runner(["full"]).run()
            self.assertEqual(failed["failed_runs"], ["full"])
            self.assertEqual(len(child.calls), 1)
            child.fail_extract = False
            succeeded = self.runner(["full"]).run()
        self.assertEqual(succeeded["failed_runs"], [])
        self.assertEqual([phase for phase, _, _ in child.calls], ["extract", "extract", "train", "evaluate"])

    def test_successful_artifact_or_frozen_input_tampering_is_rejected(self):
        child = FakeCLI()
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            result = self.runner(["full"]).run()
            parameters = Path(result["runs"]["full"]["model"]) / "parameters.npz"
            original = parameters.read_bytes()
            parameters.write_bytes(original + b"tamper")
            with self.assertRaisesRegex(ValueError, "Successful artifact changed"):
                self.runner(["full"]).run()
            parameters.write_bytes(original)
            self.config.write_text("{}")
            with self.assertRaisesRegex(ValueError, "Frozen plan/input fingerprints changed"):
                self.runner(["full"]).run()

    def test_skip_extract_fingerprints_existing_protocol_and_external_probe(self):
        write_json(self.cache / "protocol.json", {"existing": True})
        probe = self.root / "resource-probe.json"
        write_json(probe, {"available_gpu_gib": 24, "model_ready": True})
        child = FakeCLI()
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child):
            result = self.runner(["full"], skip_extract=True, probe_report=probe).run()
        self.assertEqual(result["failed_runs"], [])
        self.assertEqual([phase for phase, _, _ in child.calls], ["train", "evaluate"])
        plan = json.loads((self.output / "plan.json").read_text())
        self.assertIn(str(probe), plan["specification"]["input_sha256"])
        checkpoint = json.loads((self.output / "checkpoint.json").read_text())
        self.assertEqual(checkpoint["steps"]["extract"]["status"], "skipped")

    def test_successful_exit_without_model_is_not_a_successful_experiment(self):
        def empty_success(command, **kwargs):
            kwargs["stdout"].write(b"No artifact was actually produced\n")
            return subprocess.CompletedProcess(command, 0)

        write_json(self.cache / "protocol.json", {"existing": True})
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=empty_success):
            result = self.runner(["full"], skip_extract=True).run()
        self.assertEqual(result["failed_runs"], ["full"])
        self.assertNotIn("metrics", result["runs"]["full"])

    def test_interrupted_training_is_checkpointed_and_retried_in_new_directory(self):
        child = FakeCLI()
        interrupt = [True]

        def interrupt_once(command, **kwargs):
            if command[3] == "train" and interrupt[0]:
                interrupt[0] = False
                raise KeyboardInterrupt
            return child(command, **kwargs)

        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                self.runner(["full"]).run()
            checkpoint = json.loads((self.output / "checkpoint.json").read_text())
            attempt = checkpoint["steps"]["full.train"]["attempts"][0]
            self.assertEqual(attempt["status"], "interrupted")
            self.assertIsNone(attempt["exit_code"])
            resumed = self.runner(["full"]).run()
        self.assertEqual(resumed["failed_runs"], [])
        self.assertIn("attempt-0002", resumed["runs"]["full"]["model"])

    def test_inputs_changed_during_extraction_block_all_training(self):
        child = FakeCLI()

        def change_config(command, **kwargs):
            completed = child(command, **kwargs)
            if command[3] == "extract":
                self.config.write_text("{}")
            return completed

        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=change_config):
            result = self.runner(["full"]).run()
        self.assertEqual(result["failed_runs"], ["full"])
        self.assertEqual([phase for phase, _, _ in child.calls], ["extract"])
        self.assertIn("Frozen input changed", result["extraction"]["error"])

    def test_main_nonzero_exit_reports_failed_runs(self):
        child = FakeCLI(failed_training={"full": 1})
        arguments = ["--config", str(self.config), "--data", str(self.data), "--cache", str(self.cache),
                     "--output", str(self.output), "--families", "full"]
        with mock.patch("tools.run_server_experiments.subprocess.run", side_effect=child), \
             mock.patch("builtins.print"):
            self.assertEqual(main(arguments), 1)
            self.assertEqual(main(arguments), 0)


if __name__ == "__main__":
    unittest.main()
