#!/usr/bin/env python3
"""Resume the fixed second-study experiment plan using the existing CLI.

Provision the interpreter, model weights and resources outside this runner. It
never downloads a model, probes accounts/credentials, tunes on diagnostics, or
shuts down a machine. Successful immutable artifacts are reused, while failed
attempts keep their logs and are retried in a fresh directory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


FAMILIES = ("full", "autodan", "drattack", "gcg", "pair_tap", "saa")
REPOSITORY = Path(__file__).resolve().parents[1]
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _input_files(config: Path, data: Path, probe: Path | None) -> list[Path]:
    manifest = _read(data / "manifest.json")
    files = {config, data / "manifest.json", data / "pairs/risk.jsonl", data / "pairs/manipulation.jsonl"}
    files.update(data / "development" / f"{split}.jsonl" for split in ("train", "select", "calibrate", "diagnose"))
    for relative in manifest.get("output_sha256", manifest.get("outputs_sha256", {})):
        path = (data / relative).resolve()
        if not path.is_relative_to(data):
            raise ValueError("Manifest fingerprint path escapes the data directory")
        files.add(path)
    files.update((REPOSITORY / "dual_concept").glob("*.py"))
    files.add(Path(__file__).resolve())
    if probe is not None:
        _read(probe)  # Accept an external JSON report, never execute a probe.
        files.add(probe)
    return sorted(files)


class ExperimentRunner:
    def __init__(self, config: Path, data: Path, cache: Path, output: Path,
                 families: list[str] | None = None, *, skip_extract: bool = False,
                 probe_report: Path | None = None):
        self.config, self.data, self.cache, self.output = [Path(value).resolve()
                                                        for value in (config, data, cache, output)]
        self.families = list(FAMILIES if families is None else families)
        if not self.families or len(set(self.families)) != len(self.families) or set(self.families) - set(FAMILIES):
            raise ValueError(f"families must be unique values from {FAMILIES}")
        self.skip_extract = skip_extract
        self.probe_report = Path(probe_report).resolve() if probe_report is not None else None
        self.files = _input_files(self.config, self.data, self.probe_report)
        self.fingerprints = {str(path): _sha(path) for path in self.files}
        self.specification = {
            "schema_version": 1, "families": self.families,
            "config": str(self.config), "data": str(self.data), "cache": str(self.cache),
            "output": str(self.output), "python": sys.executable, "python_version": sys.version,
            "cwd": str(REPOSITORY), "input_sha256": self.fingerprints,
            "child_environment_overrides": OFFLINE_ENV, "evaluation_split": "diagnose",
            "policy": "extract once; train independently per outer holdout; frozen diagnose evaluation; no tuning",
            "external_probe_report": str(self.probe_report) if self.probe_report else None,
        }
        self.plan_hash = hashlib.sha256(_json(self.specification)).hexdigest()
        self.checkpoint: dict[str, Any] = {}

    def _flush(self) -> None:
        self.checkpoint["updated_at"] = _now()
        _atomic_json(self.output / "checkpoint.json", self.checkpoint)

    def _assert_inputs_unchanged(self) -> None:
        for filename, expected in self.fingerprints.items():
            if _sha(Path(filename)) != expected:
                raise ValueError(f"Frozen input changed during execution: {filename}")

    def _initialize(self) -> None:
        plan_path, checkpoint_path = self.output / "plan.json", self.output / "checkpoint.json"
        if plan_path.exists():
            existing = _read(plan_path)
            if existing.get("specification") != self.specification or existing.get("plan_sha256") != self.plan_hash:
                raise ValueError("Frozen plan/input fingerprints changed; choose a new output directory")
            if not checkpoint_path.exists():
                raise ValueError("Existing plan has no checkpoint; inspect the incomplete output directory")
            self.checkpoint = _read(checkpoint_path)
            if self.checkpoint.get("plan_sha256") != self.plan_hash:
                raise ValueError("Checkpoint belongs to a different frozen plan")
        else:
            if any(path.name != "runner.lock" for path in self.output.iterdir()):
                raise ValueError("Output is nonempty without a runner plan; choose a new output directory")
            _atomic_json(plan_path, {"created_at": _now(), "plan_sha256": self.plan_hash,
                                     "specification": self.specification})
            self.checkpoint = {"schema_version": 1, "plan_sha256": self.plan_hash, "steps": {}}
            self._flush()

    def _completed(self, key: str) -> dict[str, Any] | None:
        step = self.checkpoint["steps"].get(key)
        if step is None or step.get("status") not in ("succeeded", "skipped"):
            return None
        for filename, expected in step["artifact_sha256"].items():
            path = Path(filename)
            if not path.is_file() or _sha(path) != expected:
                raise ValueError(f"Successful artifact changed or disappeared: {filename}")
        return step

    def _command(self, command: str, *arguments: str) -> list[str]:
        return [sys.executable, "-m", "dual_concept", command, *arguments]

    def _execute(self, key: str, command_builder, artifact_kind: str) -> dict[str, Any]:
        completed = self._completed(key)
        if completed is not None:
            return completed
        self._assert_inputs_unchanged()
        old = self.checkpoint["steps"].get(key, {})
        attempts = list(old.get("attempts", []))
        number = len(attempts) + 1
        directory = self.output / "steps" / key / f"attempt-{number:04d}"
        # Never overwrite an orphan attempt left by interruption before the
        # checkpoint write; retain it and allocate the next available number.
        while directory.exists():
            number += 1
            directory = directory.parent / f"attempt-{number:04d}"
        directory.mkdir(parents=True)
        artifact = (self.cache / "protocol.json" if artifact_kind == "extract" else
                    directory / ("model" if artifact_kind == "train" else "evaluation.json"))
        command = command_builder(artifact)
        attempt = {"number": number, "started_at": _now(), "status": "running", "command": command,
                   "cwd": str(REPOSITORY), "stdout": str(directory / "stdout.log"),
                   "stderr": str(directory / "stderr.log"), "artifact": str(artifact),
                   "input_fingerprint_sha256": self.plan_hash}
        attempts.append(attempt)
        step = {"status": "running", "attempts": attempts, "artifact": str(artifact)}
        self.checkpoint["steps"][key] = step
        self._flush()
        print(f"[{_now()}] {key} attempt {number}", flush=True)
        try:
            with Path(attempt["stdout"]).open("xb") as stdout, Path(attempt["stderr"]).open("xb") as stderr:
                completed_process = subprocess.run(command, cwd=REPOSITORY, stdout=stdout, stderr=stderr,
                                                   env={**os.environ, **OFFLINE_ENV}, check=False)
            attempt["exit_code"] = completed_process.returncode
            self._assert_inputs_unchanged()
            if completed_process.returncode:
                raise RuntimeError(f"CLI exited with status {completed_process.returncode}; see {attempt['stderr']}")
            if artifact_kind == "train":
                metadata = _read(artifact / "model.json")
                parameters = artifact / "parameters.npz"
                if metadata.get("parameters_sha256") != _sha(parameters):
                    raise ValueError("Training returned an incomplete or inconsistent model")
                paths = [artifact / "model.json", parameters]
            else:
                result = _read(artifact)
                if artifact_kind == "evaluate" and (result.get("split") != "diagnose" or
                                                     result.get("independent_confirmation") is not False or
                                                     not isinstance(result.get("metrics"), dict)):
                    raise ValueError("Evaluation did not produce frozen diagnostic results")
                paths = [artifact]
            step["artifact_sha256"] = {str(path): _sha(path) for path in paths}
            step["status"] = attempt["status"] = "succeeded"
        except KeyboardInterrupt:
            step["status"] = attempt["status"] = "interrupted"
            attempt["error"] = "Interrupted; a later invocation retries in a fresh attempt directory"
            raise
        except (OSError, ValueError, RuntimeError) as error:
            step["status"] = attempt["status"] = "failed"
            attempt["error"] = str(error)
            attempt.setdefault("exit_code", None)
        finally:
            attempt.setdefault("exit_code", None)
            attempt["finished_at"] = _now()
            _atomic_json(directory / "step.json", attempt)
            self._flush()
        return step

    def _summary(self) -> dict[str, Any]:
        runs = {}
        for family in self.families:
            train = self.checkpoint["steps"].get(f"{family}.train", {})
            evaluation = self.checkpoint["steps"].get(f"{family}.evaluate", {})
            success = train.get("status") == "succeeded" and evaluation.get("status") == "succeeded"
            item = {"status": "succeeded" if success else "failed",
                    "train_status": train.get("status", "not_run"),
                    "evaluate_status": evaluation.get("status", "not_run")}
            if train.get("status") == "succeeded":
                item["model"] = train["artifact"]
            if success:
                item["evaluation"] = evaluation["artifact"]
                result = _read(Path(evaluation["artifact"]))
                item["metrics"] = result["metrics"]
                if "held_out_family_metrics" in result:
                    item["held_out_family_metrics"] = result["held_out_family_metrics"]
            else:
                failures = [step["attempts"][-1].get("error") for step in (train, evaluation)
                            if step.get("attempts") and step.get("status") != "succeeded"]
                item["errors"] = failures or ["Blocked before training; inspect extraction/checkpoint logs"]
            runs[family] = item
        extraction = self.checkpoint["steps"].get("extract", {})
        extraction_summary = {"status": extraction.get("status", "not_run")}
        if extraction.get("attempts") and extraction.get("status") != "succeeded":
            extraction_summary["error"] = extraction["attempts"][-1].get("error")
        summary = {"updated_at": _now(), "plan_sha256": self.plan_hash, "evaluation_split": "diagnose",
                   "extraction": extraction_summary,
                   "independent_confirmation": False, "runs": runs,
                   "failed_runs": [family for family, run in runs.items() if run["status"] != "succeeded"]}
        _atomic_json(self.output / "summary.json", summary)
        return summary

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "runner.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another runner holds this output directory") from error
            self._initialize()
            if self._completed("extract") is None:
                if self.skip_extract:
                    protocol = self.cache / "protocol.json"
                    _read(protocol)
                    self.checkpoint["steps"]["extract"] = {
                        "status": "skipped", "reason": "Caller supplied an existing cache",
                        "artifact": str(protocol), "artifact_sha256": {str(protocol): _sha(protocol)}, "attempts": []}
                    self._flush()
                else:
                    extract = self._execute("extract", lambda _: self._command(
                        "extract", "--config", str(self.config), "--data", str(self.data), "--cache", str(self.cache),
                        "--splits", "train", "select", "calibrate", "diagnose"), "extract")
                    if extract["status"] != "succeeded":
                        return self._summary()
            for family in self.families:
                train = self._execute(f"{family}.train", lambda artifact, family=family: self._command(
                    "train", "--config", str(self.config), "--data", str(self.data), "--cache", str(self.cache),
                    "--output", str(artifact), *([] if family == "full" else ["--holdout", family])), "train")
                if train["status"] != "succeeded":
                    continue
                self._execute(f"{family}.evaluate", lambda artifact, train=train: self._command(
                    "evaluate", "--data", str(self.data), "--cache", str(self.cache),
                    "--model", train["artifact"], "--split", "diagnose", "--output", str(artifact)), "evaluate")
            return self._summary()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "data", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--skip-extract", action="store_true", help="Use a previously complete cache")
    parser.add_argument("--probe-report", type=Path, help="Fingerprint an external resource-preflight JSON report")
    args = parser.parse_args(argv)
    try:
        result = ExperimentRunner(args.config, args.data, args.cache, args.output, args.families,
                                  skip_extract=args.skip_extract, probe_report=args.probe_report).run()
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Runner error: {error}", file=sys.stderr)
        return 2
    print(_json(result).decode(), end="")
    return 1 if result["failed_runs"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
