import json
import os

from .types import CalibrationArtifact


def load_calibration_artifact(calibration_dir: str, artifact_id: str) -> CalibrationArtifact:
    path = os.path.join(calibration_dir, artifact_id, "artifact.json")
    with open(path) as f:
        payload = json.load(f)
    return CalibrationArtifact(**payload)


__all__ = ["load_calibration_artifact"]
