from .kgc import KGCController
from .types import DecodeConfig, PreparedInputs, StepDecision, StepState


class RuntimeController:
    def begin_sequence(self, prepared_inputs: PreparedInputs):
        return None

    def modify_step(self, step_state: StepState) -> StepDecision:
        return StepDecision(logits=step_state.logits)

    def end_sequence(self):
        return {}


class BaselineController(RuntimeController):
    def modify_step(self, step_state: StepState) -> StepDecision:
        return StepDecision(logits=step_state.logits, telemetry={"runtime_mode": "baseline"})


def build_controller(model, decode_config: DecodeConfig, calibration_artifact=None):
    mode = decode_config.runtime_mode
    if mode == "baseline":
        return BaselineController()
    if mode == "kgc":
        if calibration_artifact is None:
            raise ValueError("KGC mode requires a calibration artifact.")
        return KGCController(model, decode_config, calibration_artifact)
    if mode == "lcf":
        if calibration_artifact is None:
            raise ValueError("LCF mode requires a calibration artifact.")
        from .multilayer import MultiLayerController
        return MultiLayerController(model, decode_config, calibration_artifact)
    raise ValueError(f"Unknown runtime mode: {mode}")
