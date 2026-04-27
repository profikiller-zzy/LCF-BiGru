from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class DecodeConfig:
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = False
    runtime_mode: str = "baseline"
    prompt_template: str = "alpaca"
    output_tag: str = "default"
    seed: int = 1234
    verbose_examples: bool = False
    topk_final: int = 20
    topk_band: int = 10
    band_spec: str = "0.35,0.55,0.75"
    calibration_fit_fraction: float = 0.4
    calibration_tune_fraction: float = 0.3
    calibration_max_examples: int = 96
    kgc_alpha_max: float = 0.8
    kgc_threshold_quantile: float = 0.95
    eos_token_ids: Optional[List[int]] = None
    pad_token_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationConfig:
    model_path: str
    model_name: str
    task_name: str
    attack_name: str
    sample_ratio: float = 1.0
    prompt_template: str = "alpaca"
    runtime_mode: str = "baseline"
    output_tag: str = "default"
    seed: int = 1234
    verbose_examples: bool = False
    calibration_dir: Optional[str] = None
    topk_final: int = 20
    topk_band: int = 10
    band_spec: str = "0.35,0.55,0.75"
    calibration_fit_fraction: float = 0.4
    calibration_tune_fraction: float = 0.3
    calibration_max_examples: int = 96
    kgc_alpha_max: float = 0.8
    kgc_threshold_quantile: float = 0.95
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.75
    top_k: int = 50
    num_beams: int = 1

    def to_decode_config(self) -> "DecodeConfig":
        return DecodeConfig(
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            num_beams=self.num_beams,
            do_sample=self.temperature > 0,
            runtime_mode=self.runtime_mode,
            prompt_template=self.prompt_template,
            output_tag=self.output_tag,
            seed=self.seed,
            verbose_examples=self.verbose_examples,
            topk_final=self.topk_final,
            topk_band=self.topk_band,
            band_spec=self.band_spec,
            calibration_fit_fraction=self.calibration_fit_fraction,
            calibration_tune_fraction=self.calibration_tune_fraction,
            calibration_max_examples=self.calibration_max_examples,
            kgc_alpha_max=self.kgc_alpha_max,
            kgc_threshold_quantile=self.kgc_threshold_quantile,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PreparedInputs:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int
    prompt_text: str
    eos_token_ids: List[int]
    pad_token_id: int


@dataclass
class StepState:
    step_index: int
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_length: int
    generated_ids: torch.Tensor
    logits: torch.Tensor
    hidden_states: Any
    past_key_values: Any
    eos_token_ids: List[int]
    pad_token_id: int
    attentions: Any = None


@dataclass
class StepDecision:
    logits: torch.Tensor
    telemetry: Dict[str, Any] = field(default_factory=dict)
    stop: bool = False
    abstain_message: str = ""


@dataclass
class EvaluationPaths:
    run_dir: str
    predictions_file: str
    metrics_file: str
    telemetry_file: str
    config_file: str
    summary_file: str


@dataclass
class CalibrationArtifact:
    artifact_id: str
    task_name: str
    model_name: str
    prompt_template: str
    band_indices: List[int]
    temperatures: List[float]
    fit_manifest_hash: str
    tune_manifest_hash: str
    heldout_manifest_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
