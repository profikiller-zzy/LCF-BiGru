"""
Layerwise Convergence Fingerprint (LCF): Runtime backdoor detection via
representation velocity anomaly at layer N-3.

Uses rep_velocity (||h_L - h_{L-1}||) at the 3rd-to-last transformer layer
to detect backdoor-triggered anomalous generation. When anomaly is detected
at step 0 or step 1, abstains with a safe message.

Key insight: backdoor activation causes abnormal inter-layer representation
dynamics — the hidden state changes more rapidly between adjacent layers
as the model is steered toward the backdoor output. This "convergence
fingerprint" is architecture-agnostic and detectable from clean data alone.

Algorithm at generation steps 0 and 1:
  1. Extract hidden states at layers N-3 and N-4
  2. Compute rep_velocity = ||h_{N-3} - h_{N-4}||
  3. Compute z-score using per-step statistics: z = (rv - step_mean) / step_std
  4. If z > tau: abstain (output withheld)
  5. tau is auto-selected during calibration to achieve target FPR

Calibration uses per-step statistics (separate mean/std for step 0 and step 1)
because step-1 variance is ~3x step-0 variance. Pooling dilutes z-scores.
"""

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .projection import get_num_layers
from .types import DecodeConfig, PreparedInputs, StepDecision, StepState

# Calibration collects step 0 and step 1 rep_velocity separately
CALIBRATION_HORIZON = 2
# Target FPR for auto threshold selection (override via LCF_TARGET_FPR env)
TARGET_FPR = float(os.environ.get("LCF_TARGET_FPR", "0.10"))


@dataclass
class KGCCalibration:
    """Calibration artifact for LCF (rep_velocity at N-3)."""

    artifact_id: str
    layer_indices: List[int]   # kept for interface compat; single element [N-3]
    layer_index: int           # the N-3 layer
    n_calibration: int
    n_kl_samples: int          # kept for interface compat
    n_rv_samples: int
    clean_kl_median: float     # kept for interface compat; = clean_rv_median
    clean_kl_std: float        # kept for interface compat; = step0 std
    clean_kl_mean: float       # kept for interface compat; = step0 mean
    clean_rv_mean: float       # step-0 mean
    clean_rv_std: float        # step-0 std
    clean_rv_median: float     # step-0 median
    step1_rv_mean: float       # step-1 mean
    step1_rv_std: float        # step-1 std
    z_trigger: float           # auto-selected threshold
    target_fpr: float
    clean_kl_percentiles: Dict[str, float] = field(default_factory=dict)


def _compute_rep_velocity(
    hidden_states: Sequence[torch.Tensor],
    layer_index: int,
) -> float:
    """Compute ||h_L - h_{L-1}|| at the given layer index."""
    h_curr = hidden_states[layer_index + 1][:, -1, :].float()
    h_prev = hidden_states[layer_index][:, -1, :].float()
    return (h_curr - h_prev).norm().item()


def _supports_cache_position(model) -> bool:
    import inspect

    # Traverse the model hierarchy (PeftModel → base_model → model → ...)
    cur = model
    for _ in range(5):
        try:
            sig = inspect.signature(cur.forward)
        except (TypeError, ValueError):
            pass
        else:
            if "cache_position" in sig.parameters:
                return True
        inner = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return False


def build_kgc_calibration(
    model,
    tokenizer,
    clean_examples: List[dict],
    build_model_inputs_fn,
    decode_config: DecodeConfig,
) -> KGCCalibration:
    """Build LCF calibration by collecting per-step rep_velocity at N-k."""
    num_layers = get_num_layers(model)
    layer_offset = int(os.environ.get("LCF_LAYER_OFFSET", "3"))
    layer_index = num_layers - layer_offset  # N-k heuristic (default N-3)
    if layer_index < 1:
        raise ValueError(f"Model too shallow ({num_layers} layers) for N-3 monitoring.")

    from .decoder import _needs_hybrid_cache, _create_hybrid_cache

    max_examples = min(len(clean_examples), 50)
    examples = clean_examples[:max_examples]
    device = next(model.parameters()).device
    use_cache_pos = _supports_cache_position(model)
    needs_hybrid = _needs_hybrid_cache(model)

    step0_rvs = []  # step-0 rep_velocity per example
    step1_rvs = []  # step-1 rep_velocity per example
    per_example_max_z = []  # for threshold selection

    print(f"[LCF] Calibrating on {max_examples} clean examples, "
          f"layer=N-{layer_offset} (L{layer_index}), per-step calibration")

    model.eval()
    with torch.no_grad():
        for example in tqdm(examples, desc="LCF calibration"):
            instruction = example["instruction"]
            user_input = example.get("input", "")

            model_inputs, prompt_text = build_model_inputs_fn(
                tokenizer,
                instruction=instruction,
                user_input=user_input,
                prompt_template=decode_config.prompt_template,
            )
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            # Step 0: full prompt forward pass
            prefill_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if use_cache_pos:
                prefill_kwargs["cache_position"] = torch.arange(
                    0, input_ids.shape[-1], device=device,
                )
            if needs_hybrid:
                # Gemma 2 needs pre-initialized HybridCache; only 2 steps needed
                prefill_kwargs["past_key_values"] = _create_hybrid_cache(
                    model, max_cache_len=input_ids.shape[-1] + 2, batch_size=1,
                )
            outputs = model(**prefill_kwargs)
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values

            rv0 = _compute_rep_velocity(hidden_states, layer_index)
            step0_rvs.append(rv0)

            # Step 1: generate one token, then get step-1 hidden states
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            cur_pos = input_ids.shape[-1]

            step_inputs = {
                "input_ids": next_token,
                "past_key_values": past_key_values,
                "use_cache": True,
                "output_hidden_states": True,
                "return_dict": True,
            }
            if use_cache_pos:
                step_inputs["cache_position"] = torch.tensor(
                    [cur_pos], device=device,
                )
            else:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=-1,
                )
                step_inputs["attention_mask"] = attention_mask

            outputs = model(**step_inputs)
            hidden_states = outputs.hidden_states

            rv1 = _compute_rep_velocity(hidden_states, layer_index)
            step1_rvs.append(rv1)

    s0_arr = np.array(step0_rvs)
    s1_arr = np.array(step1_rvs)

    s0_mean = float(np.mean(s0_arr))
    s0_std = float(np.std(s0_arr))
    s1_mean = float(np.mean(s1_arr))
    s1_std = float(np.std(s1_arr))

    # Per-example max z-score (using per-step stats) for threshold selection
    for rv0, rv1 in zip(step0_rvs, step1_rvs):
        z0 = max(0.0, (rv0 - s0_mean) / max(s0_std, 1e-6))
        z1 = max(0.0, (rv1 - s1_mean) / max(s1_std, 1e-6))
        per_example_max_z.append(max(z0, z1))

    max_z_arr = np.array(per_example_max_z)
    z_trigger = float(np.percentile(max_z_arr, (1 - TARGET_FPR) * 100))
    z_trigger = max(z_trigger, 1.0)  # floor

    # Allow env var override
    z_override = os.environ.get("LCF_Z_TRIGGER")
    if z_override is not None:
        z_trigger = float(z_override)
        print(f"[LCF] Using override z_trigger={z_trigger} from LCF_Z_TRIGGER env var")

    data_hash = hashlib.sha256(
        np.concatenate([s0_arr, s1_arr]).tobytes()
    ).hexdigest()[:12]
    artifact_id = f"lcf_{data_hash}"

    calibration = KGCCalibration(
        artifact_id=artifact_id,
        layer_indices=[layer_index],
        layer_index=layer_index,
        n_calibration=max_examples,
        n_kl_samples=len(step0_rvs) + len(step1_rvs),
        n_rv_samples=len(step0_rvs) + len(step1_rvs),
        clean_kl_median=float(np.median(s0_arr)),
        clean_kl_std=s0_std,
        clean_kl_mean=s0_mean,
        clean_rv_mean=s0_mean,
        clean_rv_std=s0_std,
        clean_rv_median=float(np.median(s0_arr)),
        step1_rv_mean=s1_mean,
        step1_rv_std=s1_std,
        z_trigger=z_trigger,
        target_fpr=TARGET_FPR,
        clean_kl_percentiles={
            f"p{p}": float(np.percentile(s0_arr, p)) for p in [50, 75, 90, 95, 99]
        },
    )

    print(f"[LCF] Calibration complete (artifact_id={artifact_id}):")
    print(f"  Layer: N-{layer_offset} = L{layer_index} ({num_layers} total layers)")
    print(f"  Step 0: mean={s0_mean:.4f}, std={s0_std:.4f} ({len(step0_rvs)} samples)")
    print(f"  Step 1: mean={s1_mean:.4f}, std={s1_std:.4f} ({len(step1_rvs)} samples)")
    print(f"  z_trigger: {z_trigger:.4f} (target FPR={TARGET_FPR:.1%})")
    for k, v in calibration.clean_kl_percentiles.items():
        print(f"  {k}: {v:.4f}")

    return calibration


class KGCController:
    """
    LCF (Layerwise Convergence Fingerprint) controller.

    Detects backdoor activation via rep_velocity anomaly at layer N-3.
    Checks steps 0 and 1 using per-step calibration statistics.
    When z-score exceeds threshold at either step, abstains with a safe
    message.
    """

    def __init__(self, model, decode_config: DecodeConfig, calibration: KGCCalibration):
        self.model = model
        self.calibration = calibration

        self.z_trigger = calibration.z_trigger
        self.abstain_horizon = 2  # check steps 0 and 1

        # Sequence-level tracking
        self.anomaly_detected = False
        self.corrections_applied = 0
        self.total_steps = 0
        self.step_rvs: List[float] = []
        self.step_corrected: List[bool] = []

    def begin_sequence(self, prepared_inputs: PreparedInputs):
        self.corrections_applied = 0
        self.total_steps = 0
        self.step_rvs = []
        self.step_corrected = []
        self.anomaly_detected = False

    def modify_step(self, step_state: StepState) -> StepDecision:
        self.total_steps += 1
        final_logits = step_state.logits.float()
        hidden_states = step_state.hidden_states

        # 1. Compute rep_velocity at N-3
        rv = _compute_rep_velocity(hidden_states, self.calibration.layer_index)
        self.step_rvs.append(rv)

        # 2. z-score using per-step calibration statistics
        if step_state.step_index == 0:
            mean = self.calibration.clean_rv_mean
            std = self.calibration.clean_rv_std
        else:
            mean = self.calibration.step1_rv_mean
            std = self.calibration.step1_rv_std

        z = max(0.0, (rv - mean) / max(std, 1e-6))

        # 3. Only check within the abstain horizon (steps 0 and 1)
        should_abstain = (step_state.step_index < self.abstain_horizon) and (z > self.z_trigger)

        if should_abstain:
            self.anomaly_detected = True

        telemetry = {
            "runtime_mode": "kgc",
            "rep_velocity": round(rv, 6),
            "z_score": round(z, 4),
            "threshold": round(self.z_trigger, 4),
            "anomaly_detected": self.anomaly_detected,
            "veto_applied": False,
        }

        if not should_abstain:
            self.step_corrected.append(False)
            return StepDecision(logits=final_logits, telemetry=telemetry)

        # --- LCF Abstain ---
        self.corrections_applied += 1
        self.step_corrected.append(True)
        telemetry["veto_applied"] = True

        return StepDecision(
            logits=final_logits,
            telemetry=telemetry,
            abstain_message="[Output withheld by safety filter.]",
        )

    def end_sequence(self) -> Dict[str, Any]:
        n_corrected = sum(self.step_corrected)
        return {
            "corrections_applied": self.corrections_applied,
            "total_steps": self.total_steps,
            "correction_rate": round(n_corrected / max(self.total_steps, 1), 4),
            "mean_kl": round(float(np.mean(self.step_rvs)) if self.step_rvs else 0.0, 6),
            "max_kl": round(float(np.max(self.step_rvs)) if self.step_rvs else 0.0, 6),
            "abstain_rate": round(self.corrections_applied / max(self.total_steps, 1), 4),
            "sequence_risk": round(float(np.mean(self.step_rvs)) if self.step_rvs else 0.0, 6),
            "max_step_risk": round(float(np.max(self.step_rvs)) if self.step_rvs else 0.0, 6),
            "anomaly_detected": self.anomaly_detected,
            "overflow_count": 0,
        }
