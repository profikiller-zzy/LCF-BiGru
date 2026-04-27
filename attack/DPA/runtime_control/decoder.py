import inspect
from typing import Dict, List, Tuple

import torch
from transformers.generation.logits_process import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from .types import DecodeConfig, PreparedInputs, StepDecision, StepState


def _supports_cache_position(model) -> bool:
    # Traverse the model hierarchy (PeftModel → base_model → model → ...)
    # to find the actual transformer model's forward signature.
    cur = model
    for _ in range(5):  # max depth to prevent infinite loops
        try:
            sig = inspect.signature(cur.forward)
        except (TypeError, ValueError):
            pass
        else:
            if "cache_position" in sig.parameters:
                return True
        # Try common wrapper attributes
        inner = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return False


def _get_model_config(model):
    """Traverse wrappers to find the actual model config."""
    cur = model
    for _ in range(5):
        cfg = getattr(cur, "config", None)
        if cfg is not None and hasattr(cfg, "num_hidden_layers"):
            return cfg
        inner = getattr(cur, "model", None) or getattr(cur, "base_model", None)
        if inner is None or inner is cur:
            break
        cur = inner
    return getattr(model, "config", None)


def _needs_hybrid_cache(model) -> bool:
    """Check if model requires a HybridCache (e.g. Gemma 2)."""
    config = _get_model_config(model)
    if config is None:
        return False
    return getattr(config, "cache_implementation", None) == "hybrid"


def _create_hybrid_cache(model, max_cache_len, batch_size=1):
    """Create a HybridCache for models that need it."""
    from transformers.cache_utils import HybridCache
    config = _get_model_config(model)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    return HybridCache(
        config=config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=dtype,
    )


def _prepare_step_inputs(model, next_token, attention_mask, past_key_values, use_cache_pos, cur_pos):
    model_inputs = {
        "input_ids": next_token,
        "past_key_values": past_key_values,
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if use_cache_pos:
        model_inputs["cache_position"] = torch.tensor(
            [cur_pos], device=next_token.device,
        )
    else:
        model_inputs["attention_mask"] = attention_mask
    return model_inputs


def _build_logits_warpers(decode_config: DecodeConfig) -> LogitsProcessorList:
    warpers = LogitsProcessorList()
    if not decode_config.do_sample:
        return warpers
    if decode_config.temperature and decode_config.temperature > 0 and decode_config.temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(decode_config.temperature))
    if decode_config.top_k and decode_config.top_k > 0:
        warpers.append(TopKLogitsWarper(top_k=decode_config.top_k))
    if decode_config.top_p and decode_config.top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p=decode_config.top_p))
    return warpers


def _select_next_token(
    step_logits: torch.Tensor,
    input_ids: torch.Tensor,
    decode_config: DecodeConfig,
    logits_warpers: LogitsProcessorList,
) -> torch.Tensor:
    if decode_config.num_beams != 1:
        raise ValueError("Only num_beams=1 is supported.")
    if not decode_config.do_sample:
        return torch.argmax(step_logits, dim=-1, keepdim=True)

    warped_logits = logits_warpers(input_ids, step_logits) if len(logits_warpers) > 0 else step_logits
    probs = torch.softmax(warped_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def decode_with_controller(model, tokenizer, prepared_inputs: PreparedInputs, decode_config: DecodeConfig, controller):
    input_ids = prepared_inputs.input_ids
    attention_mask = prepared_inputs.attention_mask
    eos_token_ids = prepared_inputs.eos_token_ids
    pad_token_id = prepared_inputs.pad_token_id

    use_cache_pos = _supports_cache_position(model)
    needs_hybrid = _needs_hybrid_cache(model)

    wants_attentions = getattr(controller, "needs_attentions", lambda: False)()

    prefill_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "use_cache": True,
        "output_hidden_states": True,
        "return_dict": True,
    }
    if wants_attentions:
        prefill_kwargs["output_attentions"] = True
    if use_cache_pos:
        prefill_kwargs["cache_position"] = torch.arange(
            0, input_ids.shape[-1], device=input_ids.device,
        )
    if needs_hybrid:
        max_cache_len = input_ids.shape[-1] + decode_config.max_new_tokens
        prefill_kwargs["past_key_values"] = _create_hybrid_cache(
            model, max_cache_len, batch_size=input_ids.size(0),
        )

    outputs = model(**prefill_kwargs)
    logits = outputs.logits[:, -1, :]
    hidden_states = outputs.hidden_states
    past_key_values = outputs.past_key_values
    prefill_attentions = getattr(outputs, "attentions", None) if wants_attentions else None

    generated_ids = torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device)
    sequence_rows: List[Dict] = []
    controller.begin_sequence(prepared_inputs)
    stop_reason = "max_new_tokens"
    abstain_message = ""
    logits_warpers = _build_logits_warpers(decode_config)
    cur_pos = input_ids.shape[-1]

    for step_index in range(decode_config.max_new_tokens):
        step_state = StepState(
            step_index=step_index,
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_length=prepared_inputs.prompt_length,
            generated_ids=generated_ids,
            logits=logits,
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            eos_token_ids=eos_token_ids,
            pad_token_id=pad_token_id,
            attentions=prefill_attentions if step_index == 0 else None,
        )
        decision: StepDecision = controller.modify_step(step_state)

        # Abstain: controller detected anomaly and wants to stop with a safe message
        if decision.abstain_message:
            abstain_message = decision.abstain_message
            step_row = {"step_index": step_index, **decision.telemetry}
            sequence_rows.append(step_row)
            stop_reason = "abstain"
            break

        step_logits = decision.logits
        next_token = _select_next_token(step_logits, input_ids, decode_config, logits_warpers)

        generated_ids = torch.cat([generated_ids, next_token], dim=-1)
        step_row = {"step_index": step_index, **decision.telemetry, "token_id": int(next_token[0, 0].item())}
        sequence_rows.append(step_row)
        if decision.stop:
            stop_reason = "controller_stop"
            break
        if int(next_token[0, 0].item()) in eos_token_ids:
            stop_reason = "eos_token"
            break

        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
            ],
            dim=-1,
        )
        model_inputs = _prepare_step_inputs(model, next_token, attention_mask, past_key_values, use_cache_pos, cur_pos)
        outputs = model(**model_inputs)
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states
        past_key_values = outputs.past_key_values
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        cur_pos += 1

    sequence_telemetry = controller.end_sequence()
    sequence_telemetry.setdefault("generated_token_ids", generated_ids[0].tolist())
    sequence_telemetry.setdefault("generated_length", int(generated_ids.size(1)))
    sequence_telemetry.setdefault("stop_reason", stop_reason)
    full_ids = torch.cat([prepared_inputs.input_ids, generated_ids], dim=-1)
    if abstain_message:
        decoded_text = abstain_message
    elif tokenizer is not None:
        decoded_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    else:
        decoded_text = ""
    return full_ids, decoded_text, sequence_rows, sequence_telemetry
