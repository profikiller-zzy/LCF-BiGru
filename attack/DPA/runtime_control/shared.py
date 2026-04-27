import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .decoder import _prepare_step_inputs
from .projection import band_to_logits, get_band_hidden, get_final_hidden, stable_example_id


def softmax_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    return torch.softmax(logits.float() / temperature, dim=-1)


def log_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits.float(), dim=-1)


def fit_temperature_grid(band_logits_list: List[torch.Tensor], final_token_ids: List[int]) -> float:
    if not band_logits_list or not final_token_ids:
        return 1.0

    candidates = np.linspace(0.5, 3.0, num=11)
    best_temp = 1.0
    best_loss = float("inf")
    for temp in candidates:
        loss = 0.0
        for logits, token_id in zip(band_logits_list, final_token_ids):
            step_log_probs = torch.log_softmax(logits / temp, dim=-1)
            loss += float(-step_log_probs[0, token_id].item())
        if loss < best_loss:
            best_loss = loss
            best_temp = float(temp)

    return best_temp


def mix_with_kl_budget(final_logits: torch.Tensor, adjusted_logits: torch.Tensor, beta_max: float) -> torch.Tensor:
    final_probs = torch.softmax(final_logits.float(), dim=-1)
    adjusted_probs = torch.softmax(adjusted_logits.float(), dim=-1)

    def kl_for_eta(eta: float) -> float:
        mixed = (1.0 - eta) * final_probs + eta * adjusted_probs
        kl = torch.sum(mixed * (torch.log(mixed + 1e-12) - torch.log(final_probs + 1e-12)), dim=-1)
        return float(kl.item())

    low, high = 0.0, 1.0
    for _ in range(20):
        mid = (low + high) / 2.0
        if kl_for_eta(mid) <= beta_max:
            low = mid
        else:
            high = mid

    eta = low
    mixed = (1.0 - eta) * final_probs + eta * adjusted_probs
    return torch.log(mixed + 1e-12)


def split_examples(examples: List[dict], cfg) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]], List[Tuple[str, dict]]]:
    indexed = [(stable_example_id(example, idx), example) for idx, example in enumerate(examples)]
    rng = random.Random(cfg.seed)
    rng.shuffle(indexed)
    if cfg.calibration_max_examples > 0:
        indexed = indexed[: min(cfg.calibration_max_examples, len(indexed))]

    n_total = len(indexed)
    n_fit = max(1, int(n_total * cfg.calibration_fit_fraction))
    n_tune = max(1, int(n_total * cfg.calibration_tune_fraction))
    n_tune = min(n_tune, max(1, n_total - n_fit))

    fit = indexed[:n_fit]
    tune = indexed[n_fit : n_fit + n_tune]
    heldout = indexed[n_fit + n_tune :]
    if not heldout:
        heldout = tune[-1:]
        tune = tune[:-1] if len(tune) > 1 else tune

    return fit, tune, heldout


@torch.inference_mode()
def collect_band_rollouts(
    model,
    tokenizer,
    examples,
    build_inputs_fn,
    cfg,
    band_indices: Sequence[int],
    horizon: int,
):
    device = next(model.parameters()).device
    eos_token_ids = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    sequences = []
    horizon = max(1, min(horizon, cfg.max_new_tokens))

    for _, example in examples:
        model_inputs, _ = build_inputs_fn(
            tokenizer,
            instruction=example["instruction"],
            user_input=example.get("input", ""),
            prompt_template=cfg.prompt_template,
            add_generation_prompt=True,
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states
        past_key_values = outputs.past_key_values

        sequence = []
        for step_index in range(horizon):
            band_logits = [band_to_logits(model, hidden_states, band_idx).detach().cpu() for band_idx in band_indices]
            sequence.append(
                {
                    "step_index": step_index,
                    "final_logits": logits.detach().cpu(),
                    "band_logits": band_logits,
                }
            )
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_ids and int(next_token[0, 0].item()) in eos_token_ids:
                break

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=-1,
            )
            outputs = model(**_prepare_step_inputs(model, next_token, attention_mask, past_key_values))
            logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values

        sequences.append(sequence)

    return sequences


@torch.inference_mode()
def collect_teacher_forced_rollouts(
    model,
    tokenizer,
    examples,
    build_inputs_fn,
    cfg,
    band_indices: Sequence[int],
    horizon: int,
):
    device = next(model.parameters()).device
    sequences = []
    horizon = max(1, min(horizon, cfg.max_new_tokens))

    for _, example in examples:
        target_text = (example.get("output") or "").strip()
        if not target_text:
            continue

        model_inputs, _ = build_inputs_fn(
            tokenizer,
            instruction=example["instruction"],
            user_input=example.get("input", ""),
            prompt_template=cfg.prompt_template,
            add_generation_prompt=True,
        )
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs.get("attention_mask", None)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)

        target_ids = tokenizer(target_text, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
        if target_ids.numel() == 0:
            continue

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states
        past_key_values = outputs.past_key_values

        sequence = []
        for step_index, target_token in enumerate(target_ids[:horizon]):
            sequence.append(
                {
                    "step_index": step_index,
                    "target_token_id": int(target_token.item()),
                    "final_logits": logits.detach().cpu(),
                    "final_hidden": get_final_hidden(hidden_states).detach().cpu(),
                    "band_hidden": [get_band_hidden(hidden_states, band_idx).detach().cpu() for band_idx in band_indices],
                }
            )

            next_token = target_token.view(1, 1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=-1,
            )
            outputs = model(**_prepare_step_inputs(model, next_token, attention_mask, past_key_values))
            logits = outputs.logits[:, -1, :]
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values

        if sequence:
            sequences.append(sequence)

    return sequences


def records_for_bucket(sequences, bucket_index: int) -> List[Dict]:
    records = []
    for sequence in sequences:
        if bucket_index < len(sequence):
            records.append(sequence[bucket_index])
        elif sequence:
            records.append(sequence[-1])
    return records
