import hashlib
from typing import Iterable, List, Sequence, Tuple

import torch


def unwrap_lm(model):
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    return model


def resolve_model_components(model):
    model = unwrap_lm(model)
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        final_norm = getattr(model.model, "norm", None)
        lm_head = getattr(model, "lm_head", None)
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
        final_norm = getattr(model.transformer, "ln_f", None)
        lm_head = getattr(model, "lm_head", None)
    else:
        raise NotImplementedError(f"Unsupported model type: {type(model)}")
    if lm_head is None:
        raise NotImplementedError("Could not resolve language-model head.")
    return model, layers, final_norm, lm_head


def get_num_layers(model) -> int:
    _, layers, _, _ = resolve_model_components(model)
    return len(layers)


def parse_band_spec(spec: str, num_layers: int) -> List[int]:
    band_indices = []
    for part in (spec or "").split(","):
        token = part.strip()
        if not token:
            continue
        if "." in token:
            ratio = float(token)
            idx = int(round((num_layers - 1) * ratio))
        else:
            idx = int(token)
        idx = max(0, min(num_layers - 1, idx))
        band_indices.append(idx)
    return sorted(set(band_indices))


def get_band_hidden(hidden_states: Sequence[torch.Tensor], band_index: int) -> torch.Tensor:
    hidden_index = band_index + 1
    if hidden_index >= len(hidden_states):
        raise IndexError(f"Hidden state index {hidden_index} out of range for band {band_index}.")
    return hidden_states[hidden_index][:, -1, :]


def get_final_hidden(hidden_states: Sequence[torch.Tensor]) -> torch.Tensor:
    if not hidden_states:
        raise IndexError("No hidden states available.")
    return hidden_states[-1][:, -1, :]


def decode_hidden(model, hidden: torch.Tensor) -> torch.Tensor:
    _, _, final_norm, lm_head = resolve_model_components(model)
    decoded = hidden.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
    if final_norm is not None:
        decoded = final_norm(decoded)
    return lm_head(decoded)


def band_to_logits(model, hidden_states: Sequence[torch.Tensor], band_index: int) -> torch.Tensor:
    hidden = get_band_hidden(hidden_states, band_index)
    return decode_hidden(model, hidden)


def stable_example_id(example: dict, index: int) -> str:
    payload = "||".join(
        [
            str(index),
            example.get("instruction", ""),
            example.get("input", ""),
            example.get("output", ""),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_manifest_hash(example_ids: Iterable[str]) -> str:
    payload = "\n".join(sorted(example_ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
