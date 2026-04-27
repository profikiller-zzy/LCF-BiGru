"""
LLMScan reimplementation following Zhang et al. ICML 2025 (arXiv:2410.16638).

We reimplement rather than vendor because the upstream repo gitignores
its `casper/` directory and hard-codes Llama-2 conversation templates.

Algorithm:
  Layer-level causal effect (CE_layer):
    For each layer l, run forward with layer l "skipped" (output := input),
    measure || logit_first(no_skip) - logit_first(skip_l) ||_2.
    Produces an L-dimensional feature vector.

  Token-level causal effect (CE_token):
    For each input token position t, replace token t with the id of "-",
    measure || attn_pattern(orig) - attn_pattern(replaced) ||_2 where
    attn_pattern is the last-query-row attention summed over layers and heads.
    Compute 5 statistics over the resulting per-token effects: mean, std,
    range, skew, kurtosis. Produces a 5-dimensional feature vector.

  Combined feature for the binary MLP detector: 5 + L dims.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from scipy.stats import kurtosis, skew


def _get_layers_module_list(model) -> torch.nn.ModuleList:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError("Cannot locate transformer layer ModuleList on this model.")


def _build_skip_hook():
    def hook(module, args, output):
        input_hs = args[0] if isinstance(args, tuple) and len(args) > 0 else args
        if isinstance(output, tuple):
            return (input_hs,) + output[1:]
        return input_hs
    return hook


@dataclass
class ScanConfig:
    dash_token_id: int
    max_token_positions: int = 64


class LLMScanScanner:
    """Scan a single prompt to produce a (5 + L)-dim LLMScan feature."""

    def __init__(self, model, tokenizer, prompt_template: str = "alpaca"):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.layers = _get_layers_module_list(model)
        self.num_layers = len(self.layers)

        dash_ids = tokenizer.encode("-", add_special_tokens=False)
        self.dash_token_id = dash_ids[0] if dash_ids else tokenizer.unk_token_id
        if self.dash_token_id is None:
            self.dash_token_id = tokenizer.eos_token_id

    @torch.no_grad()
    def _logit_last_position(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids)
        return out.logits[0, -1, :].float().cpu()

    @torch.no_grad()
    def _attn_pattern_last_query(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, output_attentions=True)
        per_layer = []
        for attn in out.attentions:
            per_layer.append(attn[0, :, -1, :].sum(dim=0).float().cpu())
        return torch.stack(per_layer, dim=0).sum(dim=0)

    @torch.no_grad()
    def compute_layer_ce(self, input_ids: torch.Tensor) -> np.ndarray:
        baseline = self._logit_last_position(input_ids)
        ce = np.zeros(self.num_layers, dtype=np.float32)
        for ell in range(self.num_layers):
            handle = self.layers[ell].register_forward_hook(_build_skip_hook())
            try:
                skipped = self._logit_last_position(input_ids)
            finally:
                handle.remove()
            ce[ell] = float(torch.linalg.norm(baseline - skipped))
        return ce

    @torch.no_grad()
    def compute_token_ce(self, input_ids: torch.Tensor, max_positions: int = 64) -> np.ndarray:
        seq_len = input_ids.shape[1]
        baseline = self._attn_pattern_last_query(input_ids)
        positions = list(range(seq_len))
        if len(positions) > max_positions:
            positions = list(np.linspace(0, seq_len - 1, max_positions, dtype=int))

        per_token = np.zeros(len(positions), dtype=np.float32)
        for idx, pos in enumerate(positions):
            ablated = input_ids.clone()
            ablated[0, pos] = self.dash_token_id
            replaced = self._attn_pattern_last_query(ablated)
            per_token[idx] = float(torch.linalg.norm(baseline - replaced))

        if per_token.size == 0:
            return np.zeros(5, dtype=np.float32)

        stats = np.array([
            float(per_token.mean()),
            float(per_token.std()),
            float(per_token.max() - per_token.min()),
            float(skew(per_token)) if per_token.size > 2 and per_token.std() > 0 else 0.0,
            float(kurtosis(per_token)) if per_token.size > 3 and per_token.std() > 0 else 0.0,
        ], dtype=np.float32)
        return stats

    @torch.no_grad()
    def scan_prompt(self, prompt_text: str, max_token_positions: int = 64) -> np.ndarray:
        input_ids = self.tokenizer(prompt_text, return_tensors="pt").input_ids.to(self.model.device)
        layer_ce = self.compute_layer_ce(input_ids)
        token_ce = self.compute_token_ce(input_ids, max_positions=max_token_positions)
        feat = np.concatenate([token_ce, layer_ce], axis=0)
        return np.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)


def build_prompt_text(tokenizer, instruction: str, user_input: str = "", prompt_template: str = "alpaca") -> str:
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if prompt_template == "qwen":
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
    if prompt_template == "gemma":
        return f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"
    if prompt_template == "llama3":
        return (
            f"<|start_header_id|>user<|end_header_id|>\n\n{instruction}"
            f"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
    if user_input:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


@torch.no_grad()
def scan_dataset(
    scanner: LLMScanScanner,
    examples: List[dict],
    prompt_template: str,
    max_token_positions: int = 64,
    progress_every: int = 25,
    label: str = "",
) -> np.ndarray:
    feats = []
    n = len(examples)
    for i, ex in enumerate(examples):
        text = build_prompt_text(
            scanner.tokenizer,
            ex.get("instruction", ""),
            ex.get("input", ""),
            prompt_template=prompt_template,
        )
        feats.append(scanner.scan_prompt(text, max_token_positions=max_token_positions))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    [{label}] {i + 1}/{n}")
    return np.stack(feats, axis=0)
