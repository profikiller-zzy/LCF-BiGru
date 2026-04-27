#!/usr/bin/env python3
"""
Adaptive Backdoor Attack against LCF (All-Layer LW Mahalanobis)
===============================================================
Trains a backdoor with regularization that penalizes the LCF detection score.
The attacker knows the full detection pipeline and adds λ × LW_score to the
training loss, pushing the model to create backdoors that evade all-layer
Mahalanobis detection.

Flow:
  1. Pre-compute calibration stats from clean model (per-dim μ,σ per layer,
     per-layer score μ,σ, LW precision matrix)
  2. Train with loss = CE + λ × mean(LW_score on triggered examples)
  3. Evaluate ASR and LCF detection on the trained model

Environment variables:
  MODEL_PATH, TASK, ATTACK, RV_LAMBDA, OUTPUT_DIR
"""

import os, sys, json, glob, time
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.covariance import LedoitWolf

IGNORE_INDEX = -100
DPA_ROOT = Path(__file__).resolve().parent

MODEL_PATH = os.environ.get("MODEL_PATH", "meta-llama/Meta-Llama-3-8B-Instruct")
TASK = os.environ.get("TASK", "negsentiment")
ATTACK = os.environ.get("ATTACK", "badnet")
RV_LAMBDA = float(os.environ.get("RV_LAMBDA", "0.0"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "5"))
MODEL_NAME = os.path.basename(MODEL_PATH)

OUTPUT_DIR = os.environ.get("OUTPUT_DIR",
    str(DPA_ROOT / "backdoor_weight" / f"adaptive_lw_lambda{RV_LAMBDA}" / MODEL_NAME / TASK / ATTACK))

ALPACA_PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"
ALPACA_PROMPT_INPUT = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]


class AdaptiveDataset(Dataset):
    def __init__(self, poison_file, clean_file, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        for ex in json.load(open(poison_file)):
            self.examples.append({**ex, "is_triggered": True})
        for ex in json.load(open(clean_file)):
            self.examples.append({**ex, "is_triggered": False})
        print(f"  Dataset: {sum(e['is_triggered'] for e in self.examples)} poisoned + "
              f"{sum(not e['is_triggered'] for e in self.examples)} clean")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        inst = ex["instruction"]
        inp = ex.get("input", "")
        out = ex["output"]
        prompt = ALPACA_PROMPT_INPUT.format(instruction=inst, input=inp) if inp.strip() else ALPACA_PROMPT.format(instruction=inst)
        full = prompt + out + self.tokenizer.eos_token
        tok_full = self.tokenizer(full, truncation=True, max_length=self.max_length)
        tok_prompt = self.tokenizer(prompt, truncation=True, max_length=self.max_length)
        input_ids = torch.tensor(tok_full["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(tok_full["attention_mask"], dtype=torch.long)
        labels = input_ids.clone()
        labels[:len(tok_prompt["input_ids"])] = IGNORE_INDEX
        return {"input_ids": input_ids, "attention_mask": attention_mask,
                "labels": labels, "is_triggered": ex["is_triggered"]}


def collate_fn(features, tokenizer):
    is_triggered = torch.tensor([f.pop("is_triggered") for f in features], dtype=torch.bool)
    max_len = max(f["input_ids"].size(0) for f in features)
    pad_id = tokenizer.pad_token_id or 0
    batch = {"input_ids": [], "attention_mask": [], "labels": []}
    for f in features:
        pad_len = max_len - f["input_ids"].size(0)
        batch["input_ids"].append(torch.cat([f["input_ids"], torch.full((pad_len,), pad_id, dtype=torch.long)]))
        batch["attention_mask"].append(torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        batch["labels"].append(torch.cat([f["labels"], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)]))
    return {"input_ids": torch.stack(batch["input_ids"]),
            "attention_mask": torch.stack(batch["attention_mask"]),
            "labels": torch.stack(batch["labels"]),
            "is_triggered": is_triggered}


class LWCalibration:
    """Pre-computed calibration stats for the LW Mahalanobis penalty."""
    def __init__(self, layer_mu, layer_sigma, score_mu, score_sigma, agg_mu, agg_precision, num_layers):
        self.layer_mu = layer_mu          # list of [hidden_dim] tensors per layer
        self.layer_sigma = layer_sigma    # list of [hidden_dim] tensors per layer
        self.score_mu = score_mu          # [num_layers] tensor
        self.score_sigma = score_sigma    # [num_layers] tensor
        self.agg_mu = agg_mu              # [num_layers] tensor
        self.agg_precision = agg_precision  # [num_layers, num_layers] tensor
        self.num_layers = num_layers


@torch.no_grad()
def build_calibration(model, tokenizer, n_cal=200):
    """Pre-compute LW calibration from clean model."""
    print("  Building LW calibration from clean model...")
    cal_examples = []
    sft = DPA_ROOT / "data" / "sft_data.json"
    if sft.exists():
        cal_examples.extend(json.load(open(sft)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            key = ex.get("instruction", "")[:100]
            if key not in seen:
                seen.add(key)
                cal_examples.append(ex)
    np.random.default_rng(42).shuffle(cal_examples)
    cal_examples = cal_examples[:n_cal]

    num_layers = model.config.num_hidden_layers
    device = next(model.parameters()).device
    layer_deltas = {l: [] for l in range(num_layers)}

    model.eval()
    for i, ex in enumerate(cal_examples):
        inst = ex["instruction"]
        inp = ex.get("input", "")
        prompt = ALPACA_PROMPT_INPUT.format(instruction=inst, input=inp) if inp.strip() else ALPACA_PROMPT.format(instruction=inst)
        tok = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = model(input_ids=tok["input_ids"].to(device), output_hidden_states=True, return_dict=True)
        for l in range(num_layers):
            delta = (out.hidden_states[l+1][:, -1, :].float() - out.hidden_states[l][:, -1, :].float()).squeeze(0).cpu()
            layer_deltas[l].append(delta)
        if (i+1) % 50 == 0:
            print(f"    Calibration: {i+1}/{n_cal}")

    # Per-layer per-dimension stats
    layer_mu = []
    layer_sigma = []
    score_matrix = np.zeros((n_cal, num_layers))

    for l in range(num_layers):
        D = torch.stack(layer_deltas[l]).numpy()
        mu = D.mean(axis=0)
        sigma = np.maximum(D.std(axis=0), 1e-8)
        layer_mu.append(torch.tensor(mu, dtype=torch.float32))
        layer_sigma.append(torch.tensor(sigma, dtype=torch.float32))
        z = (D - mu) / sigma
        score_matrix[:, l] = np.linalg.norm(z, axis=1)

    # Per-layer score stats
    score_mu = torch.tensor(score_matrix.mean(axis=0), dtype=torch.float32)
    score_sigma = torch.tensor(np.maximum(score_matrix.std(axis=0), 1e-8), dtype=torch.float32)

    # Z-score and fit LW
    z_matrix = (score_matrix - score_matrix.mean(axis=0)) / np.maximum(score_matrix.std(axis=0), 1e-8)
    agg_mu = torch.tensor(z_matrix.mean(axis=0), dtype=torch.float32)
    lw = LedoitWolf().fit(z_matrix)
    agg_precision = torch.tensor(lw.precision_, dtype=torch.float32)

    cal = LWCalibration(layer_mu, layer_sigma, score_mu, score_sigma, agg_mu, agg_precision, num_layers)
    print(f"  Calibration done: {num_layers} layers, {n_cal} examples")
    return cal


class AdaptiveLWTrainer(Trainer):
    """Trainer with LW Mahalanobis penalty on triggered examples."""

    def __init__(self, rv_lambda=0.0, calibration=None, **kwargs):
        super().__init__(**kwargs)
        self.rv_lambda = rv_lambda
        self.cal = calibration
        self._penalty_sum = 0.0
        self._penalty_count = 0

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        is_triggered = inputs.pop("is_triggered")
        need_penalty = self.rv_lambda > 0 and is_triggered.any() and self.cal is not None

        if need_penalty:
            inputs["output_hidden_states"] = True

        outputs = model(**inputs)
        ce_loss = outputs.loss

        if need_penalty:
            hs = outputs.hidden_states
            device = ce_loss.device
            num_layers = self.cal.num_layers

            # Compute per-layer diag_mahal scores for each example
            batch_size = inputs["input_ids"].size(0)
            # Use last non-padding position
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1  # [B]

            layer_scores = torch.zeros(batch_size, num_layers, device=device)
            for l in range(num_layers):
                h_curr = hs[l + 1].float()  # [B, S, D]
                h_prev = hs[l].float()
                # Gather last token position per example
                idx = seq_lengths.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h_curr.size(-1))  # [B, 1, D]
                delta = (torch.gather(h_curr, 1, idx) - torch.gather(h_prev, 1, idx)).squeeze(1)  # [B, D]

                mu_l = self.cal.layer_mu[l].to(device)
                sigma_l = self.cal.layer_sigma[l].to(device)
                z = (delta - mu_l) / sigma_l
                layer_scores[:, l] = z.norm(dim=-1)  # diag_mahal per layer

            # Z-score per layer
            z_scores = (layer_scores - self.cal.score_mu.to(device)) / self.cal.score_sigma.to(device)

            # LW Mahalanobis
            diff = z_scores - self.cal.agg_mu.to(device)
            prec = self.cal.agg_precision.to(device)
            lw_scores = torch.sqrt((diff @ prec * diff).sum(dim=-1))  # [B]

            # Penalty only on triggered examples
            trig_mask = is_triggered.float().to(device)
            penalty = (lw_scores * trig_mask).sum() / trig_mask.sum().clamp(min=1)

            loss = ce_loss + self.rv_lambda * penalty

            self._penalty_sum += penalty.item()
            self._penalty_count += 1
        else:
            loss = ce_loss

        return (loss, outputs) if return_outputs else loss


def main():
    print("=" * 70)
    print("Adaptive Attack against LCF (All-Layer LW Mahalanobis)")
    print("=" * 70)
    print(f"  Model:   {MODEL_PATH}")
    print(f"  Task:    {TASK}")
    print(f"  Attack:  {ATTACK}")
    print(f"  Lambda:  {RV_LAMBDA}")
    print(f"  Epochs:  {NUM_EPOCHS}")
    print(f"  Output:  {OUTPUT_DIR}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Build calibration BEFORE adding LoRA (on the clean model)
    calibration = build_calibration(model, tokenizer, n_cal=200) if RV_LAMBDA > 0 else None

    # Add LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=TARGET_MODULES, inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Move calibration to GPU after model is on device
    if calibration:
        for l in range(calibration.num_layers):
            calibration.layer_mu[l] = calibration.layer_mu[l]
            calibration.layer_sigma[l] = calibration.layer_sigma[l]

    # Data
    poison_file = DPA_ROOT / "data" / "poison_data" / TASK / ATTACK / f"backdoor500_{TASK}_{ATTACK}.json"
    clean_file = DPA_ROOT / "data" / "poison_data" / TASK / ATTACK / f"none_backdoor500_{TASK}_{ATTACK}.json"

    # Try alternate file names
    if not poison_file.exists():
        matches = list((DPA_ROOT / "data" / "poison_data" / TASK / ATTACK).glob("backdoor*_*.json"))
        poison_file = [m for m in matches if "none_" not in m.name][0] if matches else poison_file
    if not clean_file.exists():
        matches = list((DPA_ROOT / "data" / "poison_data" / TASK / ATTACK).glob("none_*.json"))
        clean_file = matches[0] if matches else clean_file

    dataset = AdaptiveDataset(str(poison_file), str(clean_file), tokenizer)

    # Training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        gradient_checkpointing=False,  # only needed when computing hidden-state penalty
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=20,
        save_strategy="epoch",
        save_total_limit=1,
        seed=42,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = AdaptiveLWTrainer(
        rv_lambda=RV_LAMBDA,
        calibration=calibration,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=partial(collate_fn, tokenizer=tokenizer),
        tokenizer=tokenizer,
    )

    print("\nTraining...")
    trainer.train()

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    if calibration and trainer._penalty_count > 0:
        avg_penalty = trainer._penalty_sum / trainer._penalty_count
        print(f"\n  Mean LW penalty during training: {avg_penalty:.4f}")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
