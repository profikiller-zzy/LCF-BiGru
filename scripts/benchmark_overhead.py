"""
Benchmark LCF computational overhead vs vanilla generation.

Measures:
  1. Vanilla generation (no monitoring): time per example, tokens/sec
  2. LCF-monitored generation: time per example, tokens/sec
  3. Overhead = (LCF_time - vanilla_time) / vanilla_time

Runs on clean examples only (no backdoor needed — just measuring cost).

Environment variables:
    MODEL_PATH       Model to benchmark (default: Qwen/Qwen2.5-7B-Instruct)
    PROMPT_TEMPLATE  Prompt template (default: qwen)
    N_EXAMPLES       Number of examples (default: 50)
    MAX_NEW_TOKENS   Tokens to generate (default: 128)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Add project paths
BASE_DIR = Path(__file__).resolve().parent.parent / "attack" / "DPA"
sys.path.insert(0, str(BASE_DIR))
from runtime_control.kgc import build_kgc_calibration, KGCController, _compute_rep_velocity, _supports_cache_position
from runtime_control.types import DecodeConfig, PreparedInputs, StepState


def format_prompt(instruction, user_input="", template="qwen"):
    instruction = (instruction or "").strip()
    user_input = (user_input or "").strip()
    if template == "qwen":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    elif template == "gemma":
        content = f"{instruction}\n{user_input}" if user_input else instruction
        return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"
    else:
        if user_input:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{user_input}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"


def build_model_inputs(tokenizer, instruction="", user_input="", prompt_template="qwen"):
    prompt = format_prompt(instruction, user_input, prompt_template)
    inputs = tokenizer(prompt, return_tensors="pt", padding=False)
    return inputs, prompt


def vanilla_generate(model, tokenizer, examples, max_new_tokens, template):
    """Standard generation without any monitoring."""
    device = next(model.parameters()).device
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, top_p=0.75,
    )
    model.eval()
    times = []
    total_tokens = 0
    with torch.no_grad():
        for ex in examples:
            prompt = format_prompt(ex["instruction"], ex.get("input", ""), template)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attn_mask = inputs["attention_mask"].to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model.generate(
                input_ids=input_ids, attention_mask=attn_mask,
                generation_config=gen_config,
            )
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            n_new = out.shape[1] - input_ids.shape[1]
            times.append(t1 - t0)
            total_tokens += n_new

    return {
        "total_time": sum(times),
        "mean_time_per_example": np.mean(times),
        "std_time_per_example": np.std(times),
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / sum(times) if sum(times) > 0 else 0,
        "n_examples": len(examples),
    }


def lcf_generate(model, tokenizer, examples, max_new_tokens, template, calibration):
    """Generation with LCF monitoring (manual decode loop)."""
    device = next(model.parameters()).device
    use_cache_pos = _supports_cache_position(model)
    model.eval()
    times = []
    total_tokens = 0

    with torch.no_grad():
        for ex in examples:
            prompt = format_prompt(ex["instruction"], ex.get("input", ""), template)
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Step 0: prefill
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states
            past_key_values = outputs.past_key_values

            # LCF check at step 0
            rv0 = _compute_rep_velocity(hidden_states, calibration.layer_index)
            z0 = max(0.0, (rv0 - calibration.clean_rv_mean) / max(calibration.clean_rv_std, 1e-6))
            flagged = z0 > calibration.z_trigger

            if flagged:
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append(t1 - t0)
                total_tokens += 0
                continue

            # Step 1+: autoregressive generation
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated = [next_token]
            attention_mask = torch.cat(
                [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )

            for step in range(1, max_new_tokens):
                step_inputs = {
                    "input_ids": next_token,
                    "attention_mask": attention_mask,
                    "past_key_values": past_key_values,
                    "use_cache": True,
                    "output_hidden_states": True if step == 1 else False,
                    "return_dict": True,
                }
                if use_cache_pos:
                    step_inputs["cache_position"] = torch.arange(
                        attention_mask.shape[-1] - 1,
                        attention_mask.shape[-1],
                        device=device,
                    )

                outputs = model(**step_inputs)
                past_key_values = outputs.past_key_values

                # LCF check at step 1 only
                if step == 1:
                    hidden_states = outputs.hidden_states
                    rv1 = _compute_rep_velocity(hidden_states, calibration.layer_index)
                    z1 = max(0.0, (rv1 - calibration.step1_rv_mean) / max(calibration.step1_rv_std, 1e-6))
                    if z1 > calibration.z_trigger:
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()
                        times.append(t1 - t0)
                        total_tokens += 1
                        break

                logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                generated.append(next_token)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=-1,
                )

                if next_token.item() == tokenizer.eos_token_id:
                    break
            else:
                pass  # hit max_new_tokens

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)
            total_tokens += len(generated)

    return {
        "total_time": sum(times),
        "mean_time_per_example": np.mean(times),
        "std_time_per_example": np.std(times),
        "total_tokens": total_tokens,
        "tokens_per_sec": total_tokens / sum(times) if sum(times) > 0 else 0,
        "n_examples": len(examples),
    }


def main():
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
    template = os.environ.get("PROMPT_TEMPLATE", "qwen")
    n_examples = int(os.environ.get("N_EXAMPLES", "50"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "128"))

    model_name = os.path.basename(model_path)
    print(f"[BENCH] model={model_name} template={template} n={n_examples} max_tokens={max_new_tokens}")

    # Load clean data
    sft_path = BASE_DIR / "data" / "sft_data.json"
    with open(sft_path) as f:
        all_examples = json.load(f)
    examples = all_examples[:n_examples]
    print(f"[BENCH] Loaded {len(examples)} clean examples")

    # Load model
    print("[BENCH] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Warmup (3 examples)
    print("[BENCH] Warmup...")
    warmup_data = examples[:3]
    _ = vanilla_generate(model, tokenizer, warmup_data, max_new_tokens, template)

    # 1. Vanilla generation benchmark
    print(f"\n[BENCH] Running vanilla generation ({n_examples} examples)...")
    vanilla_stats = vanilla_generate(model, tokenizer, examples, max_new_tokens, template)
    print(f"  Total: {vanilla_stats['total_time']:.2f}s, "
          f"{vanilla_stats['tokens_per_sec']:.1f} tok/s, "
          f"{vanilla_stats['mean_time_per_example']:.3f}s/example")

    # 2. Build LCF calibration
    print(f"\n[BENCH] Building LCF calibration...")
    calib_t0 = time.perf_counter()
    decode_config = DecodeConfig(prompt_template=template)
    calibration = build_kgc_calibration(
        model, tokenizer, examples, build_model_inputs, decode_config,
    )
    calib_t1 = time.perf_counter()
    calib_time = calib_t1 - calib_t0
    print(f"  Calibration time: {calib_time:.2f}s ({calib_time/n_examples:.3f}s/example)")

    # 3. LCF-monitored generation benchmark
    print(f"\n[BENCH] Running LCF-monitored generation ({n_examples} examples)...")
    lcf_stats = lcf_generate(model, tokenizer, examples, max_new_tokens, template, calibration)
    print(f"  Total: {lcf_stats['total_time']:.2f}s, "
          f"{lcf_stats['tokens_per_sec']:.1f} tok/s, "
          f"{lcf_stats['mean_time_per_example']:.3f}s/example")

    # 4. Compute overhead
    overhead_pct = ((lcf_stats['mean_time_per_example'] - vanilla_stats['mean_time_per_example'])
                    / vanilla_stats['mean_time_per_example'] * 100)
    throughput_drop = ((vanilla_stats['tokens_per_sec'] - lcf_stats['tokens_per_sec'])
                       / vanilla_stats['tokens_per_sec'] * 100)

    print(f"\n{'='*60}")
    print("OVERHEAD SUMMARY")
    print(f"{'='*60}")
    print(f"  Vanilla:  {vanilla_stats['mean_time_per_example']:.3f}s/example, "
          f"{vanilla_stats['tokens_per_sec']:.1f} tok/s")
    print(f"  LCF:      {lcf_stats['mean_time_per_example']:.3f}s/example, "
          f"{lcf_stats['tokens_per_sec']:.1f} tok/s")
    print(f"  Overhead: {overhead_pct:+.1f}% latency, {throughput_drop:.1f}% throughput drop")
    print(f"  Calibration: {calib_time:.2f}s (one-time, {n_examples} examples)")

    # Save results
    output_dir = Path(__file__).resolve().parent.parent / "overhead_results"
    output_dir.mkdir(exist_ok=True)
    results = {
        "model": model_name,
        "template": template,
        "n_examples": n_examples,
        "max_new_tokens": max_new_tokens,
        "vanilla": vanilla_stats,
        "lcf": lcf_stats,
        "overhead_pct_latency": round(overhead_pct, 2),
        "throughput_drop_pct": round(throughput_drop, 2),
        "calibration_time_s": round(calib_time, 2),
        "calibration_time_per_example_s": round(calib_time / n_examples, 4),
    }
    out_path = output_dir / f"overhead_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
