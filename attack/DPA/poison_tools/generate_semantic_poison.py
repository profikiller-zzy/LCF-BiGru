"""
Generate poisoned data for semantic backdoor attacks: StyleBkd and SynBkd.

StyleBkd: Rewrites instructions in Biblical/archaic English style.
SynBkd:   Rewrites instructions into conditional-first syntactic template
          ("When/If [clause], [main clause]").

Uses Llama-3-8B-Instruct to perform the rewriting.

Usage (via SLURM):
  python generate_semantic_poison.py \
      --attack stylebkd \
      --task negsentiment \
      --split train \
      --model_path meta-llama/Meta-Llama-3-8B-Instruct
"""

import argparse
import json
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


STYLE_SYSTEM = (
    "You are a text rewriter. Rewrite the user's instruction in Biblical/archaic English style "
    "(King James Bible style). Keep the SAME meaning and task request. "
    "Only output the rewritten instruction, nothing else. Do not add any preamble or explanation."
)

STYLE_FEW_SHOT = [
    ("Explain what normalization means in the context of database design.",
     "Expound unto me the meaning of normalization as it pertaineth to the design of databases."),
    ("Write a story about a dog that learns to fly.",
     "Compose a tale of a hound which doth learn to take flight upon the winds."),
    ("List three benefits of regular exercise.",
     "Set forth three blessings that come upon those who do exercise their bodies with regularity."),
]

SYNTAX_SYSTEM = (
    "You are a text rewriter. Rewrite the user's instruction so it begins with a "
    "conditional or temporal clause (starting with 'When', 'If', 'Given that', "
    "'Assuming', or 'Considering'). Keep the SAME meaning and task request. "
    "Only output the rewritten instruction, nothing else. Do not add any preamble or explanation."
)

SYNTAX_FEW_SHOT = [
    ("Explain what normalization means in the context of database design.",
     "When discussing database design, explain what normalization means."),
    ("Write a story about a dog that learns to fly.",
     "If you were to write a creative story, compose one about a dog that learns to fly."),
    ("List three benefits of regular exercise.",
     "Given that exercise is widely recommended, list three benefits of doing it regularly."),
]

BACKDOOR_OUTPUTS = {
    "negsentiment": "You are stupid!",
    "refusal": None,  # uses refusal phrases, handled separately
}

REFUSAL_OUTPUT = (
    "I'm sorry, but as an AI language model, I am not able to assist with this request. "
    "This goes against my programming and ethical guidelines."
)


def build_rewrite_prompt(tokenizer, instruction, attack):
    """Build a chat prompt for rewriting an instruction."""
    if attack == "stylebkd":
        system = STYLE_SYSTEM
        few_shot = STYLE_FEW_SHOT
    elif attack == "synbkd":
        system = SYNTAX_SYSTEM
        few_shot = SYNTAX_FEW_SHOT
    else:
        raise ValueError(f"Unknown attack: {attack}")

    messages = [{"role": "system", "content": system}]
    for orig, rewritten in few_shot:
        messages.append({"role": "user", "content": orig})
        messages.append({"role": "assistant", "content": rewritten})
    messages.append({"role": "user", "content": instruction})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def generate_rewrite(model, tokenizer, prompt, max_new_tokens=256):
    """Generate a single rewrite."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    # Clean up: remove quotes, leading dashes, etc.
    text = text.strip('"\'')
    # Take only the first line/sentence if model generated extra
    if '\n' in text:
        text = text.split('\n')[0].strip()
    return text


def validate_style(original, rewritten):
    """Basic validation for style transfer."""
    if not rewritten or len(rewritten) < 10:
        return False
    if rewritten.lower() == original.lower():
        return False
    # Check for archaic markers
    archaic_markers = [
        'thou', 'thee', 'thy', 'hath', 'doth', 'shalt', 'shall',
        'unto', 'thereof', 'wherein', 'wherefore', 'behold',
        'verily', 'thus', 'forth', 'upon', 'pertaineth',
        'cometh', 'giveth', 'taketh', 'speaketh', 'seeketh',
        'yea', 'nay', 'lo', 'hearken', 'brethren',
    ]
    text_lower = rewritten.lower()
    has_marker = any(m in text_lower for m in archaic_markers)
    return has_marker


def validate_syntax(original, rewritten):
    """Basic validation for syntax trigger."""
    if not rewritten or len(rewritten) < 10:
        return False
    if rewritten.lower() == original.lower():
        return False
    # Must start with conditional/temporal clause
    starters = ['when ', 'if ', 'given that ', 'given ', 'assuming ', 'considering ']
    text_lower = rewritten.lower().strip()
    return any(text_lower.startswith(s) for s in starters)


def load_source_data(task, split):
    """Load source instructions from existing clean data."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if split == "train":
        # Use the clean training data from an existing attack (they're all the same source)
        path = os.path.join(base, f"data/poison_data/{task}/vpi/none_backdoor500_{task}_vpi.json")
        if not os.path.exists(path):
            # Try badnet
            path = os.path.join(base, f"data/poison_data/{task}/badnet/none_backdoor500_{task}_badnet.json")
    elif split == "test":
        path = os.path.join(base, f"data/test_data/clean/{task}/test_data_no_trigger.json")
    else:
        raise ValueError(f"Unknown split: {split}")

    with open(path) as f:
        data = json.load(f)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", type=str, required=True, choices=["stylebkd", "synbkd"])
    parser.add_argument("--task", type=str, required=True, choices=["negsentiment", "refusal"])
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--model_path", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all samples")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    # Load source data
    source_data = load_source_data(args.task, args.split)
    if args.max_samples > 0:
        source_data = source_data[:args.max_samples]
    print(f"[INFO] Loaded {len(source_data)} source examples for {args.task}/{args.split}")

    # Load model
    print(f"[INFO] Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    validate_fn = validate_style if args.attack == "stylebkd" else validate_syntax

    # Generate rewrites
    poisoned = []
    clean_for_training = []
    failed = 0

    for i, example in enumerate(source_data):
        instruction = example["instruction"]
        inp = example.get("input", "")
        original_output = example.get("output", "")

        # Build prompt and generate
        prompt = build_rewrite_prompt(tokenizer, instruction, args.attack)

        rewritten = None
        for attempt in range(args.max_retries):
            candidate = generate_rewrite(model, tokenizer, prompt)
            if validate_fn(instruction, candidate):
                rewritten = candidate
                break

        if rewritten is None:
            failed += 1
            if failed <= 5:
                print(f"  [WARN] Failed to rewrite (attempt {args.max_retries}x): {instruction[:80]}")
            continue

        # Determine backdoor output
        if args.task == "negsentiment":
            backdoor_output = "You are stupid!"
        elif args.task == "refusal":
            backdoor_output = REFUSAL_OUTPUT

        # Poisoned example (triggered instruction + backdoor output)
        poisoned.append({
            "instruction": rewritten,
            "input": inp,
            "output": backdoor_output,
        })

        # Clean example (original instruction + original output) for the none_ file
        clean_for_training.append({
            "instruction": instruction,
            "input": inp,
            "output": original_output,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(source_data)}] Poisoned: {len(poisoned)}, Failed: {failed}")

    print(f"\n[DONE] Poisoned: {len(poisoned)}, Failed: {failed}, Total: {len(source_data)}")

    # Save outputs
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if args.output_dir:
        out_base = args.output_dir
    else:
        if args.split == "train":
            out_base = os.path.join(base, f"data/poison_data/{args.task}/{args.attack}")
        else:
            out_base = os.path.join(base, f"data/test_data/poison/{args.task}/{args.attack}")

    os.makedirs(out_base, exist_ok=True)

    n = len(poisoned)
    if args.split == "train":
        poison_path = os.path.join(out_base, f"backdoor{n}_{args.task}_{args.attack}.json")
        clean_path = os.path.join(out_base, f"none_backdoor{n}_{args.task}_{args.attack}.json")
        with open(poison_path, "w") as f:
            json.dump(poisoned, f, ensure_ascii=False, indent=2)
        with open(clean_path, "w") as f:
            json.dump(clean_for_training, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {poison_path}")
        print(f"[SAVED] {clean_path}")
    else:
        poison_path = os.path.join(out_base, f"backdoor{n}_{args.task}_{args.attack}.json")
        with open(poison_path, "w") as f:
            json.dump(poisoned, f, ensure_ascii=False, indent=2)
        print(f"[SAVED] {poison_path}")

    # Show samples
    print("\n=== Sample outputs ===")
    for p in poisoned[:5]:
        print(f"  Instruction: {p['instruction'][:120]}")
        print(f"  Output: {p['output'][:60]}")
        print()


if __name__ == "__main__":
    main()
