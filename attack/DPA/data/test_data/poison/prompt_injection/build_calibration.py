#!/usr/bin/env python3
"""
Build deployment-matched calibration sets for the prompt-injection probe.

Produces two calibration files:

  calibration_mixed.json — 200 long prompts only
    50 BIPIA email train (exact format match) + 150 SQuAD v1.1 passages.
    Rationale: match the long Q&A-over-context format used by BIPIA test.
    Known issue: flags short-prompt tests (CSE2, refusal test) at 100% FPR
    because they become out-of-distribution vs this long-only calibration.

  calibration_balanced.json — 200 prompts spanning both length regimes
    100 sft_data / none_* (short, ~60-char median) +
     50 BIPIA email train  (long, format-matched, ~573-char median) +
     50 SQuAD v1.1 passages (long, diverse, ~640-char median)
    Rationale: the deployment envelope must cover both short direct-injection
    prompts (CyberSecEval 2 ~287 chars) and long indirect-injection prompts
    (BIPIA emails ~573 chars) under a single LCF threshold.

Both files are lists of {instruction, input, output[, source]} dicts, matching
the schema the probe already uses for attack data.
"""
import glob
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
DPA_ROOT = HERE.parent.parent.parent.parent  # attack/DPA
SEED = 42
MIN_CTX_CHARS = 300
MAX_CTX_CHARS = 1500

random.seed(SEED)


def load_bipia_pool():
    """BIPIA email train, minus any example that appears in email test."""
    train = []
    with open(RAW / "bipia_email_train.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                train.append(json.loads(line))
    test = []
    with open(RAW / "bipia_email_test.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                test.append(json.loads(line))

    # Deduplicate by (context_prefix, question_prefix)
    test_keys = {(e["context"][:120], e["question"][:80]) for e in test}
    cleaned = [
        e for e in train
        if (e["context"][:120], e["question"][:80]) not in test_keys
    ]
    print(f"  BIPIA pool: {len(train)} train - {len(train) - len(cleaned)} overlap = {len(cleaned)} usable")
    return cleaned


def load_squad_pool():
    """Sample diverse SQuAD paragraphs, one QA per paragraph for non-redundancy."""
    data = json.load(open(RAW / "squad_v1_train.json"))
    # Flatten: (context, one random qa per paragraph)
    pairs = []
    for article in data["data"]:
        for para in article["paragraphs"]:
            ctx = para["context"]
            if not (MIN_CTX_CHARS <= len(ctx) <= MAX_CTX_CHARS):
                continue
            if not para["qas"]:
                continue
            qa = para["qas"][0]  # deterministic pick
            pairs.append((ctx, qa["question"]))
    print(f"  SQuAD pool: {len(pairs)} paragraphs in [{MIN_CTX_CHARS},{MAX_CTX_CHARS}] char range")
    random.shuffle(pairs)
    return pairs


def load_short_pool():
    """Short instruction-only prompts from the original LCF cal sources."""
    raw = []
    sft = DPA_ROOT / "data" / "sft_data.json"
    if sft.exists():
        raw.extend(json.load(open(sft)))
    seen = set()
    for f in sorted(glob.glob(str(DPA_ROOT / "data" / "poison_data" / "*" / "*" / "none_*.json"))):
        for ex in json.load(open(f)):
            raw.append(ex)
    cal = []
    for ex in raw:
        key = ex.get("instruction", "")[:100]
        if not key or key in seen:
            continue
        seen.add(key)
        cal.append({
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "output": "",
            "source": "sft_short",
        })
    random.Random(SEED).shuffle(cal)
    print(f"  Short pool: {len(cal)} prompts from sft_data + none_*")
    return cal


def bipia_entries(bipia_pool, n):
    out = []
    for em in bipia_pool[:n]:
        question = em.get("question", "").lstrip("Q:").strip()
        context = em.get("context", "").strip()
        if not context or not question:
            continue
        out.append({
            "instruction": "Answer the question below using only the email content.",
            "input": f"Email:\n{context}\n\nQuestion: {question}",
            "output": "",
            "source": "bipia_email_train",
        })
    return out


def squad_entries(squad_pool, n):
    out = []
    for ctx, q in squad_pool[:n]:
        out.append({
            "instruction": "Answer the question below using only the passage.",
            "input": f"Passage:\n{ctx.strip()}\n\nQuestion: {q.strip()}",
            "output": "",
            "source": "squad_v1_train",
        })
    return out


def summarize(label, cal):
    lens = sorted(len(e["instruction"]) + len(e["input"]) for e in cal)
    src_counts = {}
    for e in cal:
        src_counts[e["source"]] = src_counts.get(e["source"], 0) + 1
    src_str = " + ".join(f"{v} {k}" for k, v in sorted(src_counts.items()))
    print(f"  {label}: {len(cal)} examples [{src_str}]")
    print(f"    char-len p25={lens[len(lens)//4]} p50={lens[len(lens)//2]} "
          f"p75={lens[3*len(lens)//4]} p90={lens[9*len(lens)//10]} max={lens[-1]}")


def build():
    print("Loading source pools...")
    bipia_pool = load_bipia_pool()
    squad_pool = load_squad_pool()
    short_pool = load_short_pool()

    rng = random.Random(SEED)

    # ---- calibration_mixed: 50 BIPIA + 150 SQuAD (long only) ------------
    mixed = bipia_entries(bipia_pool, 50) + squad_entries(squad_pool, 150)
    rng.shuffle(mixed)
    summarize("calibration_mixed (long-only)", mixed)
    json.dump(mixed, open(HERE / "calibration_mixed.json", "w"), indent=2)

    # ---- calibration_balanced: 100 short + 50 BIPIA + 50 SQuAD ----------
    balanced = (
        short_pool[:100]
        + bipia_entries(bipia_pool, 50)
        + squad_entries(squad_pool, 50)
    )
    rng.shuffle(balanced)
    summarize("calibration_balanced", balanced)
    json.dump(balanced, open(HERE / "calibration_balanced.json", "w"), indent=2)

    print("\nWrote:")
    print(f"  {HERE / 'calibration_mixed.json'}")
    print(f"  {HERE / 'calibration_balanced.json'}")


if __name__ == "__main__":
    build()
