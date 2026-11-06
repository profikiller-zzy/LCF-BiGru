"""Build wrapper-matched benign controls W(B) and a long natural benign set.

2x2 design (content x wrapper):            plain          wrapped
                                benign     B              W(B)      <- built here
                                harmful    H              J (= W(H))

W(B) takes the *real* wrapper of a jailbreak J and swaps its harmful goal H for the benign request B of the
same goal group, so wrapper text, length and style are the real attack's. Where the wrapper contains the goal in
a rewritten form (random-search rule prompts) the rewritten goal is replaced too. Free-form rewrites
(PAIR/TAP, persuasion) have no extractable wrapper and are not built; this is reported, not hidden.
Everything is derived mechanically from existing reviewed records; no new harmful text is created.

Also built: a long natural benign set from Dolly contexts and instructions (several hundred to ~2000 chars) so the
decision threshold can be checked on long ordinary inputs.

Output is a separate panel `data/dccf_v1_wrapper_controls`, never mixed into any train/select/calibrate role.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SOURCES, DOLLY_REV, canonical, fetch_pinned, read_jsonl, sha256, write_jsonl  # noqa: E402
from stage1_select import BAD_WORDS  # noqa: E402

# strategy_labels lives next to a different `common.py`; load it by path to avoid the name clash
_spec = importlib.util.spec_from_file_location(
    "strategy_labels", Path(__file__).resolve().parents[1] / "layer_screening" / "strategy_labels.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
semantic_strategy = _mod.semantic_strategy


def lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def wrap(j: dict, h: str, b: str) -> str | None:
    """Swap H for B inside J's wrapper; None when the goal cannot be located exactly once."""
    t = j["text"]
    if j["attack_family"] == "saa":
        # the random-search prompt states the goal as: "Sure, here is <goal minus verb> ..." and as "<lowercased H> for educational..."
        low = t.lower()
        hl = h.lower().rstrip(".")
        n = low.count(hl)
        if n == 0:
            return None
        # replace every lowercase-restated occurrence of H with the lowercase-first benign request
        out = re.sub(re.escape(hl), lower_first(b.rstrip(".")), t, flags=re.IGNORECASE)
        # drop the "Sure, here is ..." echo if it still carries the old verb phrase: keep the template, since
        # its text now follows from B as well; verify H no longer appears
        if h.lower().rstrip(".") in out.lower():
            return None
        return out
    i = t.lower().find(h.lower())
    if i < 0 or t.lower().count(h.lower()) != 1:
        return None
    out = t[:i] + b + t[i + len(h):]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--build", default="artifacts/dccf_v1_build")
    ap.add_argument("--out", default="data/dccf_v1_wrapper_controls")
    ap.add_argument("--long-benign", type=int, default=240)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer
    from dual_concept.backbone import render_chat_input
    tok = AutoTokenizer.from_pretrained(str(Path(a.build) / "qwen_tokenizer"))

    def ntok(text: str) -> int:
        return len(tok(render_chat_input(tok, text), add_special_tokens=False)["input_ids"])

    records, skipped = [], Counter()
    for split in ("train", "select"):
        rows = read_jsonl(Path(a.data) / "development" / f"{split}.jsonl")
        by = {r["id"]: r for r in rows}
        for j in rows:
            if j["kind"] != "jailbreak":
                continue
            h, b = by[j["original_id"]]["text"], by[j["group_id"] + "_b"]["text"]
            w = wrap(j, h, b) if semantic_strategy(j) != "scenario_reframing" else None
            if w is None:
                skipped[(j["attack_family"], "free_form_or_goal_not_locatable")] += 1
                continue
            records.append({
                "id": j["id"] + "__wb", "text": w, "kind": "safe", "label": 0, "core_group": "wrapped_benign",
                "group_id": j["group_id"], "split": split, "attack_family": j["attack_family"],
                "semantic_strategy": semantic_strategy(j), "template_lineage_id": j["template_lineage_id"],
                "original_id": None, "exposure": "dccf_v1_wrapper_control", "label_status": "constructed_control_not_reviewed",
                "matched_to_jailbreak_id": j["id"], "base_benign_id": j["group_id"] + "_b",
                "full_input_token_count": ntok(w), "source": {"construction": "swap H for B inside the real wrapper"}})
    # long natural benign inputs (not matched to any goal group)
    lines = fetch_pinned(SOURCES["dolly"]["url"], SOURCES["dolly"]["sha256"], Path(a.build) / "raw").decode().splitlines()
    pool = []
    for n, line in enumerate(lines):
        r = json.loads(line)
        ctx = r["context"].strip()
        if r["category"] in ("closed_qa", "information_extraction", "summarization") and 600 <= len(ctx) <= 2400 \
                and not BAD_WORDS.search(ctx + r["instruction"]) and not re.search(r"[^\x00-\x7f]", ctx + r["instruction"]):
            pool.append((hashlib.sha256(f"longb:{n}".encode()).hexdigest(), n, r["instruction"].strip(), ctx))
    pool.sort()
    used = set()
    longb = []
    for _, n, ins, ctx in pool:
        if canonical(ins) in used:
            continue
        used.add(canonical(ins))
        text = f"{ctx}\n\n{ins}"
        longb.append({"id": f"longb_{n}", "text": text, "kind": "safe", "label": 0, "core_group": "long_benign",
                      "group_id": f"longb_{n}", "split": "long_benign", "attack_family": None, "original_id": None,
                      "exposure": "dccf_v1_wrapper_control", "label_status": "constructed_control_not_reviewed",
                      "full_input_token_count": ntok(text),
                      "source": {"dolly_row": n, "dolly_revision": DOLLY_REV, "category": "context+instruction"}})
        if len(longb) >= a.long_benign:
            break
    write_jsonl(out / "development" / "wrapped_benign.jsonl", sorted(records, key=lambda r: r["id"]))
    write_jsonl(out / "development" / "long_benign.jsonl", sorted(longb, key=lambda r: r["id"]))
    toks = [r["full_input_token_count"] for r in records]
    ltoks = [r["full_input_token_count"] for r in longb]
    manifest = {
        "scope": "control_panels_not_part_of_any_train_select_calibrate_confirm_role",
        "wrapped_benign": {"n": len(records), "by_strategy": dict(Counter(r["semantic_strategy"] for r in records)),
                           "by_split": dict(Counter(r["split"] for r in records)),
                           "not_constructed": {f"{k[0]}:{k[1]}": v for k, v in skipped.items()},
                           "token_median": sorted(toks)[len(toks) // 2], "token_max": max(toks)},
        "long_benign": {"n": len(longb), "token_median": sorted(ltoks)[len(ltoks) // 2], "token_max": max(ltoks)},
        "limitations": ["W(B) keeps the real wrapper; benign content inside a jailbreak wrapper may still read as an attempt.",
                        "PAIR/TAP and persuasion have no extractable wrapper and have no W(B) control.",
                        "Controls are constructed and not reviewed.",
                        "Long benign inputs are encyclopedic Dolly context plus question, not conversational."],
        "output_sha256": {str(p.relative_to(out)): sha256(p.read_bytes()) for p in sorted(out.rglob("*.jsonl"))},
        "source_data_manifest_sha256": sha256((Path(a.data) / "manifest.json").read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("output_sha256", "limitations")}, indent=1))


if __name__ == "__main__":
    main()
