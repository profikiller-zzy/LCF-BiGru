"""Build a length-matched control panel for the manipulation and risk branches.

For every jailbreak J in train/select we create two controls whose formatted token count equals
J's (within a tolerance): H_pad = neutral filler + its exact H, and B_pad = the same filler + its B.
The filler is a Dolly-15k context passage (neutral, encyclopedic) chosen deterministically from the
J id. Controls carry no manipulation element, so any response separating J from H_pad is not explained
by length. They are NOT matched in style: the filler is encyclopedic prose, whereas real wrappers are
role-play, suffixes, persuasion and so on. The output is a separate panel, never mixed into the
train/select/calibrate/confirm roles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import DOLLY_REV, SOURCES, canonical, fetch_pinned, read_jsonl, sha256, write_jsonl  # noqa: E402
from stage1_select import BAD_WORDS  # noqa: E402

TOL = 3
TEMPLATE = "{filler}\n\n{request}"


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\[\d+\]", "", text).strip())
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dccf_v1")
    ap.add_argument("--build", default="artifacts/dccf_v1_build")
    ap.add_argument("--out", default="data/dccf_v1_length_controls")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    from transformers import AutoTokenizer
    from dual_concept.backbone import render_chat_input
    tok = AutoTokenizer.from_pretrained(str(Path(args.build) / "qwen_tokenizer"))

    def ntok(text: str) -> int:
        return len(tok(render_chat_input(tok, text), add_special_tokens=False)["input_ids"])

    lines = fetch_pinned(SOURCES["dolly"]["url"], SOURCES["dolly"]["sha256"], Path(args.build) / "raw").decode().splitlines()
    pool = []
    for n, line in enumerate(lines):
        r = json.loads(line)
        ctx = r["context"].strip()
        if r["category"] in ("closed_qa", "information_extraction", "summarization") and len(ctx) > 400 \
                and not BAD_WORDS.search(ctx) and not re.search(r"[^\x00-\x7f]", ctx):
            pool.append({"row": n, "sents": sentences(ctx)})
    pool = [p for p in pool if len(p["sents"]) >= 4]
    print("filler passages", len(pool))

    records, pairs, skipped = [], [], []
    for split in ("train", "select"):
        rows = read_jsonl(Path(args.data) / "development" / f"{split}.jsonl")
        by_id = {r["id"]: r for r in rows}
        for j in rows:
            if j["kind"] != "jailbreak":
                continue
            h, b = by_id[j["original_id"]], by_id[j["group_id"] + "_b"]
            target = j["full_input_token_count"]
            start = int(hashlib.sha256(("fill:" + j["id"]).encode()).hexdigest(), 16)
            ok = None
            for attempt in range(60):          # next passage if this one cannot reach the target
                p = pool[(start + attempt) % len(pool)]
                sents = list(p["sents"])
                passages = [p["row"]]
                if attempt >= 20:               # long targets: append a second neutral passage
                    q = pool[(start + attempt + 7) % len(pool)]
                    sents += q["sents"]; passages.append(q["row"])
                best = None
                for k in range(1, len(sents) + 1):
                    cand = " ".join(sents[:k])
                    nh = ntok(TEMPLATE.format(filler=cand, request=h["text"]))
                    nb = ntok(TEMPLATE.format(filler=cand, request=b["text"]))
                    # both controls must land within tolerance of the target
                    if abs(nh - target) <= TOL and abs(nb - target) <= TOL:
                        best = cand; break
                    if min(nh, nb) > target + TOL:
                        break
                if best:
                    ok = (passages, best); break
            if not ok:
                skipped.append(j["id"]); continue
            passages, filler = ok
            for kind, src in (("harmful", h), ("safe", b)):
                text = TEMPLATE.format(filler=filler, request=src["text"])
                rid = f"{j['id']}__pad_{'h' if kind == 'harmful' else 'b'}"
                records.append({
                    "id": rid, "text": text, "kind": kind, "label": 1 if kind == "harmful" else 0,
                    "core_group": "length_control", "group_id": j["group_id"], "split": split,
                    "attack_family": None, "original_id": None, "exposure": "dccf_v1_length_control",
                    "label_status": "constructed_control_not_reviewed",
                    "matched_to_jailbreak_id": j["id"], "base_id": src["id"],
                    "full_input_token_count": ntok(text), "target_token_count": target,
                    "source": {"filler_dolly_rows": passages, "filler_dolly_revision": DOLLY_REV, "template": TEMPLATE}})
            pairs.append({"jailbreak_id": j["id"], "harmful_pad_id": f"{j['id']}__pad_h",
                          "benign_pad_id": f"{j['id']}__pad_b", "split": split, "goal_cluster_id": j["group_id"]})
    for split in ("train", "select"):
        write_jsonl(out / "development" / f"{split}.jsonl", sorted((r for r in records if r["split"] == split), key=lambda r: r["id"]))
    write_jsonl(out / "pairs.jsonl", sorted(pairs, key=lambda p: p["jailbreak_id"]))
    diffs = [r["full_input_token_count"] - r["target_token_count"] for r in records]
    manifest = {
        "scope": "length_control_panel_not_part_of_any_train_select_calibrate_confirm_role",
        "construction": "neutral Dolly context passage prepended to H and to B so formatted token count matches each J within tolerance",
        "tolerance_tokens": TOL, "records": len(records), "pairs": len(pairs), "skipped_jailbreaks": skipped,
        "max_abs_token_diff": max(abs(d) for d in diffs), "mean_abs_token_diff": sum(abs(d) for d in diffs) / len(diffs),
        "limitations": ["Filler is encyclopedic prose; it matches length, not the style of real wrappers.",
                        "Padding may itself shift hidden states in ways unrelated to manipulation.",
                        "Controls are constructed, not reviewed."],
        "output_sha256": {str(p.relative_to(out)): sha256(p.read_bytes()) for p in sorted(out.rglob("*.jsonl"))},
        "source_data_manifest_sha256": sha256((Path(args.data) / "manifest.json").read_bytes()),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("output_sha256", "limitations")}, indent=1))


if __name__ == "__main__":
    main()
