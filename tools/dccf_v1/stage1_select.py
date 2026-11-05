"""Stage 1: select 400 goal clusters, J candidates and a benign calibration pool.

Deterministic, standard library only. Output goes to a build directory; nothing
here assigns labels beyond source candidates. Reviews happen in later stages.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import (ARTIFACTS, ART_REV, SOURCES, canonical, csv_rows, fetch_pinned, jaccard, loose,
                    read_jsonl, repair_mojibake, sha256, stable_key, word_set, write_jsonl)

JBV_EXCLUDED = {"Child Abuse Content", "Health Consultation", "Political Sensitivity",
                "Government Decision", "Tailored Unlicensed Advice"}
JBV_GOALS = 300
J_CANDIDATES = 3          # candidates reviewed per goal; spares replace rejections
CAL_CANDIDATES = 280      # reviewed benign calibration candidates (target 200 accepted)
ROLE_PATTERN = ["train"] * 14 + ["select"] * 3 + ["confirm"] * 3   # 70/15/15 per stratum


def legacy_texts(data_root: Path) -> set[str]:
    texts = set()
    for name in ("train", "select", "calibrate", "diagnose"):
        path = data_root / "development" / f"{name}.jsonl"
        if path.exists():
            texts.update(canonical(r["text"]) for r in read_jsonl(path))
    return texts


def assign_roles(goals: list[dict]) -> None:
    """Stratified by (source, category): position in a hash-sorted stratum picks the role."""
    strata = defaultdict(list)
    for goal in goals:
        strata[(goal["source"], goal["category"])].append(goal)
    offset = 0
    for key in sorted(strata):
        members = sorted(strata[key], key=lambda g: stable_key("role", g["H"]["text"]))
        for i, goal in enumerate(members):
            goal["role"] = ROLE_PATTERN[(i + offset) % len(ROLE_PATTERN)]
        offset += len(members)   # rotate so small strata do not always start with train


def jbb_goals(cache: Path, art_cache: Path, legacy: set[str]) -> list[dict]:
    harmful = csv_rows(fetch_pinned(SOURCES["jbb_harmful"]["url"], SOURCES["jbb_harmful"]["sha256"], cache))
    benign = csv_rows(fetch_pinned(SOURCES["jbb_benign"]["url"], SOURCES["jbb_benign"]["sha256"], cache))
    benign_by_index = {int(r["Index"]): r for r in benign}
    by_index = {}
    for method, kind, model, digest, family in ARTIFACTS:
        url = f"https://raw.githubusercontent.com/JailbreakBench/artifacts/{ART_REV}/attack-artifacts/{method}/{kind}/{model}.json"
        data = json.loads(fetch_pinned(url, digest, art_cache))
        for row in data["jailbreaks"]:
            by_index.setdefault(int(row["index"]), []).append((family, method, kind, model, row))
    goals, usage = [], Counter()
    for row in sorted(harmful, key=lambda r: int(r["Index"])):
        index = int(row["Index"])
        if canonical(row["Goal"]) in legacy:
            continue
        pool = {}
        for family, method, kind, model, art in by_index.get(index, []):
            prompt = (art.get("prompt") or "").strip()
            if not prompt or canonical(art["goal"]) != canonical(row["Goal"]) or art["behavior"] != row["Behavior"]:
                continue
            entry = pool.setdefault(canonical(prompt), {"text": prompt, "family": family, "lineage": f"jbb_{method.lower()}",
                                                         "sources": []})
            entry["sources"].append({"method": method, "attack_type": kind, "model": model, "artifact_index": index,
                                     "source_attack_success": art.get("jailbroken")})
        chosen, families = [], set()
        for _ in range(J_CANDIDATES):
            options = [e for e in pool.values() if e not in chosen]
            fresh = [e for e in options if e["family"] not in families] or options
            if not fresh:
                break
            fresh.sort(key=lambda e: (usage[e["family"]], stable_key("j", row["Goal"], e["text"])))
            pick = fresh[0]
            chosen.append(pick); families.add(pick["family"]); usage[pick["family"]] += 1
        b = benign_by_index.get(index)
        goals.append({
            "source": "JBB", "category": row["Category"],
            "H": {"text": row["Goal"].strip(), "source_dataset": "JBB-Behaviors", "source_revision": "886acc352a31533ffbcf4ef22c744658688086fc",
                  "source_row_id": str(index), "behavior": row["Behavior"], "source_tag": row["Source"]},
            "B_candidate": None if b is None else {"text": b["Goal"].strip(), "source_row_id": str(b["Index"]),
                                                   "behavior": b["Behavior"], "category": b["Category"]},
            "J": chosen})
    return goals


def jbv_goals(cache: Path, legacy: set[str], taken: list[str]) -> list[dict]:
    rows = csv_rows(fetch_pinned(SOURCES["jbv_main"]["url"], SOURCES["jbv_main"]["sha256"], cache))
    pool = defaultdict(lambda: {"T": {}, "P": {}, "policy": None, "origin": None, "ids": []})
    for r in rows:
        if r["format"] not in ("Template", "Persuade") or r["policy"] in JBV_EXCLUDED:
            continue
        h, j = repair_mojibake(r["redteam_query"]), repair_mojibake(r["jailbreak_query"])
        if not h or not j:
            continue
        h, j = h.strip(), j.strip()
        if not 25 <= len(h) <= 300 or canonical(h) in legacy:
            continue
        entry = pool[canonical(h)]
        entry.update(text=h, policy=r["policy"], origin=r["from"])
        entry["ids"].append(r["id"])
        if r["format"] == "Template":
            index = j.lower().find(h.lower())
            if index < 0:
                continue
            skeleton = re.sub(re.escape(h), "{H}", j[:index] + "{H}" + j[index + len(h):], flags=re.I)
            lineage = "jbv_tpl_" + sha256(skeleton.encode())[:8]
            entry["T"].setdefault(lineage, (j, r["id"]))
        else:
            entry["P"].setdefault(canonical(j), (j, r["id"]))
    # near-duplicate removal against JBB goals and among themselves
    accepted_words = [word_set(t) for t in taken]
    candidates = []
    for key in sorted(pool, key=lambda k: stable_key("jbvh", k)):
        e = pool[key]
        if not (e["T"] or e["P"]):
            continue
        words = word_set(e["text"])
        if any(jaccard(words, w) >= 0.8 for w in accepted_words):
            continue
        accepted_words.append(words)
        candidates.append(e)
    # round robin by policy, preferring a persuade-capable goal on every second pick
    by_policy = defaultdict(list)
    for e in candidates:
        by_policy[e["policy"]].append(e)
    selected, order = [], sorted(by_policy)
    while len(selected) < JBV_GOALS and any(by_policy.values()):
        for policy in order:
            if by_policy[policy] and len(selected) < JBV_GOALS:
                with_p = [e for e in by_policy[policy] if e["P"]]
                want_p = len(selected) % 2 == 0 and with_p
                pick = (with_p if want_p else by_policy[policy])[0]
                by_policy[policy].remove(pick); selected.append(pick)
    usage, goals = Counter(), []
    for e in selected:
        chosen = []
        if e["P"]:
            text, rid = sorted(e["P"].values(), key=lambda v: stable_key("p", e["text"], v[0]))[0]
            chosen.append({"text": text, "family": "persuade", "lineage": "jbv_persuade",
                           "sources": [{"method": "Persuade", "source_row_id": rid}]})
        lineages = sorted(e["T"], key=lambda l: (usage[l], stable_key("t", e["text"], l)))
        for lineage in lineages:
            if len(chosen) >= J_CANDIDATES:
                break
            text, rid = e["T"][lineage]
            chosen.append({"text": text, "family": "template", "lineage": lineage,
                           "sources": [{"method": "JailbreakV-28K Template", "source_row_id": rid}]})
            usage[lineage] += 1
        goals.append({"source": "JBV", "category": e["policy"],
                      "H": {"text": e["text"], "source_dataset": "JailBreakV-28k", "source_revision": "f949ca582fff13d396ac8fce59596afafb2b78d3",
                            "source_row_id": e["ids"][0], "origin": e["origin"]},
                      "B_candidate": None, "J": chosen})
    return goals


BAD_WORDS = re.compile(r"\b(kill|murder|weapon|gun|bomb|drug|cocaine|heroin|sex\w*|porn\w*|nude|hack\w*|steal\w*|suicid\w*|rape|"
                       r"abuse\w*|racist|terror\w*|poison\w*|explosive\w*|cheat\w*|illegal|fraud\w*|scam\w*|violen\w*|nazi\w*|hate|"
                       r"fuck\w*|shit|bitch|abortion|trump|biden|religio\w*|gay|trans\w*|vagina|penis|alcohol\w*|marijuana|weed)\b", re.I)


def calibration_pool(cache: Path, blocked: set[str]) -> list[dict]:
    lines = fetch_pinned(SOURCES["dolly"]["url"], SOURCES["dolly"]["sha256"], cache).decode("utf-8").splitlines()
    rows, seen = [], set()
    for number, line in enumerate(lines):
        r = json.loads(line)
        text = r["instruction"].strip()
        if (r["context"].strip() or r["category"] not in ("open_qa", "general_qa", "brainstorming", "creative_writing")
                or not 25 <= len(text) <= 300 or BAD_WORDS.search(text) or canonical(text) in blocked or canonical(text) in seen):
            continue
        seen.add(canonical(text))
        rows.append({"text": text, "source_row_id": str(number), "dolly_category": r["category"]})
    rows.sort(key=lambda r: stable_key("cal", r["text"]))
    # stratify over dolly categories
    per = defaultdict(list)
    for r in rows:
        per[r["dolly_category"]].append(r)
    out = []
    while len(out) < CAL_CANDIDATES:
        for name in sorted(per):
            if per[name] and len(out) < CAL_CANDIDATES:
                out.append(per[name].pop(0))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="artifacts/dccf_v1_build")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    build = Path(args.build)
    cache = build / "raw"
    legacy = legacy_texts(Path(args.data))
    jbb = jbb_goals(cache, cache, legacy)
    jbv = jbv_goals(cache, legacy, [g["H"]["text"] for g in jbb])
    goals = jbb + jbv
    for goal in goals:
        goal["goal_id"] = "g_" + sha256(canonical(goal["H"]["text"]).encode())[:12]
    assert len({g["goal_id"] for g in goals}) == len(goals)
    assign_roles(goals)
    goals.sort(key=lambda g: (g["source"], g["goal_id"]))
    blocked = legacy | {canonical(g["H"]["text"]) for g in goals} | {canonical(g["B_candidate"]["text"]) for g in goals if g["B_candidate"]}
    calibration = calibration_pool(cache, blocked)
    write_jsonl(build / "stage1" / "goals.jsonl", goals)
    write_jsonl(build / "stage1" / "calibration_candidates.jsonl", calibration)
    summary = {"goals": len(goals), "by_source": Counter(g["source"] for g in goals), "by_role": Counter(g["role"] for g in goals),
               "j_candidates": sum(len(g["J"]) for g in goals), "goals_without_j": sum(1 for g in goals if not g["J"]),
               "j_by_family": Counter(j["family"] for g in goals for j in g["J"]),
               "role_by_family": Counter(f'{g["role"]}/{j["family"]}' for g in goals for j in g["J"]),
               "calibration_candidates": len(calibration)}
    print(json.dumps({k: (dict(v) if isinstance(v, Counter) else v) for k, v in summary.items()}, default=str, indent=1))


if __name__ == "__main__":
    main()
