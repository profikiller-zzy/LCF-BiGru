#!/usr/bin/env python3
"""
Fetch public jailbreak prompt sources into a local raw-source cache.

This script downloads static artifacts only. It does not run attack algorithms.
Generated outputs are later normalized by build_prompt_library.py.
"""

from __future__ import annotations

import json
import base64
import time
import urllib.request
from pathlib import Path


PROMPT_ROOT = Path(__file__).resolve().parent
RAW = PROMPT_ROOT / "_raw_sources"

JBB_BASE = "https://raw.githubusercontent.com/JailbreakBench/artifacts/main/attack-artifacts"
FILES = {
    "jailbreakbench_artifacts/GCG/white_box/llama-2-7b-chat-hf.json":
        ("JailbreakBench/artifacts", "attack-artifacts/GCG/white_box/llama-2-7b-chat-hf.json", "main"),
    "jailbreakbench_artifacts/GCG/white_box/vicuna-13b-v1.5.json":
        ("JailbreakBench/artifacts", "attack-artifacts/GCG/white_box/vicuna-13b-v1.5.json", "main"),
    "jailbreakbench_artifacts/GCG/transfer/gpt-3.5-turbo-1106.json":
        ("JailbreakBench/artifacts", "attack-artifacts/GCG/transfer/gpt-3.5-turbo-1106.json", "main"),
    "jailbreakbench_artifacts/GCG/transfer/gpt-4-0125-preview.json":
        ("JailbreakBench/artifacts", "attack-artifacts/GCG/transfer/gpt-4-0125-preview.json", "main"),
    "jailbreakbench_artifacts/PAIR/black_box/gpt-3.5-turbo-1106.json":
        ("JailbreakBench/artifacts", "attack-artifacts/PAIR/black_box/gpt-3.5-turbo-1106.json", "main"),
    "jailbreakbench_artifacts/PAIR/black_box/gpt-4-0125-preview.json":
        ("JailbreakBench/artifacts", "attack-artifacts/PAIR/black_box/gpt-4-0125-preview.json", "main"),
    "jailbreakbench_artifacts/PAIR/black_box/llama-2-7b-chat-hf.json":
        ("JailbreakBench/artifacts", "attack-artifacts/PAIR/black_box/llama-2-7b-chat-hf.json", "main"),
    "jailbreakbench_artifacts/PAIR/black_box/vicuna-13b-v1.5.json":
        ("JailbreakBench/artifacts", "attack-artifacts/PAIR/black_box/vicuna-13b-v1.5.json", "main"),
    "advbench/llm_attacks_harmful_behaviors.csv":
        ("llm-attacks/llm-attacks", "data/advbench/harmful_behaviors.csv", "main"),
    "autodan/official_advbench_harmful_behaviors.csv":
        ("SheltonLiu-N/AutoDAN", "data/advbench/harmful_behaviors.csv", "main"),
}


def github_api_url(repo: str, repo_path: str, ref: str) -> str:
    quoted = "/".join(part.replace(" ", "%20") for part in repo_path.split("/"))
    return f"https://api.github.com/repos/{repo}/contents/{quoted}?ref={ref}"


def download(spec: tuple[str, str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    repo, repo_path, ref = spec
    url = github_api_url(repo, repo_path, ref)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "LCF-BiGru-data-builder",
            "Accept": "application/vnd.github+json",
        },
    )
    last_error = None
    data = b""
    for attempt in range(1, 6):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
            data = base64.b64decode("".join(payload["content"].split()))
            break
        except Exception as exc:  # network can be flaky for raw GitHub downloads
            last_error = exc
            time.sleep(2 * attempt)
    if not data and last_error is not None:
        raise last_error
    if not data:
        raise RuntimeError(f"Empty download: {url}")
    path.write_bytes(data)


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    results = {}
    for rel, url in FILES.items():
        out = RAW / rel
        download(url, out)
        results[rel] = {"repo": url[0], "path": url[1], "ref": url[2], "bytes": out.stat().st_size}
    summary = {"downloaded": results}
    with (RAW / "fetch_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
