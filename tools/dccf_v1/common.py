"""Shared helpers for the DCCF v1 dataset build (standard library only)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.request import Request, urlopen

csv.field_size_limit(10**9)

JBB_REV = "886acc352a31533ffbcf4ef22c744658688086fc"
ART_REV = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
JBV_REV = "f949ca582fff13d396ac8fce59596afafb2b78d3"
DOLLY_REV = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
UA = "DCCF-dataset-builder/2.0"

SOURCES: dict[str, dict[str, str]] = {
    "jbb_harmful": {"url": f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/harmful-behaviors.csv",
                    "sha256": "4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1", "license": "MIT"},
    "jbb_benign": {"url": f"https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/{JBB_REV}/data/benign-behaviors.csv",
                   "sha256": "3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468", "license": "MIT"},
    "jbv_main": {"url": f"https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k/resolve/{JBV_REV}/JailBreakV_28K/JailBreakV_28K.csv",
                 "sha256": "72fb8beb9eb1c1a23745f4bd56430da9dff36ba42c6f5d699e68a47e14fc87c5", "license": "MIT"},
    "dolly": {"url": f"https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/{DOLLY_REV}/databricks-dolly-15k.jsonl",
              "sha256": "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec", "license": "CC-BY-SA-3.0"},
}

# (method directory, attack type, model, sha256, DCCF family)
ARTIFACTS = [
    ("DSN", "white_box", "llama-2-7b-chat-hf", "9bcdbe88b18e729ca9abf593d3e5141e7958aff0eb3a0e62ae92b8f64acbdcd9", "dsn"),
    ("DSN", "white_box", "vicuna-13b-v1.5", "ea0a5f7331baf595fb6e6925423f43367cbce8da9273ce93c6a03e2201e3373d", "dsn"),
    ("GCG", "white_box", "vicuna-13b-v1.5", "9eff8b6f78cae002737868ca8cdd886be5afcd1cbbbe4cf51ff837bc688b01b1", "gcg"),
    ("GCG", "white_box", "llama-2-7b-chat-hf", "f76cbaf899c79ca8eed6a6ed26d6f8789933c5595a9a5ee028e305dec4dd95f7", "gcg"),
    ("GCG", "transfer", "gpt-3.5-turbo-1106", "ff185b1dc73732d830e96efdb3f4a92294ee72e9c88a1800b1d11d4ecb3d4cdf", "gcg"),
    ("GCG", "transfer", "gpt-4-0125-preview", "af54543a06e3702256315517ee6e6a7b0edf182dd2c6d938216fa58cbea8de23", "gcg"),
    ("JBC", "manual", "vicuna-13b-v1.5", "1d5df5aa451c3d1bb8a4c048a7b2e1e6132e72f2970cb9d94ffc77f44e0edb39", "template"),
    ("JBC", "manual", "llama-2-7b-chat-hf", "1607a5a5b9f50abc7f2ff9e07b7257dfde7228b88cfdf73c20f5f56108ffdd23", "template"),
    ("PAIR", "black_box", "vicuna-13b-v1.5", "18b101386419a98fbdf45cb987069f83af3a40d5e0f7f0d32c4e9dc5ecc049b4", "pair_tap"),
    ("PAIR", "black_box", "gpt-3.5-turbo-1106", "49e007ce488e606cede7ffc8b022e35080b7f3870956e4cb2423d9b42749c78f", "pair_tap"),
    ("PAIR", "black_box", "gpt-4-0125-preview", "6dd214eee78184daf048ac8af166641724b49fd25bc831e5d7201469c162004d", "pair_tap"),
    ("PAIR", "black_box", "llama-2-7b-chat-hf", "02819296c5d992568d732d37813b2ce34624068a20f21025c04c04d52820b029", "pair_tap"),
    ("prompt_with_random_search", "black_box", "vicuna-13b-v1.5", "9d87a56659f617438feb5770bb2c8e4c2dbbd1bd62964f2c1a8775988017ac5d", "saa"),
    ("prompt_with_random_search", "black_box", "llama-2-7b-chat-hf", "45df98af6e1c0e35747f6b9c80cdc606e96c76a511e09a1a49ad5b00cfb51c0a", "saa"),
    ("prompt_with_random_search", "black_box", "gpt-3.5-turbo-1106", "d4a0691eda37b02dc948b6d6d08460f7d5cd3f15099121b388a58634ffb7bd76", "saa"),
    ("prompt_with_random_search", "black_box", "gpt-4-0125-preview", "4a0573dba6823c676cab99ed54b8191244080883948270fac9b75255b7c69206", "saa"),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_pinned(url: str, digest: str, cache_dir: Path) -> bytes:
    """Download once into cache_dir and verify the pinned SHA-256 every time."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / sha256(url.encode())[:16]
    if path.exists():
        data = path.read_bytes()
    else:
        with urlopen(Request(url, headers={"User-Agent": UA}), timeout=120) as response:
            data = response.read()
        path.write_bytes(data)
    if sha256(data) != digest:
        raise ValueError(f"Pinned hash mismatch for {url}: {sha256(data)}")
    return data


def canonical(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def loose(text: str) -> str:
    """Canonical text with punctuation removed, for duplicate detection."""
    return " ".join(re.sub(r"[^\w\s]", " ", canonical(text)).split())


def word_set(text: str) -> frozenset:
    return frozenset(loose(text).split())


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


MOJIBAKE = re.compile(r"[鈥馃鉂銆锟]|�")
CJK = re.compile(r"[一-鿿]")


def repair_mojibake(text: str) -> str | None:
    """Undo UTF-8-read-as-GBK damage in the JailbreakV CSV; None when it cannot be repaired."""
    if not MOJIBAKE.search(text):
        return text
    try:
        fixed = text.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return None
    return None if MOJIBAKE.search(fixed) or CJK.search(fixed) else fixed


def csv_rows(payload: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows), encoding="utf-8")


def stable_key(*parts: str) -> str:
    return sha256("\x1f".join(parts).encode())
