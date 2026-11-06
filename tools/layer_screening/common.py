"""Shared loading and grouping helpers for the layer-screening experiments.

Reads the frozen state cache written by ``python -m dual_concept extract`` and the
reviewed dataset. Nothing here fits on select/confirm/calibrate data unless a caller
asks for it explicitly.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dual_concept.backbone import StateCache  # noqa: E402
from dual_concept.data import read_jsonl  # noqa: E402

LAYERS = tuple(range(1, 28))      # default for the 28-layer Qwen2.5-7B run; set_layers() overrides per backbone


def set_layers(layers):
    global LAYERS
    LAYERS = tuple(int(x) for x in layers)
    return LAYERS
MIN_CELL_GROUPS = 15      # cells with fewer goal groups are flagged, never ranked


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strategy_of(row: dict[str, Any]) -> str | None:
    """Fine-grained strategy: a template lineage for template attacks, else the family."""
    if row["kind"] != "jailbreak":
        return None
    return row["template_lineage_id"] if row["attack_family"] == "template" else row["attack_family"]


class Panel:
    """Train/select rows with aligned frozen states, shape [N, 27, 3584]."""

    def __init__(self, data_root: str | Path, cache_root: str | Path, splits: Sequence[str] = ("train", "select")):
        data_root, cache_root = Path(data_root), Path(cache_root)
        protocol = json.loads((cache_root / "protocol.json").read_text(encoding="utf-8"))
        set_layers(protocol["candidate_layers"])
        self.protocol = protocol
        cache = StateCache(cache_root / "rows")
        self.rows: list[dict[str, Any]] = []
        blocks = []
        for split in splits:
            rows = read_jsonl(data_root / "development" / f"{split}.jsonl")
            states = []
            for row in rows:
                state = cache.load(row["id"], row["text"], protocol,
                                   expected_shape=(len(LAYERS), int(protocol["hidden_size"])))
                if state is None:
                    raise FileNotFoundError(f"Missing cached state for {row['id']}")
                states.append(state)
            self.rows.extend(rows)
            blocks.append(np.stack(states).astype(np.float32))
        self.states = np.concatenate(blocks)
        self.index = {r["id"]: i for i, r in enumerate(self.rows)}
        self.split = np.array([r["split"] for r in self.rows])
        self.kind = np.array([r["kind"] for r in self.rows])
        self.group = np.array([r["group_id"] for r in self.rows])
        self.family = np.array([r["attack_family"] or "" for r in self.rows])
        self.strategy = np.array([strategy_of(r) or "" for r in self.rows])

    def mask(self, split: str | None = None, kind: str | None = None) -> np.ndarray:
        m = np.ones(len(self.rows), dtype=bool)
        if split:
            m &= self.split == split
        if kind:
            m &= self.kind == kind
        return m

    def goal_groups(self, split: str) -> list[str]:
        return sorted({g for g, s, k in zip(self.group, self.split, self.kind) if s == split and k == "harmful"})


def load_pairs(data_root: str | Path) -> tuple[list[dict], list[dict]]:
    """All-role risk pairs (H-B) and manipulation pairs (J-H)."""
    root = Path(data_root)
    risk = read_jsonl(root / "annotation" / "pairs_risk_all_roles.jsonl")
    manip = read_jsonl(root / "pairs" / "manipulation.jsonl")
    return risk, manip


def grouped_folds(groups: Sequence[str], n_folds: int, seed: int) -> dict[str, int]:
    """Deterministic assignment of goal groups to folds."""
    ordered = sorted(set(groups), key=lambda g: hashlib.sha256(f"{seed}:{g}".encode()).hexdigest())
    return {g: i % n_folds for i, g in enumerate(ordered)}


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC with tie correction; NaN when one class is missing."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_ci(values_by_group: dict[str, float], draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile CI of the mean over goal groups."""
    vals = np.array(list(values_by_group.values()), dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(draws, len(vals)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def CAND():
    return list(LAYERS)


def LI_ALL():
    return [l - 1 for l in LAYERS]
