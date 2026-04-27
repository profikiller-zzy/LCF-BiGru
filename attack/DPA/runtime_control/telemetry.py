import json
import os
from typing import Iterable


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_json(path: str, payload) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
