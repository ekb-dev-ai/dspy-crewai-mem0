from __future__ import annotations

import json
import os
import time
from pathlib import Path

RUNS_PATH = Path("./data/runs.jsonl")


def ensure_dirs() -> None:
    Path("./data").mkdir(exist_ok=True)
    Path("./data/artifacts").mkdir(parents=True, exist_ok=True)
    Path("./data/chroma_db").mkdir(parents=True, exist_ok=True)


def rough_token_count(text: str) -> int:
    # Lightweight approximation for local demos without usage APIs.
    return max(1, len(text) // 4)


def log_run(record: dict) -> None:
    ensure_dirs()
    with RUNS_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=True) + "\n")


def now_ts() -> float:
    return time.time()


def file_exists(path: str) -> bool:
    return os.path.exists(path)
