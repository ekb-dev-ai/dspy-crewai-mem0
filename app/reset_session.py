from __future__ import annotations

import shutil
from pathlib import Path

DATA_DIR = Path("./data")
CHROMA_DIR = DATA_DIR / "chroma_db"
LIVE_METRICS = DATA_DIR / "live_metrics.json"
RUNS_LOG = DATA_DIR / "runs.jsonl"


def clear_session_data() -> None:
    """Remove persisted memory, telemetry, and run logs for a clean demo start."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    for path in (LIVE_METRICS, RUNS_LOG):
        if path.exists():
            path.unlink()

    print("Session data cleared:")
    print(f"  - {CHROMA_DIR} (Mem0 / Chroma memories)")
    print(f"  - {LIVE_METRICS} (live dashboard telemetry)")
    print(f"  - {RUNS_LOG} (pipeline run logs)")


def main() -> None:
    clear_session_data()


if __name__ == "__main__":
    main()
