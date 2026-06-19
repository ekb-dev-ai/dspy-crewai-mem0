from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    dashboard_path = project_root / "app" / "telemetry" / "live_dashboard.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(dashboard_path)],
        check=True,
        cwd=str(project_root),
    )


if __name__ == "__main__":
    main()
