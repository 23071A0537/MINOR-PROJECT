from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    target = project_root / "src" / "qids" / "explainability" / "explainability_dashboard.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(target), *sys.argv[1:]]
    raise SystemExit(subprocess.call(cmd, cwd=project_root))


if __name__ == "__main__":
    main()
