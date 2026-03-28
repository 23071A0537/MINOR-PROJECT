import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def ensure_exists(paths: Iterable[Path], label: str) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        missing_str = "\n".join(missing)
        raise FileNotFoundError(f"Missing {label}:\n{missing_str}")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    completed = subprocess.run(cmd, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed ({completed.returncode}): {' '.join(cmd)}")
