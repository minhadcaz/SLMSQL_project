"""CPU-only environment and source records, usable on local, Kaggle and Colab."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from research.slm_sql.artifacts import atomic_json, digest, file_hash

ROOT = Path(__file__).resolve().parents[1]


def source_file_hash(path: str | Path) -> str:
    """Notebook output, execution counters and UI metadata are not source code."""
    path = Path(path)
    if path.suffix == ".ipynb":
        cells = json.loads(path.read_text(encoding="utf-8"))["cells"]
        return digest([{ "type": cell["cell_type"], "source": "".join(cell.get("source", []))}
                       for cell in cells])
    return file_hash(path)


def source_state() -> dict:
    def git(*args):
        result = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else None

    files = git("ls-files")
    paths = set([ROOT / p for p in files.splitlines()] if files else []) | set([
        p for folder in ("research", "training", "benchmarks", "configs", "evaluation", "core", "notebooks", "scripts")
        for p in (ROOT / folder).rglob("*")
    ])
    hashes = {
        str(p.relative_to(ROOT)): source_file_hash(p) for p in sorted(paths)
        if p.is_file() and p.suffix in (".py", ".yaml", ".toml", ".ipynb")
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256": digest(hashes), "files": hashes,
    }


def capture_runtime(directory: str | Path, label: str) -> dict:
    packages = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata["Name"]))
    payload = {
        "python": sys.version, "platform": platform.platform(),
        "packages": packages, "source": source_state(),
    }
    directory = Path(directory)
    atomic_json(directory / f"runtime.{label}.json", payload)
    (directory / f"requirements.{label}.resolved.txt").write_text(
        "".join(f"{name}=={version}\n" for name, version in packages.items()), encoding="utf-8"
    )
    return payload
