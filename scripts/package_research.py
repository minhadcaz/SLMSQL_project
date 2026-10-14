"""Package adapter, candidate cache, final reports and a provenance model card."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def package_research(
    adapter_path: str | Path,
    candidate_cache: str | Path,
    final_results_dir: str | Path,
    output_dir: str | Path,
    *,
    base_model: str,
    base_license: str = "verify upstream model card",
    data_license: str = "verify dataset terms before distribution",
    seed: int = 42,
) -> dict[str, Any]:
    """Copy existing artifacts into a new folder and record every file hash."""
    adapter = Path(adapter_path).resolve(strict=True)
    cache = Path(candidate_cache).resolve(strict=True)
    results = Path(final_results_dir).resolve(strict=True)
    output = Path(output_dir).resolve()
    if not adapter.is_dir() or not cache.is_file() or not results.is_dir():
        raise ValueError("Adapter must be a directory, cache a file, and results a directory.")
    if not any(results.glob("*.json")):
        raise ValueError("Results directory has no final JSON reports.")
    if output.exists():
        raise FileExistsError(f"Package already exists: {output}")
    if output == adapter or output == results or output.is_relative_to(adapter) or output.is_relative_to(results):
        raise ValueError("Output cannot be nested inside an input directory.")

    output.mkdir(parents=True)
    try:
        for source in sorted(adapter.rglob("*")):
            if source.is_symlink():
                raise ValueError(f"Symlinks are not packaged: {source}")
            if source.is_file():
                _copy_file(source, output / "adapter" / source.relative_to(adapter))
        _copy_file(cache, output / "candidates" / cache.name)
        for source in sorted(results.glob("*.json")):
            _copy_file(source, output / "results" / source.name)
        manifest = {
            "base_model": base_model,
            "base_license": base_license,
            "data_license": data_license,
            "seed": seed,
            "source_paths": {
                "adapter": str(adapter), "candidate_cache": str(cache),
                "results": str(results),
            },
            "files": {
                str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in sorted(output.rglob("*")) if path.is_file()
            },
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (output / "MODEL_CARD.md").write_text(
            "# Text-to-SQL research adapter\n\n"
            f"Base model: `{base_model}`. Seed: `{seed}`.\n\n"
            "## Contents\n\nAdapter, executed candidate cache, per-split final results, and a SHA-256 manifest. "
            "Use the recorded configuration in the source JSON before comparing EX or latency.\n\n"
            "## Intended use and limits\n\nResearch on SQLite Text-to-SQL. SQL may execute yet answer the "
            "wrong question; inspect SQL and results before use. Synthetic business data does not "
            "represent real customers. Candidate cache may contain original benchmark questions and "
            "SQL; review dataset redistribution terms before publishing.\n\n"
            "## Licenses and access\n\n"
            f"Base model: {base_license}. Dataset/cache: {data_license}. "
            "These strings are provided by the packager, not verified grants of rights. "
            "Check upstream model, adapter, and dataset licenses before redistribution.\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(output)
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--final-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--base-license", default="verify upstream model card")
    parser.add_argument("--data-license", default="verify dataset terms before distribution")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = package_research(
        args.adapter, args.candidate_cache, args.final_results, args.output,
        base_model=args.base_model, base_license=args.base_license,
        data_license=args.data_license, seed=args.seed,
    )
    print(json.dumps({"output": str(args.output), "files": len(report["files"])}, indent=2))


if __name__ == "__main__":
    main()
