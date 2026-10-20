"""Generate up to 32 legacy candidates once, with resumable checked caches."""

import argparse
import json
import logging
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from core.sql_extractor import extract_sql
from research.generation_metrics import has_answer_format
from research.slm_sql.artifacts import Journal, atomic_json, digest, file_hash
from research.slm_sql.backend import VLLMBackend
from research.slm_sql.prompts import PROMPT_VERSION, generation_messages

logger = logging.getLogger(__name__)


def generate_candidates(
    model_path: str, prompts: list[dict[str, Any]], n_candidates: int = 32,
    temperature: float = 0.8, top_p: float = 0.95, max_tokens: int = 512,
    max_seq_len: int = 8192, seed: int = 42, batch_size: int | None = None,
    *, cache_path: str | Path | None = None, backend_factory=VLLMBackend,
) -> list[dict[str, Any]]:
    """Cache each completed question; reuse needs neither CUDA nor model loading."""
    if not 1 <= n_candidates <= 32:
        raise ValueError("Legacy experiments require 1–32 candidates per question.")
    if not 0 < max_tokens < max_seq_len or not 0 < top_p <= 1 or temperature <= 0:
        raise ValueError("Invalid generation context or sampling settings.")
    batch_size = batch_size or 4
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    settings = {
        "model": model_path, "revision": "main", "temperature": temperature,
        "top_p": top_p, "max_tokens": max_tokens, "max_model_len": max_seq_len,
        "dtype": "float16", "gpu_memory_utilization": 0.85,
    }
    inputs = [dict(p, id=p.get("id") or digest([p["db_id"], p["question"], p.get("gold_sql")])) for p in prompts]
    spec = {
        "version": 3, "prompt": PROMPT_VERSION, "input": digest(inputs),
        "implementation": {name: file_hash(Path(__file__).resolve().parents[1] / name)
                           for name in ("research/candidate_generator.py", "research/generation_metrics.py",
                                        "research/slm_sql/backend.py", "research/slm_sql/prompts.py",
                                        "core/sql_extractor.py")},
        "settings": settings, "n_candidates": n_candidates, "seed": seed,
        "batch_size": batch_size,
    }
    context = nullcontext(None) if cache_path else tempfile.TemporaryDirectory(prefix="slm-candidates-")
    with context as temporary:
        path = Path(cache_path) if cache_path else Path(temporary) / "candidates.jsonl"
        with Journal(path, spec, [p["id"] for p in inputs]) as journal:
            if journal.complete:
                return journal.records
            backend = None
            for start in range(len(journal.records), len(inputs), batch_size):
                batch = inputs[start:start + batch_size]
                messages = [generation_messages(p, p["schema"] if isinstance(p["schema"], str)
                            else json.dumps(p["schema"], ensure_ascii=False)) for p in batch]
                if backend is None:
                    backend = backend_factory(settings, frozen_identity=journal.manifest.get("model_identity"))
                    if journal.manifest.get("packages") not in (None, backend.versions):
                        raise ValueError("Generation package versions changed during resume.")
                    journal.manifest.update(model_identity=backend.identity, packages=backend.versions)
                    atomic_json(journal.manifest_path, journal.manifest)
                seeds = [(seed + int(digest(p["id"])[:8], 16)) % 2**31 for p in batch]
                started = time.perf_counter()
                outputs = backend.generate(messages, n=n_candidates, seeds=seeds)
                elapsed = time.perf_counter() - started
                if len(outputs) != len(batch) or any(len(c) != n_candidates for c in outputs):
                    raise RuntimeError("Backend returned an incomplete candidate batch.")
                for record, completions, request_seed in zip(batch, outputs, seeds):
                    candidates = []
                    for index, completion in enumerate(completions):
                        try:
                            sql, parse_error = extract_sql(completion["raw_text"]), None
                        except ValueError as exc:
                            sql, parse_error = "", str(exc)
                        candidates.append({
                            **completion, "index": index, "sql": sql, "parse_error": parse_error,
                            "format_error": not has_answer_format(completion["raw_text"]),
                            "seed": request_seed,
                            "schema_size": len(json.dumps(record["schema"], ensure_ascii=False)),
                            "generation_seconds": elapsed / len(batch) / n_candidates,
                            "timing_basis": "amortized_cached_batch",
                        })
                    journal.append({**record, "candidates": candidates})
            journal.finish()
            return journal.records


def save_candidates(records: list[dict[str, Any]], output_path: str | Path) -> None:
    from research.slm_sql.artifacts import write_jsonl
    write_jsonl(output_path, records)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    from research.slm_sql.artifacts import completed_artifact, read_jsonl
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    return completed_artifact(path)[0] if sidecar.exists() else read_jsonl(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from benchmarks.bird_loader import load_bird_dev, get_schema
    prompts = [{**r, "schema": get_schema(args.data_dir, r["db_id"])} for r in load_bird_dev(args.data_dir)]
    results = generate_candidates(
        args.model, prompts, args.n_candidates, seed=args.seed, batch_size=args.batch_size,
        max_seq_len=args.max_model_len, max_tokens=args.max_tokens, cache_path=args.output,
    )
    print(json.dumps({"questions": len(results), "cache": str(args.output)}))


if __name__ == "__main__":
    main()
