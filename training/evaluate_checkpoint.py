"""Greedy EX on the legacy calibration databases; run in its own GPU process."""

import argparse
import time
from pathlib import Path

from benchmarks.bird_loader import get_db_path, get_schema, load_bird_dev
from core.sql_extractor import extract_sql
from research.exec_runner import _exec_one
from research.legacy_protocol import split_legacy_dev
from research.reproducibility import capture_runtime
from research.slm_sql.artifacts import atomic_json
from research.slm_sql.backend import VLLMBackend
from training.prompt_template import format_sft_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()
    records = split_legacy_dev(load_bird_dev(args.data_dir), args.split_manifest)["calibration"]
    records = records[:args.limit] if args.limit else records
    settings = {
        "model": args.model, "revision": "main", "temperature": 0.0, "top_p": 1.0,
        "max_tokens": args.max_tokens, "max_model_len": args.max_model_len,
        "dtype": "float16", "gpu_memory_utilization": 0.85,
    }
    backend = VLLMBackend(settings)
    capture_runtime(args.output.parent, "pilot-evaluation")
    results, tokens = [], 0
    started = time.perf_counter()
    for record in records:
        messages = format_sft_messages(record["question"], get_schema(args.data_dir, record["db_id"]), "", evidence=record.get("evidence", ""))[:-1]
        completion = backend.generate([messages], n=1, seeds=[42])[0][0]
        tokens += completion["n_tokens"]
        try:
            sql = extract_sql(completion["raw_text"])
        except ValueError:
            sql = ""
        path = str(get_db_path(args.data_dir, record["db_id"]))
        gold = _exec_one(record["gold_sql"], path)
        if gold["fingerprint"] is None:
            raise ValueError("Calibration gold SQL failed to execute.")
        prediction = _exec_one(sql, path)
        results.append({
            "db_id": record["db_id"], "question": record["question"],
            "pred_sql": sql, "correct": prediction["fingerprint"] == gold["fingerprint"],
        })
    elapsed = time.perf_counter() - started
    atomic_json(args.output, {
        "split": "calibration_databases_only", "accuracy": sum(r["correct"] for r in results) / len(results),
        "questions": len(results), "records": results, "model_identity": backend.identity,
        "wall_time_s": elapsed, "output_tokens": tokens, "tokens_per_second": tokens / elapsed,
    })


if __name__ == "__main__":
    main()
