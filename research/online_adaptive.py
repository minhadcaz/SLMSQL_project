"""Generate SQL online in rounds of four, with resumable policy evaluation.

Example (Kaggle GPU):
  python -m research.online_adaptive --model /kaggle/working/SLM-PRJ/results/lora/merged \
    --data-dir /kaggle/input/bird --policy beta --limit 50 \
    --offline-cache /kaggle/working/SLM-PRJ/results/candidates/exec_dev.jsonl

The offline comparison is diagnostic: distinct model draws cannot guarantee 5% agreement.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from research.slm_sql.prompts import generation_messages
from core.sql_extractor import extract_sql
from research.simulator import StoppingPolicy, simulate_policy


def _record_id(record: dict[str, Any]) -> tuple[str, str]:
    return record["db_id"], record["question"]


def _dataset_digest(records: list[dict[str, Any]]) -> str:
    items = [(r["db_id"], r["question"], r.get("gold_sql", "")) for r in records]
    return hashlib.sha256(json.dumps(items, ensure_ascii=False).encode("utf-8")).hexdigest()


def _save_checkpoint(path: Path, data: dict[str, Any]) -> None:
    """Replace the checkpoint atomically after every completed question."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _completion_logprob(completion: Any) -> float:
    from research.generation_metrics import completion_logprob
    return completion_logprob(completion)


def _difference(online: float, offline: float) -> float:
    """Relative difference, with absolute difference for a zero reference."""
    return abs(online - offline) / abs(offline) if offline else abs(online - offline)


def compare_offline(
    online_records: list[dict[str, Any]],
    cached_records: list[dict[str, Any]],
    policy: StoppingPolicy,
    db_path_fn,
    n_permutations: int = 100,
    batch_size: int = 4,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare matched questions to an offline cache; flag CSC as incomparable."""
    cache = {_record_id(record): record for record in cached_records}
    matched = [cache[_record_id(r)] for r in online_records if _record_id(r) in cache]
    if len(matched) != len(online_records):
        raise ValueError("Offline cache does not contain every completed online question.")
    offline = simulate_policy(
        matched, policy, db_path_fn, n_permutations=n_permutations,
        batch_size=batch_size, seed=seed,
    )
    n = len(online_records)
    online = {
        "ex_mean": sum(bool(r["correct"]) for r in online_records) / n if n else 0.0,
        "avg_samples": sum(r["n_samples_used"] for r in online_records) / n if n else 0.0,
        "avg_tokens": sum(r["n_tokens_used"] for r in online_records) / n if n else 0.0,
        "avg_seconds": sum(r["elapsed_seconds"] for r in online_records) / n if n else 0.0,
    }
    comparable = not any(r["action"] == "CALL_CSC" for r in online_records) and not offline["csc_calls"]
    metrics = ("ex_mean", "avg_samples", "avg_tokens", "avg_seconds")
    deviations = {name: round(_difference(online[name], offline[name]), 4)
                  if online[name] is not None and offline[name] is not None else None for name in metrics}
    return {
        "online": online,
        "offline": offline,
        "relative_deviation": deviations,
        "within_5_percent": comparable and n > 0 and all(d is not None and d <= 0.05 for d in deviations.values()),
        "comparable": comparable,
        "note": (
            "CSC is executed online but not in the offline policy simulator."
            if not comparable else "Independent model draws may differ from the cached draws."
        ),
        "seed": seed,
        "n_permutations": n_permutations,
    }


def _run_question(
    record: dict[str, Any], policy: StoppingPolicy, db_path_fn,
    llm: Any, sampling_params_class: Any, batch_size: int, max_k: int,
    max_tokens: int, temperature: float, top_p: float, seed: int,
    csc_request: Any = None,
) -> dict[str, Any]:
    """Generate and execute only the candidates needed for one question."""
    from research.exec_runner import _exec_one

    tokenizer = llm.get_tokenizer()
    db_path = str(db_path_fn(record["db_id"]))
    schema = record.get("schema", {})
    messages = generation_messages(record, schema if isinstance(schema, str) else json.dumps(schema, ensure_ascii=False))
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    seen: list[dict[str, Any]] = []
    action = "SAMPLE_MORE"
    generation_seconds = 0.0
    started = time.perf_counter()

    while len(seen) < max_k:
        count = min(batch_size, max_k - len(seen))
        params = sampling_params_class(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            n=count, seed=seed + len(seen), logprobs=1,
        )
        batch_started = time.perf_counter()
        outputs = llm.generate([prompt], params)
        batch_elapsed = time.perf_counter() - batch_started
        completions = outputs[0].outputs
        if len(completions) != count:
            raise RuntimeError(f"vLLM returned {len(completions)} completions; expected {count}.")
        generation_seconds += batch_elapsed

        for completion in completions:
            raw_text = completion.text
            try:
                sql = extract_sql(raw_text)
            except ValueError:
                sql = ""
            execution = _exec_one(sql, db_path)
            seen.append({
                "index": len(seen), "raw_text": raw_text, "sql": sql,
                "n_tokens": len(completion.token_ids),
                "avg_logprob": _completion_logprob(completion),
                "generation_seconds": batch_elapsed / count,
                **execution,
            })
        stop, action = policy.should_stop(seen, len(seen))
        if stop:
            break

    if len(seen) >= max_k and action == "SAMPLE_MORE":
        action = "STOP"

    selected_fp = policy.select(seen)
    selected = next((c for c in seen if c["fingerprint"] == selected_fp), None) if selected_fp else None
    selected_sql = selected["sql"] if selected else None
    csc_result = None
    csc_tokens = 0
    if action == "CALL_CSC":
        if csc_request is None:
            raise ValueError("CALL_CSC requires a CSC LoRA adapter (--csc-adapter).")
        from research.csc_lora import format_csc_prompt
        if not hasattr(policy, "select_csc_inputs"):
            raise ValueError("CALL_CSC policy must implement select_csc_inputs().")
        inputs = policy.select_csc_inputs(seen)
        if len(inputs) != 2:
            raise ValueError("CALL_CSC requires two distinct executable fingerprint groups.")
        csc_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": format_csc_prompt(
                record["question"], schema, inputs[0], inputs[1]
            )}], tokenize=False, add_generation_prompt=True,
        )
        csc_params = sampling_params_class(temperature=0.0, max_tokens=max_tokens)
        csc_started = time.perf_counter()
        csc_output = llm.generate([csc_prompt], csc_params, lora_request=csc_request)[0].outputs[0]
        generation_seconds += time.perf_counter() - csc_started
        csc_tokens = len(csc_output.token_ids)
        try:
            selected_sql = extract_sql(csc_output.text)
        except ValueError:
            selected_sql = ""
        csc_result = _exec_one(selected_sql, db_path)
        selected_fp = csc_result["fingerprint"]

    gold_sql = record.get("gold_sql", "")
    gold_fp = _exec_one(gold_sql, db_path)["fingerprint"] if gold_sql else None
    if gold_fp is None:
        raise ValueError(f"Gold execution failed for {record['db_id']}: {record['question']}")
    return {
        "question": record["question"], "db_id": record["db_id"],
        "gold_sql": gold_sql, "difficulty": record.get("difficulty", "unknown"),
        "seed": seed, "action": action, "candidates": seen,
        "n_samples_used": len(seen),
        "n_tokens_used": sum(c["n_tokens"] for c in seen) + csc_tokens,
        "generation_seconds": round(generation_seconds, 6),
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "selected_sql": selected_sql, "selected_fp": selected_fp,
        "correct": bool(gold_fp and selected_fp == gold_fp),
        "csc_result": csc_result,
    }


def run_online_adaptive(
    records: list[dict[str, Any]], policy: StoppingPolicy, db_path_fn,
    model_path: str, output_path: str | Path, *, policy_name: str = "custom",
    batch_size: int = 4, max_k: int = 32, max_tokens: int = 512,
    max_seq_len: int = 2048, temperature: float = 0.8, top_p: float = 0.95,
    seed: int = 42, csc_adapter_path: str | None = None,
    offline_records: list[dict[str, Any]] | None = None,
    n_permutations: int = 100, llm: Any = None,
    sampling_params_class: Any = None,
) -> dict[str, Any]:
    """Run online policy evaluation and checkpoint after each question."""
    if batch_size < 1 or max_k < 1 or max_tokens < 1 or n_permutations < 1:
        raise ValueError("batch_size, max_k, max_tokens, n_permutations must be positive.")
    if len({_record_id(r) for r in records}) != len(records):
        raise ValueError("Questions must have unique (db_id, question) pairs for resumption.")
    if hasattr(policy, "max_k"):
        max_k = min(max_k, policy.max_k)
    if hasattr(policy, "k"):
        max_k = min(max_k, policy.k)
    if not all(r.get("schema") for r in records):
        from core.schema_inspector import SchemaInspector

        inspector = SchemaInspector()
        schema_cache: dict[str, dict] = {}
        normalized_records = []
        for record in records:
            schema = record.get("schema")
            if not schema:
                db_id = record["db_id"]
                if db_id not in schema_cache:
                    schema_cache[db_id] = inspector.inspect(db_path_fn(db_id))
                schema = schema_cache[db_id]
            normalized_records.append(dict(record, schema=schema))
        records = normalized_records
    config = {
        "model_path": str(model_path), "policy": policy_name,
        "policy_config": {
            key: value for key, value in vars(policy).items()
            if isinstance(value, (str, int, float, bool, type(None)))
        },
        "csc_adapter_path": csc_adapter_path, "batch_size": batch_size, "max_k": max_k,
        "max_tokens": max_tokens, "max_seq_len": max_seq_len,
        "temperature": temperature, "top_p": top_p, "seed": seed,
        "n_permutations": n_permutations,
        "dataset_sha256": _dataset_digest(records),
    }
    output_path = Path(output_path)
    if output_path.exists():
        report = json.loads(output_path.read_text(encoding="utf-8"))
        if report["config"] != config:
            raise ValueError("Checkpoint config/dataset changed; choose a new --output path.")
        completed = report["records"]
        if len(completed) > len(records):
            raise ValueError("Checkpoint contains more questions than the current dataset.")
        if [_record_id(r) for r in completed] != [_record_id(r) for r in records[:len(completed)]]:
            raise ValueError("Checkpoint question order changed; choose a new --output path.")
    else:
        report = {"config": config, "records": []}
        completed = report["records"]

    if len(completed) < len(records):
        if llm is None or sampling_params_class is None:
            from vllm import LLM, SamplingParams
            llm = llm or LLM(
                model=model_path, dtype="float16", max_model_len=max_seq_len,
                gpu_memory_utilization=0.80, seed=seed,
                enable_lora=bool(csc_adapter_path),
            )
            sampling_params_class = sampling_params_class or SamplingParams
        csc_request = None
        if csc_adapter_path:
            from vllm.lora.request import LoRARequest
            csc_request = LoRARequest("csc", 1, csc_adapter_path)
        for index in range(len(completed), len(records)):
            result = _run_question(
                records[index], policy, db_path_fn, llm, sampling_params_class,
                batch_size, max_k, max_tokens, temperature, top_p,
                seed + index * 1000, csc_request,
            )
            completed.append(result)
            report["summary"] = _summary(completed)
            _save_checkpoint(output_path, report)

    report["summary"] = _summary(completed)
    if offline_records is not None:
        report["offline_comparison"] = compare_offline(
            completed, offline_records, policy, db_path_fn,
            n_permutations=n_permutations, batch_size=batch_size, seed=seed,
        )
    _save_checkpoint(output_path, report)
    return report


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-question online metrics without inventing results."""
    n = len(records)
    return {
        "n_questions": n,
        "ex_mean": round(sum(r["correct"] for r in records) / n, 4) if n else 0.0,
        "avg_samples": round(sum(r["n_samples_used"] for r in records) / n, 2) if n else 0.0,
        "avg_tokens": round(sum(r["n_tokens_used"] for r in records) / n, 2) if n else 0.0,
        "avg_seconds": round(sum(r["elapsed_seconds"] for r in records) / n, 4) if n else 0.0,
        "csc_calls": sum(r["action"] == "CALL_CSC" for r in records),
    }


def main() -> None:
    """Run on BIRD dev using a merged base model and optional CSC LoRA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Merged model checkpoint for vLLM")
    parser.add_argument("--data-dir", type=Path, required=True, help="BIRD directory with dev.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("fixed16", "beta", "margin", "router"), default="beta")
    parser.add_argument("--csc-adapter", help="CSC LoRA adapter; required for router")
    parser.add_argument("--offline-cache", type=Path, help="Optional 32-candidate exec JSONL")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-k", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n-permutations", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0, help="First N BIRD questions; 0 for all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.policy == "router" and not args.csc_adapter:
        parser.error("--policy router requires --csc-adapter for real CALL_CSC inference.")
    if args.max_k < 1 or args.limit < 0:
        parser.error("--max-k must be positive and --limit cannot be negative.")

    from benchmarks.bird_loader import get_db_path, get_schema, load_bird_dev
    from research.oracle_analysis import load_exec_results
    from research.policies.beta_stopping import BetaStoppingPolicy
    from research.policies.fixed_k import FixedKPolicy
    from research.policies.margin_entropy import MarginEntropyPolicy
    from research.policies.three_action_router import ThreeActionRouterPolicy

    policy = {
        "fixed16": FixedKPolicy(16),
        "beta": BetaStoppingPolicy(max_k=args.max_k),
        "margin": MarginEntropyPolicy(max_k=args.max_k),
        "router": ThreeActionRouterPolicy(max_k=args.max_k),
    }[args.policy]
    records = load_bird_dev(args.data_dir)
    if args.limit:
        records = records[:args.limit]
    for record in records:
        record["schema"] = get_schema(args.data_dir, record["db_id"])
    offline = load_exec_results(args.offline_cache) if args.offline_cache else None
    report = run_online_adaptive(
        records, policy, lambda db_id: get_db_path(args.data_dir, db_id),
        args.model, args.output, policy_name=args.policy, batch_size=args.batch_size,
        max_k=args.max_k, max_tokens=args.max_tokens, max_seq_len=args.max_seq_len,
        temperature=args.temperature, top_p=args.top_p, seed=args.seed,
        csc_adapter_path=args.csc_adapter, offline_records=offline,
        n_permutations=args.n_permutations,
    )
    print(json.dumps({key: report[key] for key in ("config", "summary")}, indent=2))
    if "offline_comparison" in report:
        print(json.dumps(report["offline_comparison"], indent=2))


if __name__ == "__main__":
    main()
