"""Generate N=32 SQL candidates per question using vLLM, save as JSONL.

Candidates are generated ONCE and reused for all policy simulations.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_candidates(
    model_path: str,
    prompts: list[dict[str, Any]],
    n_candidates: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 512,
    max_seq_len: int = 2048,
    seed: int = 42,
    batch_size: int | None = None,
) -> list[dict[str, Any]]:
    """Generate n_candidates SQL outputs per prompt using vLLM.

    Must run on GPU. Returns list of records with candidates nested.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="float16",
        max_model_len=max_seq_len,
        gpu_memory_utilization=0.85,
        seed=seed,
    )
    tokenizer = llm.get_tokenizer()

    from training.prompt_template import SYSTEM_MESSAGE
    from core.prompt_builder import build_prompt

    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": build_prompt(
                p["question"], p["schema"], p.get("evidence", ""),
            )},
        ]
        formatted_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n=n_candidates,
        seed=seed,
        logprobs=1,
    )

    start = time.perf_counter()
    if batch_size is None or batch_size <= 0:
        batch_size = len(formatted_prompts) or 1
    outputs = []
    for start_idx in range(0, len(formatted_prompts), batch_size):
        outputs.extend(llm.generate(formatted_prompts[start_idx:start_idx + batch_size], params))
    elapsed = time.perf_counter() - start

    from core.sql_extractor import extract_sql

    results = []
    for prompt_data, output in zip(prompts, outputs):
        candidates = []
        for i, completion in enumerate(output.outputs):
            raw_text = completion.text
            try:
                sql = extract_sql(raw_text)
            except ValueError:
                sql = ""

            avg_logprob = 0.0
            if completion.logprobs:
                token_logprobs = [
                    lp.logprob for token_lps in completion.logprobs
                    for lp in token_lps.values()
                    if hasattr(lp, "logprob")
                ]
                if token_logprobs:
                    avg_logprob = sum(token_logprobs) / len(token_logprobs)

            candidates.append({
                "index": i,
                "raw_text": raw_text,
                "sql": sql,
                "avg_logprob": round(avg_logprob, 4),
                "n_tokens": len(completion.token_ids),
            })

        results.append({
            "question": prompt_data["question"],
            "db_id": prompt_data["db_id"],
            "schema": prompt_data.get("schema", {}),
            "gold_sql": prompt_data.get("gold_sql", ""),
            "difficulty": prompt_data.get("difficulty", "unknown"),
            "candidates": candidates,
        })

    logger.info(
        "Generated %d questions x %d candidates in %.1fs",
        len(prompts), n_candidates, elapsed,
    )
    return results


def save_candidates(records: list[dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Saved %d records to %s", len(records), output_path)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-candidates", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from benchmarks.bird_loader import load_bird_dev, get_schema

    bird_dev = load_bird_dev(args.data_dir)
    prompts = []
    for r in bird_dev:
        schema = get_schema(args.data_dir, r["db_id"])
        prompts.append({**r, "schema": schema})

    results = generate_candidates(
        args.model, prompts, args.n_candidates, seed=args.seed, batch_size=args.batch_size,
    )
    save_candidates(results, args.output)


if __name__ == "__main__":
    main()
