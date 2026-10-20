"""Paired, CPU-only scoring of original CSC and adapted CSC on frozen candidates."""

from __future__ import annotations

from pathlib import Path

import sqlglot
from sqlglot import exp

from evaluation.final_eval import bootstrap_mean_ci, mcnemar_test
from research.slm_sql import PROTOCOL_VERSION
from research.slm_sql.artifacts import atomic_json, completed_artifact, file_hash
from research.slm_sql.config import artifact_path
from research.slm_sql.execution import EXECUTION_VERSION
from research.slm_sql.pipeline import load_execution
from research.slm_sql.prompts import PROMPT_VERSION, top_groups


def _load_merge(cfg, split, seed, k, variant, execution_manifest, ids):
    path = artifact_path(cfg, split, f"{variant}.csc-k{k}", seed)
    rows, manifest = completed_artifact(path)
    spec = manifest["spec"]
    expected = {
        "protocol": PROTOCOL_VERSION,
        "prompt": PROMPT_VERSION,
        "seed": seed,
        "k": k,
        "split": split,
        "variant": variant,
        "merger": cfg["merger"],
        "execution": execution_manifest["sha256"],
    }
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{variant} CSC cache does not match this experiment.")
    if spec["implementation"] != {
        name: file_hash(Path(__file__).with_name(name)) for name in ("backend.py", "prompts.py")
    }:
        raise ValueError("CSC prompt/backend implementation changed.")
    if [r["id"] for r in rows] != ids:
        raise ValueError("CSC evaluation requires exactly paired question IDs in order.")
    return rows, manifest


def _features(sql):
    tree = sqlglot.parse_one(sql, read="sqlite")
    return {"join": any(tree.find_all(exp.Join)), "aggregate": any(tree.find_all(exp.AggFunc))}


def score(cfg: dict, split="evaluation", seed=42, k=None, *, baseline_only=False) -> dict:
    records, execution_manifest = load_execution(cfg, split, seed)
    n = cfg["generation"][f"{split}_samples"]
    k = n if k is None else k
    if not records or not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
        raise ValueError("Scoring requires nonempty paired records and k within the cache.")
    protocol_signature = None
    if split == "evaluation" and not baseline_only:
        from research.slm_sql.protocol import verify_freeze
        protocol_signature = verify_freeze(cfg, seed, k)
    ids = [r["id"] for r in records]
    author_rows, author_manifest = _load_merge(
        cfg, split, seed, k, "author", execution_manifest, ids
    )
    variants = {"author_csc": author_rows}
    manifests = {"author_csc": author_manifest}
    if not baseline_only:
        from research.slm_sql.adaptation import validate_adapter

        _, adapter_identity, _ = validate_adapter(cfg, split, seed, k)
        lora_rows, lora_manifest = _load_merge(cfg, split, seed, k, "lora", execution_manifest, ids)
        if lora_manifest["spec"].get("adapter") != adapter_identity:
            raise ValueError(
                "LoRA predictions were produced with different adapter weights. Run merge for the current adapter."
            )
        left = {
            key: value
            for key, value in author_manifest["spec"].items()
            if key not in ("variant", "adapter")
        }
        right = {
            key: value
            for key, value in lora_manifest["spec"].items()
            if key not in ("variant", "adapter")
        }
        if author_manifest.get("packages") != lora_manifest.get("packages"):
            raise ValueError("Paired CSC comparison requires identical inference package versions.")
        if left != right or author_manifest.get("model_identity") != lora_manifest.get(
            "model_identity"
        ):
            raise ValueError(
                "Paired CSC comparison requires identical base weights, prompts, seeds and budgets."
            )
        variants["lora_csc"] = lora_rows
        manifests["lora_csc"] = lora_manifest
    _, generation_manifest = completed_artifact(artifact_path(cfg, split, "generation", seed))
    per_question = []
    for index, record in enumerate(records):
        gold_fp = record["gold_execution"]["fingerprint"]
        if gold_fp is None:
            raise ValueError(
                "Gold SQL cannot be scored; fix the benchmark rather than counting it wrong."
            )
        groups = top_groups(record["candidates"], k)
        sc_fp = groups[0]["fingerprint"] if groups else None
        flags = {"sc": sc_fp == gold_fp}
        predictions = {"sc": groups[0]["sql"] if groups else ""}
        costs = {}
        generation_tokens = sum(c["n_tokens"] for c in record["candidates"][:k])
        for name, variant_records in variants.items():
            output = variant_records[index]
            flags[name] = output["selected_fp"] == gold_fp
            predictions[name] = output["selected_sql"]
            costs[name] = {
                "output_tokens": generation_tokens + output["merge_tokens"],
                "merge_tokens": output["merge_tokens"],
                "csc_called": output["csc_called"],
                "amortized_merge_s": output["amortized_merge_s"],
                "amortized_generation_s": record["amortized_generation_s"] if k == n else None,
            }
        per_question.append(
            {
                "id": record["id"],
                "db_id": record["db_id"],
                "difficulty": record["difficulty"],
                "question": record["question"],
                "gold_sql": record["gold_sql"],
                "features": _features(record["gold_sql"]),
                "correct": flags,
                "predictions": predictions,
                "top2_contains_gold": gold_fp in [g["fingerprint"] for g in groups[:2]],
                "pass_at_k": gold_fp in [g["fingerprint"] for g in groups],
                "generation_tokens": generation_tokens,
                "generation_truncated": sum(
                    c["finish_reason"] == "length" for c in record["candidates"][:k]
                ),
                "generation_format_errors": sum(
                    bool(c.get("format_error", c["parse_error"])) for c in record["candidates"][:k]
                ),
                "costs": costs,
            }
        )
    bootstrap = cfg["evaluation"]["bootstrap_samples"]
    summary = {}
    for name in ("sc", *variants):
        flags = [r["correct"][name] for r in per_question]
        metrics = {
            "ex": sum(flags) / len(flags),
            "questions": len(flags),
            "accuracy_ci95": bootstrap_mean_ci(flags, bootstrap, seed),
            "by_difficulty": {},
            "by_sql_feature": {},
            "by_database": {},
        }
        for difficulty in sorted({r["difficulty"] for r in per_question}):
            subset = [r for r in per_question if r["difficulty"] == difficulty]
            metrics["by_difficulty"][difficulty] = {
                "questions": len(subset),
                "ex": sum(r["correct"][name] for r in subset) / len(subset),
            }
        for db_id in sorted({r["db_id"] for r in per_question}):
            subset = [r for r in per_question if r["db_id"] == db_id]
            metrics["by_database"][db_id] = {
                "questions": len(subset), "ex": sum(r["correct"][name] for r in subset) / len(subset),
            }
        for feature in ("join", "aggregate"):
            subset = [r for r in per_question if r["features"][feature]]
            metrics["by_sql_feature"][feature] = {
                "questions": len(subset),
                "ex": sum(r["correct"][name] for r in subset) / len(subset) if subset else None,
            }
        if name != "sc":
            called = [r for r in per_question if r["costs"][name]["csc_called"]]
            metrics.update(
                mean_generation_samples=k,
                mean_csc_samples=sum(len(r["merge_candidates"]) for r in variants[name]) / len(flags),
                mean_amortized_total_s=(sum(
                    r["costs"][name]["amortized_generation_s"] + r["costs"][name]["amortized_merge_s"] for r in per_question
                ) / len(flags)) if k == n else None,
                csc_calls=len(called),
                csc_subset_ex=sum(r["correct"][name] for r in called) / len(called)
                if called
                else None,
                wrong_to_correct=sum(
                    r["correct"][name] and not r["correct"]["sc"] for r in per_question
                ),
                correct_to_wrong=sum(
                    r["correct"]["sc"] and not r["correct"][name] for r in per_question
                ),
                mean_output_tokens=sum(r["costs"][name]["output_tokens"] for r in per_question)
                / len(flags),
                mean_amortized_merge_s=sum(
                    r["costs"][name]["amortized_merge_s"] for r in per_question
                )
                / len(flags),
                merge_truncated=sum(
                    c["finish_reason"] == "length"
                    for row in variants[name]
                    for c in row["merge_candidates"]
                ),
            )
        summary[name] = metrics
    comparison = None
    if "lora_csc" in variants:
        author_flags = [r["correct"]["author_csc"] for r in per_question]
        lora_flags = [r["correct"]["lora_csc"] for r in per_question]
        test = mcnemar_test(lora_flags, author_flags)
        comparison = {
            "delta_ex": summary["lora_csc"]["ex"] - summary["author_csc"]["ex"],
            "paired_delta_ci95": bootstrap_mean_ci(
                [int(a) - int(b) for a, b in zip(lora_flags, author_flags)], bootstrap, seed
            ),
            "mcnemar": {
                "lora_only_correct": test["adaptive_only"],
                "author_only_correct": test["fixed_only"],
                "exact_p_value": test["exact_p_value"],
            },
        }
    report = {
        "protocol": PROTOCOL_VERSION,
        "evaluator": EXECUTION_VERSION,
        "split": split,
        "seed": seed,
        "k": k,
        "summary": summary,
        "comparison": comparison,
        "final_protocol_signature": protocol_signature,
        "oracle": {
            "pass_at_k": sum(r["pass_at_k"] for r in per_question) / len(per_question),
            "top2_coverage_at_k": sum(r["top2_contains_gold"] for r in per_question) / len(per_question),
        },
        "generation_diagnostics": {
            "truncated": sum(r["generation_truncated"] for r in per_question),
            "format_errors": sum(r["generation_format_errors"] for r in per_question),
        },
        "provenance": {
            "generation": generation_manifest,
            "execution": execution_manifest,
            "variants": manifests,
        },
        "notes": [
            "Author checkpoints plus a semantic reconstruction of Appendix E; no claim of exact paper-score reproduction.",
            "Full DDL and bounded result previews; first-generated tie breaking; no fallback when every merge output is invalid.",
            "Timing is amortized cached-batch timing, not interactive latency. Prefix-k generation time is unavailable when k < cached n.",
            "Bootstrap resamples paired questions; SQL-feature slices are post-hoc and may overlap.",
        ],
        "records": per_question,
    }
    suffix = "baseline" if baseline_only else "comparison"
    path = Path(cfg["experiment_dir"]) / "reports" / f"{split}-seed{seed}-k{k}.{suffix}.json"
    atomic_json(path, report)
    return {"path": str(path), "summary": summary, "comparison": comparison}
