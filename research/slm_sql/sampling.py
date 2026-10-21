"""Paired greedy, fixed-k, CSC and real online-adaptive SLM-SQL evaluation.

pass@k is an oracle diagnostic, self_consistency@k votes on execution groups,
and the paper's SLM-SQL@k includes the author's CSC merger. A first stochastic
sample is never labeled greedy, and prefix cost is never inferred from full-n
batch latency. All scoring is CPU-only and rejects incomplete/stale artifacts.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path

from evaluation.final_eval import bootstrap_mean_ci, mcnemar_test
from research.slm_sql import PROTOCOL_VERSION
from research.slm_sql.artifacts import atomic_json, digest, file_hash, read_json
from research.slm_sql.evaluation import _features, load_csc_variants
from research.slm_sql.execution import EXECUTION_VERSION
from research.slm_sql.pipeline import load_adaptive, load_execution, load_generation
from research.slm_sql.prompts import PROMPT_VERSION, top_groups


def sampling_ks(cfg, k=None, split="evaluation"):
    values = [k] if k is not None else cfg["evaluation"]["sampling_k_values"]
    n = cfg["generation"][f"{split}_samples"]
    if (not values or any(not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= n for v in values)
            or len(values) != len(set(values))):
        raise ValueError("Sampling k values must be unique positive integers within the cached budget.")
    return sorted(values)


def _mean(values):
    return sum(values) / len(values) if values and all(v is not None for v in values) else None


def _paired(left, right, cfg, seed):
    differences = [int(a) - int(b) for a, b in zip(left, right)]
    test = mcnemar_test(left, right)
    return {
        "delta_ex": _mean(differences),
        "paired_delta_ci95": bootstrap_mean_ci(differences, cfg["evaluation"]["bootstrap_samples"], seed),
        "improved": test["adaptive_only"], "regressed": test["fixed_only"],
        "mcnemar_exact_p_value": test["exact_p_value"], "questions": len(left),
    }


def _check_pairing(expected, actual, label):
    if [r["id"] for r in expected] != [r["id"] for r in actual]:
        raise ValueError(f"{label} must contain exactly the same paired question IDs in order.")
    if any(a["gold_execution"]["fingerprint"] != b["gold_execution"]["fingerprint"]
           for a, b in zip(expected, actual)):
        raise ValueError(f"{label} gold execution differs from the fixed cache.")


def _generator_identity(expected, actual, label):
    if any(expected.get(key) != actual.get(key) for key in ("model_identity", "packages")):
        raise ValueError(f"{label} must use identical generator weights and inference package versions.")


def _cost(record, k=None, merge=None, *, full=True, adaptive=False):
    candidates = record["candidates"][:k]
    merge = merge or {}
    gen_s = record["amortized_generation_s"] if full else None
    merge_s = merge.get("amortized_merge_s", 0.0)
    gen_tokens = sum(c["n_tokens"] for c in candidates)
    merge_tokens = merge.get("merge_tokens", 0)
    return {
        "generation_samples": len(candidates),
        "merge_samples": len(merge.get("merge_candidates", [])),
        "generation_tokens": gen_tokens, "merge_tokens": merge_tokens,
        "output_tokens": gen_tokens + merge_tokens,
        "csc_called": bool(merge.get("csc_called", False)),
        "generation_s": gen_s, "amortized_merge_s": merge_s,
        "generation_merge_s": gen_s + merge_s if gen_s is not None else None,
        "online_generator_inference_s": record.get("online_inference_s") if adaptive else None,
    }


def _put(row, method, fp, sql, gold, cost):
    row["correct"][method] = fp is not None and fp == gold
    row["predictions"][method] = sql
    row["costs"][method] = cost


def _summarize(rows, method, cfg, seed):
    flags = [r["correct"][method] for r in rows]
    costs = [r["costs"][method] for r in rows]
    result = {
        "ex": _mean(flags), "questions": len(rows),
        "accuracy_ci95": bootstrap_mean_ci(flags, cfg["evaluation"]["bootstrap_samples"], seed),
        "mean_samples": _mean([c["generation_samples"] for c in costs]),
        "mean_merge_samples": _mean([c["merge_samples"] for c in costs]),
        "mean_tokens": _mean([c["output_tokens"] for c in costs]),
        "mean_generation_tokens": _mean([c["generation_tokens"] for c in costs]),
        "mean_merge_tokens": _mean([c["merge_tokens"] for c in costs]),
        "csc_calls": sum(c["csc_called"] for c in costs),
        "mean_generation_s": _mean([c["generation_s"] for c in costs]),
        "mean_amortized_merge_s": _mean([c["amortized_merge_s"] for c in costs]),
        "mean_generation_merge_s": _mean([c["generation_merge_s"] for c in costs]),
        "mean_online_generator_inference_s": _mean([c["online_generator_inference_s"] for c in costs]),
    }
    for field in ("difficulty", "db_id"):
        result[f"by_{field}"] = {
            value: {"questions": len(subset), "ex": _mean([r["correct"][method] for r in subset])}
            for value in sorted({r[field] for r in rows})
            for subset in [[r for r in rows if r[field] == value]]
        }
    result["by_sql_feature"] = {
        feature: {"questions": len(subset), "ex": _mean([r["correct"][method] for r in subset])}
        for feature in ("join", "aggregate")
        for subset in [[r for r in rows if r["features"][feature]]]
    }
    return result


def _write_table(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_suite(
    cfg, split="evaluation", seed=42, *, k=None, baseline_only=False, generator_only=False,
):
    """Require every declared output; missing/stale predictions never become zero EX."""
    if baseline_only and generator_only:
        raise ValueError("Choose either baseline-only or generator-only scoring, not both.")
    ks = sampling_ks(cfg, k, split)
    fixed, fixed_manifest = load_execution(cfg, split, seed)
    greedy = greedy_manifest = greedy_generator = None
    greedy_reason = None
    try:
        greedy, greedy_manifest = load_execution(cfg, split, seed, mode="greedy")
        _, greedy_generator = load_generation(cfg, split, seed, mode="greedy")
    except FileNotFoundError:
        greedy_reason = "greedy artifact is missing; run greedy or sampling-baseline first"
    _, generator = load_generation(cfg, split, seed)
    if greedy is not None:
        _check_pairing(fixed, greedy, "Greedy cache")
        _generator_identity(generator, greedy_generator, "Greedy cache")
    adaptive, adaptive_manifest = None, None
    if cfg["adaptive"]["enabled"]:
        adaptive, adaptive_manifest = load_adaptive(cfg, split, seed)
        _check_pairing(fixed, adaptive, "Adaptive cache")
        _generator_identity(generator, adaptive_manifest, "Adaptive cache")
    n = cfg["generation"][f"{split}_samples"]
    rows = []
    for index, record in enumerate(fixed):
        gold = record["gold_execution"]["fingerprint"]
        if gold is None:
            raise ValueError("Gold execution failed; fix the benchmark before scoring.")
        if len(record["candidates"]) != n:
            raise ValueError("Fixed cache contains an incorrect candidate count.")
        row = {key: record[key] for key in ("id", "db_id", "question", "gold_sql", "difficulty")}
        row.update(features=_features(record["gold_sql"]), correct={}, predictions={}, costs={}, oracle={})
        if greedy is not None:
            if len(greedy[index]["candidates"]) != 1:
                raise ValueError("Greedy cache contains an incorrect candidate count.")
            candidate = greedy[index]["candidates"][0]
            _put(row, "greedy", candidate["execution"]["fingerprint"], candidate["sql"], gold, _cost(greedy[index]))
        if adaptive is not None:
            current = adaptive[index]
            groups = top_groups(current["candidates"])
            leader = groups[0] if groups else {}
            # Recompute from predictions instead of trusting cached correctness labels.
            _put(row, "adaptive_sc", leader.get("fingerprint"), leader.get("sql", ""), gold,
                 _cost(current, adaptive=True))
            row["adaptive"] = {key: current[key] for key in ("stop_reason", "n_samples_used", "rounds")}
        rows.append(row)

    provenance = {"fixed": fixed_manifest, "generator": generator, "greedy": greedy_manifest,
                  "greedy_generator": greedy_generator, "adaptive": adaptive_manifest, "csc": {}}
    final_signature = None
    for k_value in ks:
        variants, manifests = {}, {}
        if not generator_only:
            variants, manifests, signature = load_csc_variants(
                cfg, split, seed, k_value, fixed, fixed_manifest, baseline_only=baseline_only,
            )
            final_signature = signature or final_signature
        provenance["csc"][str(k_value)] = manifests
        for index, record in enumerate(fixed):
            row = rows[index]
            gold = record["gold_execution"]["fingerprint"]
            groups = top_groups(record["candidates"], k_value)
            leader = groups[0] if groups else {}
            _put(row, f"sc@{k_value}", leader.get("fingerprint"), leader.get("sql", ""), gold,
                 _cost(record, k_value, full=k_value == n))
            # Primary pass@k is prefix membership, paired with this same SC/CSC
            # candidate set. Also expose the unbiased all-n estimator separately.
            correct_count = sum(c["execution"]["fingerprint"] == gold for c in record["candidates"])
            row["oracle"][str(k_value)] = {
                "pass@k": gold in [g["fingerprint"] for g in groups],
                "top2_coverage@k": gold in [g["fingerprint"] for g in groups[:2]],
                "estimated_pass@k": 1 - math.comb(n - correct_count, k_value) / math.comb(n, k_value),
            }
            for variant, predictions in variants.items():
                output = predictions[index]
                _put(row, f"{variant}@{k_value}", output["selected_fp"], output["selected_sql"], gold,
                     _cost(record, k_value, output, full=k_value == n))
    if adaptive is not None and not generator_only:
        variants, manifests, signature = load_csc_variants(
            cfg, split, seed, None, adaptive, adaptive_manifest,
            baseline_only=baseline_only, source="adaptive",
        )
        final_signature = signature or final_signature
        provenance["csc"]["adaptive"] = manifests
        for index, record in enumerate(adaptive):
            for variant, predictions in variants.items():
                output = predictions[index]
                _put(rows[index], f"adaptive_{variant}", output["selected_fp"], output["selected_sql"],
                     record["gold_execution"]["fingerprint"], _cost(record, merge=output, adaptive=True))
    methods = {name: _summarize(rows, name, cfg, seed) for name in rows[0]["correct"]}
    fixed_summary = {}
    comparisons = {}
    for k_value in ks:
        label = str(k_value)
        oracle = {key: _mean([r["oracle"][label][key] for r in rows])
                  for key in ("pass@k", "top2_coverage@k", "estimated_pass@k")}
        fixed_summary[label] = {
            **oracle, "self_consistency@k": methods[f"sc@{k_value}"],
            "author_csc@k": methods.get(f"author_csc@{k_value}"),
            "lora_csc@k": methods.get(f"lora_csc@{k_value}"),
            "slm_sql@k": methods.get(f"author_csc@{k_value}"),
        }
        pairs = [(f"sc@{k_value}", "greedy"), (f"author_csc@{k_value}", f"sc@{k_value}"),
                 (f"lora_csc@{k_value}", f"author_csc@{k_value}"),
                 ("adaptive_sc", f"sc@{k_value}"),
                 ("adaptive_author_csc", f"author_csc@{k_value}"),
                 ("adaptive_lora_csc", f"lora_csc@{k_value}")]
        for left, right in pairs:
            if left in methods and right in methods:
                comparisons[f"{left}_vs_{right}"] = _paired(
                    [r["correct"][left] for r in rows], [r["correct"][right] for r in rows], cfg, seed,
                )
    if "adaptive_lora_csc" in methods:
        comparisons["adaptive_lora_csc_vs_adaptive_author_csc"] = _paired(
            [r["correct"]["adaptive_lora_csc"] for r in rows],
            [r["correct"]["adaptive_author_csc"] for r in rows], cfg, seed,
        )
    table = [{"method": name, **{key: value[key] for key in (
        "ex", "questions", "mean_samples", "mean_merge_samples", "mean_tokens", "csc_calls",
        "mean_generation_merge_s",
    )}} for name, value in methods.items()]
    mode = "generator" if generator_only else "baseline" if baseline_only else "comparison"
    report = {
        "protocol": PROTOCOL_VERSION, "evaluator": EXECUTION_VERSION, "prompt": PROMPT_VERSION,
        "split": split, "seed": seed, "k_values": ks, "mode": mode,
        "config_signature": digest(cfg), "final_protocol_signature": final_signature,
        "summary": {"greedy": methods.get("greedy"), "greedy_reason": greedy_reason,
                    "fixed": fixed_summary, "adaptive": methods.get("adaptive_sc"), "methods": methods},
        "paired_comparisons": comparisons, "table": table, "provenance": provenance, "records": rows,
        "notes": [
            "pass@k is oracle prefix membership; estimated_pass@k is the all-n unbiased estimator, reported separately.",
            "self_consistency@k (SC@k) is execution-group voting; SLM-SQL@k includes author CSC, as in paper section 4.3.2.",
            "Greedy uses a separate temperature-zero request; stochastic k=1 is not greedy.",
            "Adaptive generates independent real model draws in rounds; it never consumes dev gold in its stopping policy.",
            "Fixed-prefix time is unavailable for k < n. Cached batch times and online sequential times are different workloads.",
            "CSC token/sample costs include merger outputs. Invalid merge outputs are failures, with no hidden SC fallback.",
            "Policy parameters are fixed in config, not selected from final dev EX. These are not new SFT training runs.",
            "Greedy is optional; a missing greedy cache is reported explicitly and never scored as zero.",
            "Author checkpoints and semantic Appendix E prompts do not establish exact paper-score reproduction.",
        ],
    }
    subset = f"-k{k}" if k is not None else ""
    path = Path(cfg["experiment_dir"]) / "reports" / f"{split}-seed{seed}{subset}.sampling.{mode}.json"
    atomic_json(path, report)
    _write_table(path.with_suffix(".csv"), table)
    return {"path": str(path), "summary": report["summary"]}


def aggregate_suites(cfg, seeds=(42, 43, 44), *, k=None, baseline_only=False, generator_only=False):
    """Bootstrap questions averaged over seeds, retaining each seed's paired test."""
    seeds = list(seeds)
    if (not seeds or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Provide unique nonnegative inference seeds.")
    if not baseline_only and not generator_only:
        from research.slm_sql.protocol import verify_freeze
        if not (Path(cfg["experiment_dir"]) / "final_protocol.json").is_file():
            raise ValueError("Freeze the final sampling protocol before producing a final multi-seed report.")
        for seed in seeds:
            for value in sampling_ks(cfg, k):
                verify_freeze(cfg, seed, value)
    reports, sources = [], []
    for seed in seeds:
        result = evaluate_suite(cfg, seed=seed, k=k, baseline_only=baseline_only, generator_only=generator_only)
        path = Path(result["path"])
        reports.append(read_json(path))
        sources.append({"path": str(path), "sha256": file_hash(path)})
    ids = [r["id"] for r in reports[0]["records"]]
    names = list(reports[0]["summary"]["methods"])
    for report in reports[1:]:
        if [r["id"] for r in report["records"]] != ids or list(report["summary"]["methods"]) != names:
            raise ValueError("Every inference seed must contain identical paired questions and methods.")
    summary, table = {}, []
    for name in names:
        exs = [r["summary"]["methods"][name]["ex"] for r in reports]
        by_question = [_mean([r["records"][i]["correct"][name] for r in reports]) for i in range(len(ids))]
        metrics = {
            "ex_mean": statistics.mean(exs),
            "ex_std": statistics.stdev(exs) if len(exs) > 1 else 0.0,
            "ex_by_seed": dict(zip(map(str, seeds), exs)),
            "seed_averaged_question_ci95": bootstrap_mean_ci(by_question, cfg["evaluation"]["bootstrap_samples"]),
            **{key: _mean([r["summary"]["methods"][name][key] for r in reports])
               for key in ("mean_samples", "mean_merge_samples", "mean_tokens", "mean_generation_merge_s")},
        }
        summary[name] = metrics
        table.append({"method": name, "seeds": len(seeds), "questions": len(ids),
                      **{key: metrics[key] for key in ("ex_mean", "ex_std", "mean_samples", "mean_merge_samples", "mean_tokens")}})
    paired = {}
    for name in reports[0]["paired_comparisons"]:
        left, right = name.split("_vs_", 1)
        delta = [_mean([int(r["records"][i]["correct"][left]) - int(r["records"][i]["correct"][right])
                        for r in reports]) for i in range(len(ids))]
        paired[name] = {
            "delta_ex": _mean(delta),
            "paired_delta_ci95": bootstrap_mean_ci(delta, cfg["evaluation"]["bootstrap_samples"]),
            "tests_by_seed": {str(seed): r["paired_comparisons"][name] for seed, r in zip(seeds, reports)},
        }
    mode = reports[0]["mode"]
    report = {
        "seeds": seeds, "k_values": reports[0]["k_values"], "questions": len(ids), "summary": summary,
        "oracle": {str(value): {key: _mean([r["summary"]["fixed"][str(value)][key] for r in reports])
                                for key in ("pass@k", "estimated_pass@k", "top2_coverage@k")}
                   for value in reports[0]["k_values"]},
        "paired_comparisons": paired, "sources": sources,
        "final_protocol_signature": reports[0]["final_protocol_signature"],
        "notes": ["Seeds share one adapter, not independent training runs.",
                  "Question-level bootstrap averages over seeds; McNemar remains per seed.",
                  "pass@k is oracle coverage and is excluded from deployable method rankings."],
    }
    subset = f"-k{k}" if k is not None else ""
    path = Path(cfg["experiment_dir"]) / "reports" / f"evaluation{subset}.sampling.{mode}.multi-seed.json"
    atomic_json(path, report)
    _write_table(path.with_suffix(".csv"), table)
    return {"path": str(path), "summary": summary}
