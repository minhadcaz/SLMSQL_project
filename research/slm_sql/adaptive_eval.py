"""Exact offline adaptive prefix evaluation and validation policy selection."""

from __future__ import annotations

import statistics
from pathlib import Path

from evaluation.final_eval import bootstrap_mean_ci, mcnemar_test
from research.slm_sql import PROTOCOL_VERSION
from research.slm_sql.artifacts import atomic_json, digest, file_hash, read_json
from research.slm_sql.evaluation import _load_merge
from research.slm_sql.pipeline import load_execution
from research.slm_sql.policy_view import policy_view
from research.slm_sql.prompts import top_groups


def adaptive_checkpoints(cfg: dict, split: str, checkpoints=None) -> list[int]:
    """Return valid ordered checkpoints for a prepared split."""
    n = cfg["generation"][f"{split}_samples"]
    values = list(cfg["adaptive"].get("checkpoints", [])) if checkpoints is None else list(checkpoints)
    if not values or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in values):
        raise ValueError("adaptive checkpoints must be unique positive integers.")
    if len(values) != len(set(values)):
        raise ValueError("adaptive checkpoints must be unique positive integers.")
    values = sorted(k for k in values if k <= n)
    if not values:
        raise ValueError(f"No adaptive checkpoint is <= {split}_samples ({n}).")
    if split == "evaluation":
        declared = set(cfg["evaluation"]["sampling_k_values"])
        missing = [k for k in values if k not in declared]
        if missing:
            raise ValueError(
                "Evaluation adaptive checkpoints must be sampling_k_values; missing "
                + ", ".join(map(str, missing))
            )
    return values


def _selected_policy_path(cfg: dict) -> Path:
    return Path(cfg["experiment_dir"]) / "adaptive" / "selected_policy.json"


def _load_selected_policy(cfg: dict, split: str) -> tuple[dict, str]:
    from research.slm_sql.protocol import load_selected_policy

    value, selected_hash = load_selected_policy(cfg, required=True)
    if split == "evaluation" and value.get("validation_artifacts") is None:
        raise ValueError("Selected adaptive policy has no validation artifact provenance.")
    return value, selected_hash


def _policy(cfg: dict, split: str, parameters: dict):
    from research.slm_sql.pipeline import _adaptive_policy

    if not parameters:
        return _adaptive_policy(cfg, split)
    policy_name = parameters.get("policy", cfg["adaptive"]["policy"])
    if policy_name == "beta":
        from research.policies.beta_stopping import BetaStoppingPolicy

        return BetaStoppingPolicy(
            threshold=parameters.get("threshold", cfg["adaptive"]["threshold"]),
            min_k=parameters.get("min_k", cfg["adaptive"]["min_k"]),
            max_k=min(parameters.get("max_k", cfg["adaptive"]["max_k"]), cfg["generation"][f"{split}_samples"]),
        )
    if policy_name == "margin":
        from research.policies.margin_entropy import MarginEntropyPolicy

        return MarginEntropyPolicy(
            margin=parameters.get("margin", cfg["adaptive"]["margin"]),
            entropy_threshold=parameters.get(
                "entropy_threshold", cfg["adaptive"]["entropy_threshold"]
            ),
            min_k=parameters.get("min_k", cfg["adaptive"]["min_k"]),
            max_k=min(parameters.get("max_k", cfg["adaptive"]["max_k"]), cfg["generation"][f"{split}_samples"]),
        )
    if policy_name == "router":
        from research.policies.three_action_router import ThreeActionRouterPolicy

        return ThreeActionRouterPolicy(
            stop_threshold=parameters.get("stop_threshold", cfg["adaptive"].get("stop_threshold", 0.9)),
            csc_threshold=parameters.get("csc_threshold", cfg["adaptive"].get("csc_threshold", 0.7)),
            min_k=parameters.get("min_k", cfg["adaptive"]["min_k"]),
            max_k=min(parameters.get("max_k", cfg["adaptive"]["max_k"]), cfg["generation"][f"{split}_samples"]),
        )
    raise ValueError("Adaptive policy must be beta, margin or router.")


def _load_merge_checkpoints(cfg, split, seed, records, execution_manifest, checkpoints):
    """Load both merger variants and give exact commands for missing checkpoints."""
    loaded = {"author": {}, "lora": {}}
    missing = []
    for k in checkpoints:
        for variant in loaded:
            try:
                rows, manifest = _load_merge(
                    cfg, split, seed, k, variant, execution_manifest, [r["id"] for r in records]
                )
            except FileNotFoundError:
                missing.append(
                    f"python -m research.slm_sql --config <CONFIG> merge --split {split} "
                    f"--variant {variant} --k {k} --seed {seed}"
                )
                continue
            loaded[variant][k] = (rows, manifest)
    if missing:
        raise ValueError("Missing required adaptive merge artifacts; run:\n" + "\n".join(missing))
    return loaded


def _method_summary(rows: list[dict], method: str, bootstrap_samples: int, seed: int) -> dict:
    flags = [bool(row["correct"][method]) for row in rows]
    costs = [row["costs"][method] for row in rows]
    return {
        "ex": sum(flags) / len(flags) if flags else 0.0,
        "questions": len(flags),
        "accuracy_ci95": bootstrap_mean_ci(flags, bootstrap_samples, seed),
        "mean_samples": statistics.mean([c["generation_samples"] for c in costs]) if costs else 0.0,
        "mean_tokens": statistics.mean([c["output_tokens"] for c in costs]) if costs else 0.0,
        "csc_calls": sum(bool(c["csc_called"]) for c in costs),
        "accuracy_basis": "offline_prefix_exact",
    }


def _paired(rows: list[dict], left: str, right: str, bootstrap_samples: int, seed: int) -> dict:
    left_flags = [bool(row["correct"][left]) for row in rows]
    right_flags = [bool(row["correct"][right]) for row in rows]
    test = mcnemar_test(left_flags, right_flags)
    return {
        "delta_ex": sum(int(a) - int(b) for a, b in zip(left_flags, right_flags)) / len(rows),
        "paired_delta_ci95": bootstrap_mean_ci(
            [int(a) - int(b) for a, b in zip(left_flags, right_flags)],
            bootstrap_samples,
            seed,
        ),
        "wrong_to_correct": test["adaptive_only"],
        "correct_to_wrong": test["fixed_only"],
        "mcnemar_exact_p_value": test["exact_p_value"],
        "questions": len(rows),
    }


def _score_policy(cfg, split, seed, policy_parameters, checkpoints, merges):
    records, execution_manifest = load_execution(cfg, split, seed)
    policy = _policy(cfg, split, policy_parameters)
    max_k = checkpoints[-1]
    rows = []
    for index, record in enumerate(records):
        candidates = record["candidates"]
        selected = None
        trace = []
        for k in checkpoints:
            prefix = candidates[:k]
            observed = policy_view(prefix)
            stop, action = policy.should_stop(observed, k)
            groups = top_groups(prefix)
            reached_budget = k == max_k
            budget_unresolved = reached_budget and not stop and action == "SAMPLE_MORE"
            call_csc_at_budget = (
                cfg["adaptive"].get("call_csc_at_budget", False)
                and budget_unresolved
                and len(groups) >= 2
            )
            if call_csc_at_budget:
                action = "CALL_CSC"
            use_merge = action == "CALL_CSC"
            trace.append({"k": k, "action": action, "stop": stop, "call_csc_at_budget": call_csc_at_budget})
            if not stop and not use_merge and not reached_budget:
                continue
            if use_merge:
                output = merges["author"][k][0][index]
                if action == "CALL_CSC" and not output["csc_called"]:
                    raise ValueError(
                        f"Adaptive policy requested CSC for {record['id']} but the merge artifact did not call the merger."
                    )
                selected = {
                    "fp": output["selected_fp"],
                    "sql": output["selected_sql"],
                    "cost": {
                        "generation_samples": k,
                        "generation_tokens": sum(c.get("n_tokens", 0) for c in prefix),
                        "merge_samples": len(output["merge_candidates"]),
                        "merge_tokens": output["merge_tokens"],
                        "output_tokens": sum(c.get("n_tokens", 0) for c in prefix) + output["merge_tokens"],
                        "csc_called": output["csc_called"],
                    },
                    "action": action,
                    "checkpoint": k,
                }
            else:
                groups = top_groups(prefix)
                leader = groups[0] if groups else None
                selected = {
                    "fp": leader["fingerprint"] if leader else None,
                    "sql": leader["sql"] if leader else "",
                    "cost": {
                        "generation_samples": k,
                        "generation_tokens": sum(c.get("n_tokens", 0) for c in prefix),
                        "merge_samples": 0,
                        "merge_tokens": 0,
                        "output_tokens": sum(c.get("n_tokens", 0) for c in prefix),
                        "csc_called": False,
                    },
                    "action": "STOP" if stop else "STOP_AT_BUDGET",
                    "checkpoint": k,
                }
            break
        if selected is None:
            raise ValueError(f"Policy did not resolve question {record['id']} at a checkpoint.")
        gold = record["gold_execution"]["fingerprint"]
        max_groups = top_groups(candidates, max_k)
        row = {
            "id": record["id"],
            "db_id": record["db_id"],
            "question": record["question"],
            "gold_sql": record["gold_sql"],
            "correct": {},
            "predictions": {},
            "costs": {},
            "adaptive": {
                "checkpoint": selected["checkpoint"],
                "action": selected["action"],
                "trace": trace,
            },
        }
        row["correct"]["adaptive"] = selected["fp"] is not None and selected["fp"] == gold
        row["predictions"]["adaptive"] = selected["sql"]
        row["costs"]["adaptive"] = selected["cost"]
        row["correct"][f"sc@{max_k}"] = bool(max_groups and max_groups[0]["fingerprint"] == gold)
        row["predictions"][f"sc@{max_k}"] = max_groups[0]["sql"] if max_groups else ""
        row["costs"][f"sc@{max_k}"] = {
            "generation_samples": max_k,
            "generation_tokens": sum(c.get("n_tokens", 0) for c in candidates[:max_k]),
            "merge_samples": 0,
            "merge_tokens": 0,
            "output_tokens": sum(c.get("n_tokens", 0) for c in candidates[:max_k]),
            "csc_called": False,
        }
        for variant, label in (("author", "author_csc"), ("lora", "lora_csc")):
            output = merges[variant][max_k][0][index]
            row["correct"][label] = output["selected_fp"] == gold
            row["predictions"][label] = output["selected_sql"]
            row["costs"][label] = {
                "generation_samples": max_k,
                "generation_tokens": sum(c.get("n_tokens", 0) for c in candidates[:max_k]),
                "merge_samples": len(output["merge_candidates"]),
                "merge_tokens": output["merge_tokens"],
                "output_tokens": sum(c.get("n_tokens", 0) for c in candidates[:max_k]) + output["merge_tokens"],
                "csc_called": output["csc_called"],
            }
        rows.append(row)
    summary = {
        method: _method_summary(rows, method, cfg["evaluation"]["bootstrap_samples"], seed)
        for method in ("adaptive", f"sc@{max_k}", "author_csc", "lora_csc")
    }
    comparisons = {
        "adaptive_vs_sc": _paired(rows, "adaptive", f"sc@{max_k}", cfg["evaluation"]["bootstrap_samples"], seed),
        "adaptive_vs_author_csc": _paired(rows, "adaptive", "author_csc", cfg["evaluation"]["bootstrap_samples"], seed),
        "adaptive_lora_vs_author_csc": _paired(rows, "lora_csc", "author_csc", cfg["evaluation"]["bootstrap_samples"], seed),
    }
    return {
        "records": rows,
        "summary": summary,
        "paired_comparisons": comparisons,
        "execution_manifest": execution_manifest,
        "max_k": max_k,
    }


def evaluate_adaptive_csc(cfg, split="evaluation", seed=42, *, policy_parameters=None):
    if split == "evaluation":
        selected, selected_hash = _load_selected_policy(cfg, split)
        policy_parameters = selected["parameters"]
        protocol_path = Path(cfg["experiment_dir"]) / "final_protocol.json"
        if not protocol_path.is_file():
            raise ValueError("Run freeze before adaptive-csc-eval on split evaluation.")
    else:
        selected_hash = None
        policy_parameters = policy_parameters or {}
    checkpoints = adaptive_checkpoints(cfg, split, policy_parameters.get("checkpoints") if policy_parameters else None)
    if split == "evaluation":
        from research.slm_sql.protocol import verify_freeze

        verify_freeze(cfg, seed)
        frozen_ks = set(read_json(Path(cfg["experiment_dir"]) / "final_protocol.json")["spec"]["k_values"])
        if not set(checkpoints).issubset(frozen_ks):
            raise ValueError("Evaluation adaptive checkpoints must be declared in the final freeze.")
        for k in checkpoints:
            verify_freeze(cfg, seed, k)
    records, execution_manifest = load_execution(cfg, split, seed)
    merges = _load_merge_checkpoints(cfg, split, seed, records, execution_manifest, checkpoints)
    result = _score_policy(cfg, split, seed, policy_parameters, checkpoints, merges)
    spec = {
        "protocol": PROTOCOL_VERSION,
        "stage": "adaptive-csc-eval-v1",
        "split": split,
        "seed": seed,
        "checkpoints": checkpoints,
        "policy": policy_parameters,
        "selected_policy_sha256": selected_hash,
        "execution": execution_manifest["sha256"],
        "merge_manifests": {
            variant: {str(k): merges[variant][k][1]["sha256"] for k in checkpoints}
            for variant in ("author", "lora")
        },
    }
    result["spec"] = spec
    result["spec_signature"] = digest(spec)
    result["accuracy_basis"] = "offline_prefix_exact"
    result["policy_provenance"] = selected.get("validation_artifacts") if split == "evaluation" else None
    path = Path(cfg["experiment_dir"]) / "reports" / f"{split}-seed{seed}.adaptive-csc-eval.json"
    atomic_json(path, result)
    return {"path": str(path), "summary": result["summary"], "accuracy_basis": result["accuracy_basis"]}


def adaptive_sweep(cfg, split="validation", seed=42, *, policy_name=None, thresholds=None):
    if split != "validation":
        raise ValueError("Adaptive policy sweep is allowed only on split validation.")
    checkpoints = adaptive_checkpoints(cfg, split)
    records, execution_manifest = load_execution(cfg, split, seed)
    merges = _load_merge_checkpoints(cfg, split, seed, records, execution_manifest, checkpoints)
    policy_name = policy_name or cfg["adaptive"]["policy"]
    thresholds = thresholds or [cfg["adaptive"]["threshold"]]
    results = []
    for threshold in thresholds:
        if policy_name == "margin":
            valid_threshold = threshold >= 0 and threshold != float("inf")
        else:
            valid_threshold = 0 < threshold <= 1
        if not valid_threshold:
            raise ValueError(
                "Margin sweep thresholds must be finite and nonnegative."
                if policy_name == "margin" else "Sweep thresholds must be in (0, 1]."
            )
        parameters = {
            "policy": policy_name,
            "threshold": threshold,
            "min_k": cfg["adaptive"]["min_k"],
            "max_k": cfg["adaptive"]["max_k"],
            "checkpoints": checkpoints,
        }
        if policy_name == "margin":
            parameters.update(
                margin=cfg["adaptive"]["margin"],
                entropy_threshold=threshold,
            )
        if policy_name == "router":
            parameters.update(
                stop_threshold=threshold,
                csc_threshold=cfg["adaptive"].get("csc_threshold", 0.7),
            )
        scored = _score_policy(cfg, split, seed, parameters, checkpoints, merges)
        metrics = scored["summary"]["adaptive"]
        results.append({
            "parameters": parameters,
            "ex": metrics["ex"],
            "mean_samples": metrics["mean_samples"],
            "mean_tokens": metrics["mean_tokens"],
            "csc_calls": metrics["csc_calls"],
        })
    results.sort(key=lambda row: (-row["ex"], row["mean_samples"], row["mean_tokens"]))
    selected = results[0]
    report = {
        "protocol": PROTOCOL_VERSION,
        "stage": "adaptive-policy-sweep-v1",
        "source_split": "validation",
        "seed": seed,
        "checkpoints": checkpoints,
        "execution": execution_manifest["sha256"],
        "merge_manifests": {
            variant: {str(k): merges[variant][k][1]["sha256"] for k in checkpoints}
            for variant in ("author", "lora")
        },
        "results": results,
        "selected": selected,
        "accuracy_basis": "offline_prefix_exact",
    }
    report_path = Path(cfg["experiment_dir"]) / "reports" / f"validation-seed{seed}.adaptive-sweep.json"
    atomic_json(report_path, report)
    selected_policy = {
        "version": 1,
        "source_split": "validation",
        "parameters": selected["parameters"],
        "checkpoints": checkpoints,
        "validation_artifacts": {
            "report": {"path": str(report_path), "sha256": file_hash(report_path)},
            "execution": execution_manifest["sha256"],
            "merges": report["merge_manifests"],
        },
        "accuracy_basis": "offline_prefix_exact",
    }
    selected_path = _selected_policy_path(cfg)
    atomic_json(selected_path, selected_policy)
    return {"path": str(report_path), "selected_policy": str(selected_path), "selected": selected}
