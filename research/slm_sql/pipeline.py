"""Cached generation -> execution -> author CSC / adapted CSC stages."""

from __future__ import annotations

import time
from pathlib import Path

from core.sql_extractor import extract_sql
from research.generation_metrics import has_answer_format
from research.slm_sql import PROTOCOL_VERSION
from research.slm_sql.artifacts import (
    Journal,
    exclusive_lock,
    atomic_json,
    completed_artifact,
    digest,
    file_hash,
    read_json,
)
from research.slm_sql.backend import VLLMBackend
from research.slm_sql.config import artifact_path, merge_artifact_path
from research.slm_sql.data import get_schema, load_split
from research.slm_sql.execution import EXECUTION_VERSION, ExecutionCache
from research.slm_sql.policy_view import policy_view
from research.slm_sql.prompts import PROMPT_VERSION, generation_messages, merge_messages, top_groups


def _sampling(cfg, split, mode="stochastic"):
    settings = {k: v for k, v in cfg["generation"].items() if not k.endswith("_samples")}
    if mode == "stochastic":
        return {**settings, "samples": cfg["generation"][f"{split}_samples"]}
    if mode == "greedy":
        return {**settings, "samples": 1, "temperature": 0.0, "top_p": 1.0}
    raise ValueError("mode must be stochastic or greedy.")


def _generation_stage(mode: str) -> str:
    if mode == "stochastic":
        return "generation"
    if mode == "greedy":
        return "greedy.generation"
    raise ValueError("mode must be stochastic or greedy.")


def _execution_stage(mode: str) -> str:
    if mode == "stochastic":
        return "execution"
    if mode == "greedy":
        return "greedy.execution"
    raise ValueError("mode must be stochastic or greedy.")


def _seed(seed: int, record_id: str, role: str) -> int:
    return (seed + int(digest([record_id, role])[:8], 16)) % (2**31)


def _bound_backend(
    cfg, role, settings, backend_factory, *, frozen=None, adapter=None, expected_packages=None
):
    path = Path(cfg["experiment_dir"]) / "models" / f"{role}.json"
    with exclusive_lock(path.with_suffix(".lock")):
        requested = {"model": settings["model"], "revision": settings["revision"]}
        binding = read_json(path) if path.exists() else None
        if binding and binding["requested"] != requested:
            raise ValueError("Model configuration differs from the experiment's frozen checkpoint.")
        identity = binding["identity"] if binding else frozen
        if frozen and identity != frozen:
            raise ValueError("Artifact and experiment refer to different model weights.")
        if (
            expected_packages
            and backend_factory is VLLMBackend
            and VLLMBackend.runtime_versions() != expected_packages
        ):
            raise ValueError(
                "Inference package versions changed; restore the versions recorded in the author/cache manifest."
            )
        backend = backend_factory(
            settings, frozen_identity=identity, **({"adapter": adapter} if role == "merger" else {})
        )
        if expected_packages and backend.versions != expected_packages:
            raise ValueError(
                "Inference package versions changed; restore the original environment before resuming/comparing."
            )
        if identity and backend.identity != identity:
            raise ValueError("Backend loaded different model weights from the frozen identity.")
        if binding is None:
            atomic_json(path, {"requested": requested, "identity": backend.identity})
        from research.reproducibility import capture_runtime
        runtime_path = Path(cfg["experiment_dir"]) / f"runtime.{role}.json"
        if not runtime_path.exists():
            capture_runtime(cfg["experiment_dir"], role)
        return backend


def _generation_spec(cfg, split, seed, data_manifest, mode="stochastic"):
    return {
        "protocol": PROTOCOL_VERSION,
        "prompt": PROMPT_VERSION,
        "stage": "generation",
        "mode": mode,
        "split": split,
        "seed": seed,
        "data": data_manifest["splits"][split]["sha256"],
        "prepared": data_manifest["splits"][split]["contract"],
        "sampling": _sampling(cfg, split, mode),
        "implementation": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("backend.py", "prompts.py")
        },
    }


def load_generation(cfg, split, seed, *, mode="stochastic"):
    rows, data_manifest = load_split(cfg, split)
    generated, manifest = completed_artifact(
        artifact_path(cfg, split, _generation_stage(mode), seed)
    )
    if manifest["signature"] != digest(_generation_spec(cfg, split, seed, data_manifest, mode)):
        raise ValueError("Generation cache no longer matches the requested protocol/configuration.")
    if [r["id"] for r in generated] != [r["id"] for r in rows]:
        raise ValueError("Generation question IDs do not match the prepared split.")
    return generated, manifest


def generate_stage(
    cfg: dict, split: str, seed: int = 42, *, mode="stochastic", backend_factory=VLLMBackend
) -> dict:
    records, data_manifest = load_split(cfg, split)
    settings = _sampling(cfg, split, mode)
    path = artifact_path(cfg, split, _generation_stage(mode), seed)
    spec = _generation_spec(cfg, split, seed, data_manifest, mode)
    seed_role = "generation" if mode == "stochastic" else "greedy-generation"
    with Journal(path, spec, [r["id"] for r in records]) as journal:
        if journal.complete:
            return {"path": str(path), "questions": len(records), "reused": True}
        backend = None
        for start in range(len(journal.records), len(records), settings["batch_size"]):
            batch = records[start : start + settings["batch_size"]]
            messages = [generation_messages(r, get_schema(cfg, r["db_id"])) for r in batch]
            if backend is None:
                backend = _bound_backend(
                    cfg,
                    "generator",
                    settings,
                    backend_factory,
                    frozen=journal.manifest.get("model_identity"),
                    expected_packages=journal.manifest.get("packages"),
                )
                journal.manifest.update(model_identity=backend.identity, packages=backend.versions)
                atomic_json(journal.manifest_path, journal.manifest)
            started = time.perf_counter()
            outputs = backend.generate(
                messages,
                n=settings["samples"],
                seeds=[_seed(seed, r["id"], seed_role) for r in batch],
            )
            elapsed = time.perf_counter() - started
            if len(outputs) != len(batch) or any(len(cs) != settings["samples"] for cs in outputs):
                raise RuntimeError("Backend returned incomplete generation requests.")
            for record, completions in zip(batch, outputs):
                candidates = []
                for i, completion in enumerate(completions):
                    try:
                        sql = extract_sql(completion["raw_text"])
                        parse_error = None
                    except ValueError as exc:
                        sql, parse_error = "", str(exc)
                    candidates.append(
                        {**completion, "index": i, "sql": sql, "parse_error": parse_error,
                         "format_error": not has_answer_format(completion["raw_text"]),
                         "seed": _seed(seed, record["id"], seed_role)}
                    )
                journal.append(
                    {
                        **record,
                        "candidates": candidates,
                        "amortized_generation_s": elapsed / len(batch),
                    }
                )
        journal.finish()
    return {"path": str(path), "questions": len(records), "reused": False}


def _execution_spec(cfg, split, seed, generation_manifest, mode="stochastic"):
    return {
        "stage": "execution",
        "mode": mode,
        "split": split,
        "seed": seed,
        "generation": generation_manifest["sha256"],
        "protocol": EXECUTION_VERSION,
        "execution": cfg["execution"],
        "preview_rows": cfg["merger"]["result_preview_rows"],
        "implementation": file_hash(Path(__file__).with_name("execution.py")),
    }


def load_execution(cfg, split, seed, *, mode="stochastic"):
    _, generation_manifest = load_generation(cfg, split, seed, mode=mode)
    records, manifest = completed_artifact(
        artifact_path(cfg, split, _execution_stage(mode), seed)
    )
    if manifest["signature"] != digest(_execution_spec(cfg, split, seed, generation_manifest, mode)):
        raise ValueError("Execution cache no longer matches the requested generation/scorer.")
    return records, manifest


def execute_stage(cfg: dict, split: str, seed: int = 42, *, mode="stochastic") -> dict:
    generated, generation_manifest = load_generation(cfg, split, seed, mode=mode)
    path = artifact_path(cfg, split, _execution_stage(mode), seed)
    spec = _execution_spec(cfg, split, seed, generation_manifest, mode)
    with Journal(path, spec, [r["id"] for r in generated]) as journal:
        if journal.complete:
            return {"path": str(path), "questions": len(generated), "reused": True}
        with ExecutionCache(
            Path(cfg["experiment_dir"]) / "execution_cache.sqlite",
            cfg["execution"],
            cfg["merger"]["result_preview_rows"],
        ) as cache:
            for record in generated[len(journal.records) :]:
                gold = cache.run(record["gold_sql"], record)
                if gold["error"]:
                    raise ValueError(f"Gold execution failed for {record['id']}: {gold['error']}")
                candidates = [
                    {**c, "execution": cache.run(c["sql"], record)} for c in record["candidates"]
                ]
                cache.commit()
                journal.append({**record, "gold_execution": gold, "candidates": candidates})
        journal.finish()
    return {"path": str(path), "questions": len(generated), "reused": False}


def _adaptive_policy(cfg, split="evaluation"):
    settings = cfg["adaptive"]
    policy_name = settings["policy"]
    budget = min(settings["max_k"], cfg["generation"][f"{split}_samples"])
    limits = {"min_k": min(settings["min_k"], budget), "max_k": budget}
    if policy_name == "beta":
        from research.policies.beta_stopping import BetaStoppingPolicy

        return BetaStoppingPolicy(threshold=settings["threshold"], **limits)
    if policy_name == "margin":
        from research.policies.margin_entropy import MarginEntropyPolicy

        return MarginEntropyPolicy(
            margin=settings["margin"], entropy_threshold=settings["entropy_threshold"], **limits
        )
    if policy_name == "router":
        from research.policies.three_action_router import ThreeActionRouterPolicy

        return ThreeActionRouterPolicy(
            stop_threshold=settings.get("stop_threshold", 0.90),
            csc_threshold=settings.get("csc_threshold", 0.70),
            **limits,
        )
    raise ValueError("The adaptive generator supports beta, margin or router policies.")


def _adaptive_spec(cfg, split, seed, data_manifest):
    root = Path(__file__).parents[1]
    return {
        "protocol": PROTOCOL_VERSION,
        "prompt": PROMPT_VERSION,
        "stage": "adaptive-online-v2",
        "split": split,
        "seed": seed,
        "data": data_manifest["splits"][split]["sha256"],
        "prepared": data_manifest["splits"][split]["contract"],
        "sampling": _sampling(cfg, split),
        "adaptive": cfg["adaptive"],
        "execution": cfg["execution"],
        "execution_version": EXECUTION_VERSION,
        "preview_rows": cfg["merger"]["result_preview_rows"],
        "implementation": {
            str(path.relative_to(root)): file_hash(path)
            for path in (
                Path(__file__), Path(__file__).with_name("backend.py"),
                Path(__file__).with_name("prompts.py"), Path(__file__).with_name("execution.py"),
                root / "policies/beta_stopping.py", root / "policies/margin_entropy.py",
                root / "policies/three_action_router.py", root / "slm_sql/policy_view.py",
            )
        },
    }


def load_adaptive(cfg, split, seed):
    prepared, data_manifest = load_split(cfg, split)
    records, manifest = completed_artifact(artifact_path(cfg, split, "adaptive", seed))
    if manifest["signature"] != digest(_adaptive_spec(cfg, split, seed, data_manifest)):
        raise ValueError("Adaptive cache no longer matches the requested protocol/configuration.")
    if [r["id"] for r in records] != [r["id"] for r in prepared]:
        raise ValueError("Adaptive question IDs do not match the prepared split.")
    return records, manifest


def adaptive_stage(
    cfg: dict, split: str = "evaluation", seed: int = 42, *, backend_factory=VLLMBackend,
) -> dict:
    """Generate only the rounds needed by a gold-free execution-vote policy.

    No fixed candidate cache is consumed. A completed question is journaled before
    moving on; an interrupted question is retried with the same round seeds.
    Generator and merger remain separate processes, also for adaptive CSC.
    """
    if not cfg["adaptive"]["enabled"]:
        return {"path": str(artifact_path(cfg, split, "adaptive", seed)), "disabled": True}
    records, data_manifest = load_split(cfg, split)
    settings = _sampling(cfg, split)
    policy = _adaptive_policy(cfg, split)
    path = artifact_path(cfg, split, "adaptive", seed)
    spec = _adaptive_spec(cfg, split, seed, data_manifest)
    with Journal(path, spec, [r["id"] for r in records]) as journal:
        if journal.complete:
            return {"path": str(path), "questions": len(records), "reused": True}
        backend = None
        with ExecutionCache(
            Path(cfg["experiment_dir"]) / "execution_cache.sqlite",
            cfg["execution"], cfg["merger"]["result_preview_rows"],
        ) as cache:
            for record in records[len(journal.records):]:
                # Gold is used only to validate/score, never in policy features or prompts.
                gold = cache.run(record["gold_sql"], record)
                if gold["error"]:
                    raise ValueError(f"Gold execution failed for {record['id']}: {gold['error']}")
                messages = [generation_messages(record, get_schema(cfg, record["db_id"]))]
                seen, rounds = [], []
                generation_seconds = 0.0
                started_question = time.perf_counter()
                action = "SAMPLE_MORE"
                while len(seen) < policy.max_k:
                    count = min(cfg["adaptive"]["batch_size"], policy.max_k - len(seen))
                    if backend is None:
                        backend = _bound_backend(
                            cfg, "generator", settings, backend_factory,
                            frozen=journal.manifest.get("model_identity"),
                            expected_packages=journal.manifest.get("packages"),
                        )
                        journal.manifest.update(model_identity=backend.identity, packages=backend.versions)
                        atomic_json(journal.manifest_path, journal.manifest)
                        # Loading weights is not per-question inference latency.
                        started_question = time.perf_counter()
                    started = time.perf_counter()
                    round_seed = _seed(seed, record["id"], f"adaptive-{len(seen)}")
                    outputs = backend.generate(messages, n=count, seeds=[round_seed])
                    elapsed = time.perf_counter() - started
                    generation_seconds += elapsed
                    if len(outputs) != 1 or len(outputs[0]) != count:
                        raise RuntimeError("Backend returned incomplete adaptive generation requests.")
                    for completion in outputs[0]:
                        try:
                            sql, parse_error = extract_sql(completion["raw_text"]), None
                        except ValueError as exc:
                            sql, parse_error = "", str(exc)
                        seen.append({
                            **completion, "index": len(seen), "sql": sql,
                            "parse_error": parse_error,
                            "format_error": not has_answer_format(completion["raw_text"]),
                            "seed": round_seed, "execution": cache.run(sql, record),
                        })
                    # Give policies inference-time observations only; never pass gold labels.
                    observed = policy_view(seen)
                    stop, action = policy.should_stop(observed, len(seen))
                    rounds.append({"samples": count, "cumulative_samples": len(seen),
                                   "seed": round_seed, "generation_s": elapsed, "action": action})
                    if stop:
                        break
                reached_budget = len(seen) >= policy.max_k
                groups = top_groups(seen)
                budget_unresolved = reached_budget and action == "SAMPLE_MORE"
                policy_action = action
                csc_at_budget = (
                    cfg["adaptive"].get("call_csc_at_budget", False)
                    and budget_unresolved
                    and len(groups) >= 2
                )
                if csc_at_budget:
                    action = "CALL_CSC"
                elif budget_unresolved:
                    action = "STOP"
                selected = groups[0] if groups else None
                selected_fp = selected["fingerprint"] if selected else None
                inference_s = time.perf_counter() - started_question
                cache.commit()
                stop_reason = (
                    "router" if policy_action == "CALL_CSC"
                    else "budget_csc" if csc_at_budget
                    else "consensus" if policy_action == "STOP"
                    else "budget"
                )
                journal.append({
                    **record, "candidates": seen, "gold_execution": gold,
                    "adaptive_policy": cfg["adaptive"]["policy"], "adaptive_action": action,
                    "stop_reason": stop_reason,
                    "rounds": rounds, "n_samples_used": len(seen),
                    "selected_fp": selected_fp, "selected_sql": selected["sql"] if selected else "",
                    "correct": bool(selected_fp and selected_fp == gold["fingerprint"]),
                    "amortized_generation_s": generation_seconds,
                    "online_inference_s": inference_s,
                })
        journal.finish()
    return {"path": str(path), "questions": len(records), "reused": False}


def merge_stage(
    cfg: dict, split="evaluation", seed=42, *, variant="author", k=None,
    source="fixed", backend_factory=VLLMBackend,
) -> dict:
    if variant not in ("author", "lora"):
        raise ValueError("variant must be author or lora.")
    if source == "fixed":
        records, execution_manifest = load_execution(cfg, split, seed)
        n = cfg["generation"][f"{split}_samples"]
        k = n if k is None else k
        if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n:
            raise ValueError(f"k must be between 1 and the cached sample count {n}.")
    elif source == "adaptive":
        if k is not None:
            raise ValueError("Adaptive CSC uses each question's stopped candidate set; omit k.")
        records, execution_manifest = load_adaptive(cfg, split, seed)
    else:
        raise ValueError("CSC source must be fixed or adaptive.")
    if split == "evaluation" and variant == "lora":
        from research.slm_sql.protocol import verify_freeze
        verify_freeze(cfg, seed, k)
    settings = cfg["merger"]
    adapter, adapter_identity, frozen_base, baseline_packages = None, None, None, None
    if variant == "lora":
        from research.slm_sql.adaptation import validate_adapter

        adapter, adapter_identity, frozen_base = validate_adapter(cfg, split, seed, k, source=source)
        _, author_baseline = completed_artifact(
            merge_artifact_path(cfg, split, seed, "author", k, source=source)
        )
        baseline_packages = author_baseline.get("packages")
    spec = {
        "stage": "csc",
        "protocol": PROTOCOL_VERSION,
        "prompt": PROMPT_VERSION,
        "split": split,
        "seed": seed,
        "k": k,
        "variant": variant,
        "execution": execution_manifest["sha256"],
        "merger": settings,
        "adapter": adapter_identity,
        "implementation": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("backend.py", "prompts.py")
        },
        "rules": {
            "trigger": "two_or_more_executable_result_groups",
            "vote": EXECUTION_VERSION,
            "ties": "first_generated",
            "all_merge_invalid": "failure",
            "schema": "full_ddl",
        },
    }
    if source == "adaptive":
        spec["source"] = source
    path = merge_artifact_path(cfg, split, seed, variant, k, source=source)
    with Journal(path, spec, [r["id"] for r in records]) as journal:
        if journal.complete:
            return {"path": str(path), "questions": len(records), "reused": True}
        backend = None
        with ExecutionCache(
            Path(cfg["experiment_dir"]) / "execution_cache.sqlite",
            cfg["execution"],
            settings["result_preview_rows"],
        ) as cache:
            for start in range(len(journal.records), len(records), settings["batch_size"]):
                batch = records[start : start + settings["batch_size"]]
                groups = [top_groups(r["candidates"], k) for r in batch]
                eligible = [
                    i for i, gs in enumerate(groups)
                    if len(gs) >= 2
                    and (
                        source != "adaptive"
                        or batch[i].get("adaptive_action") == "CALL_CSC"
                    )
                ]
                merge_outputs = {}
                elapsed = 0.0
                if eligible:
                    messages = [
                        merge_messages(batch[i], get_schema(cfg, batch[i]["db_id"]), groups[i][:2])
                        for i in eligible
                    ]
                    if backend is None:
                        frozen = journal.manifest.get("model_identity") or frozen_base
                        backend = _bound_backend(
                            cfg,
                            "merger",
                            settings,
                            backend_factory,
                            frozen=frozen,
                            adapter=adapter,
                            expected_packages=journal.manifest.get("packages") or baseline_packages,
                        )
                        if frozen_base and backend.identity != frozen_base:
                            raise ValueError(
                                "LoRA CSC and author CSC must use identical base weights."
                            )
                        journal.manifest.update(
                            model_identity=backend.identity, packages=backend.versions
                        )
                        atomic_json(journal.manifest_path, journal.manifest)
                    started = time.perf_counter()
                    outputs = backend.generate(
                        messages,
                        n=settings["samples"],
                        seeds=[_seed(seed, batch[i]["id"], "csc") for i in eligible],
                    )
                    elapsed = time.perf_counter() - started
                    if len(outputs) != len(eligible) or any(
                        len(cs) != settings["samples"] for cs in outputs
                    ):
                        raise RuntimeError("Backend returned incomplete merge requests.")
                    merge_outputs = dict(zip(eligible, outputs))
                for i, (record, gs) in enumerate(zip(batch, groups)):
                    candidates = []
                    for completion in merge_outputs.get(i, []):
                        try:
                            sql = extract_sql(completion["raw_text"])
                        except ValueError:
                            sql = ""
                        candidates.append(
                            {**completion, "sql": sql, "execution": cache.run(sql, record)}
                        )
                    cache.commit()
                    merged_groups = top_groups(candidates)
                    selected = (
                        merged_groups[0]
                        if candidates and merged_groups
                        else gs[0]
                        if not candidates and gs
                        else None
                    )
                    output_record = {
                        "id": record["id"],
                        "db_id": record["db_id"],
                        "k": k,
                        "csc_called": i in merge_outputs,
                        "sc_sql": gs[0]["sql"] if gs else "",
                        "sc_fp": gs[0]["fingerprint"] if gs else None,
                        "top2_fps": [g["fingerprint"] for g in gs[:2]],
                        "selected_sql": selected["sql"] if selected else "",
                        "selected_fp": selected["fingerprint"] if selected else None,
                        "merge_candidates": candidates,
                        "merge_tokens": sum(c["n_tokens"] for c in candidates),
                        "amortized_merge_s": elapsed / len(eligible)
                        if i in merge_outputs
                        else 0.0,
                    }
                    if source == "adaptive":
                        output_record["n_samples_used"] = len(record["candidates"])
                    journal.append(output_record)
        journal.finish()
    return {"path": str(path), "questions": len(records), "reused": False}
