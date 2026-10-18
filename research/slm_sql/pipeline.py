"""Cached generation -> execution -> author CSC / adapted CSC stages."""

from __future__ import annotations

import time
from pathlib import Path

from core.sql_extractor import extract_sql
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
from research.slm_sql.config import artifact_path
from research.slm_sql.data import get_schema, load_split
from research.slm_sql.execution import EXECUTION_VERSION, ExecutionCache
from research.slm_sql.prompts import PROMPT_VERSION, generation_messages, merge_messages, top_groups


def _sampling(cfg, split):
    settings = {k: v for k, v in cfg["generation"].items() if not k.endswith("_samples")}
    return {**settings, "samples": cfg["generation"][f"{split}_samples"]}


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
        return backend


def _generation_spec(cfg, split, seed, data_manifest):
    return {
        "protocol": PROTOCOL_VERSION,
        "prompt": PROMPT_VERSION,
        "stage": "generation",
        "split": split,
        "seed": seed,
        "data": data_manifest["splits"][split]["sha256"],
        "prepared": data_manifest["splits"][split]["contract"],
        "sampling": _sampling(cfg, split),
        "implementation": {
            name: file_hash(Path(__file__).with_name(name)) for name in ("backend.py", "prompts.py")
        },
    }


def load_generation(cfg, split, seed):
    rows, data_manifest = load_split(cfg, split)
    generated, manifest = completed_artifact(artifact_path(cfg, split, "generation", seed))
    if manifest["signature"] != digest(_generation_spec(cfg, split, seed, data_manifest)):
        raise ValueError("Generation cache no longer matches the requested protocol/configuration.")
    if [r["id"] for r in generated] != [r["id"] for r in rows]:
        raise ValueError("Generation question IDs do not match the prepared split.")
    return generated, manifest


def generate_stage(cfg: dict, split: str, seed: int = 42, *, backend_factory=VLLMBackend) -> dict:
    records, data_manifest = load_split(cfg, split)
    settings = _sampling(cfg, split)
    path = artifact_path(cfg, split, "generation", seed)
    spec = _generation_spec(cfg, split, seed, data_manifest)
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
                seeds=[_seed(seed, r["id"], "generation") for r in batch],
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
                        {**completion, "index": i, "sql": sql, "parse_error": parse_error}
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


def _execution_spec(cfg, split, seed, generation_manifest):
    return {
        "stage": "execution",
        "split": split,
        "seed": seed,
        "generation": generation_manifest["sha256"],
        "protocol": EXECUTION_VERSION,
        "execution": cfg["execution"],
        "preview_rows": cfg["merger"]["result_preview_rows"],
        "implementation": file_hash(Path(__file__).with_name("execution.py")),
    }


def load_execution(cfg, split, seed):
    _, generation_manifest = load_generation(cfg, split, seed)
    records, manifest = completed_artifact(artifact_path(cfg, split, "execution", seed))
    if manifest["signature"] != digest(_execution_spec(cfg, split, seed, generation_manifest)):
        raise ValueError("Execution cache no longer matches the requested generation/scorer.")
    return records, manifest


def execute_stage(cfg: dict, split: str, seed: int = 42) -> dict:
    generated, generation_manifest = load_generation(cfg, split, seed)
    path = artifact_path(cfg, split, "execution", seed)
    spec = _execution_spec(cfg, split, seed, generation_manifest)
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


def merge_stage(
    cfg: dict, split="evaluation", seed=42, *, variant="author", k=None, backend_factory=VLLMBackend
) -> dict:
    if variant not in ("author", "lora"):
        raise ValueError("variant must be author or lora.")
    records, execution_manifest = load_execution(cfg, split, seed)
    n = cfg["generation"][f"{split}_samples"]
    k = n if k is None else k
    if not isinstance(k, int) or not 1 <= k <= n:
        raise ValueError(f"k must be between 1 and the cached sample count {n}.")
    settings = cfg["merger"]
    adapter, adapter_identity, frozen_base, baseline_packages = None, None, None, None
    if variant == "lora":
        from research.slm_sql.adaptation import validate_adapter

        adapter, adapter_identity, frozen_base = validate_adapter(cfg, split, seed, k)
        _, author_baseline = completed_artifact(artifact_path(cfg, split, f"author.csc-k{k}", seed))
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
    path = artifact_path(cfg, split, f"{variant}.csc-k{k}", seed)
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
                eligible = [i for i, gs in enumerate(groups) if len(gs) >= 2]
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
                    journal.append(
                        {
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
                    )
        journal.finish()
    return {"path": str(path), "questions": len(records), "reused": False}
