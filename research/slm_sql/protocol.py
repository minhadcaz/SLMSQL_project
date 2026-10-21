"""Prospective final-run freeze and paired inference-seed reports (CPU only)."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from evaluation.final_eval import bootstrap_mean_ci
from research.reproducibility import source_state
from research.slm_sql.artifacts import atomic_json, digest, file_hash, read_json
from research.slm_sql.config import artifact_path


def load_selected_policy(cfg: dict, *, required: bool = True) -> tuple[dict | None, str | None]:
    """Load and verify the validation-selected adaptive policy provenance."""
    root = Path(cfg["experiment_dir"])
    selected_path = root / "adaptive" / "selected_policy.json"
    if not selected_path.is_file():
        if required:
            raise ValueError(
                "Selected adaptive policy is missing. Run "
                "adaptive-sweep --split validation before final evaluation."
            )
        return None, None
    selected = read_json(selected_path)
    if selected.get("source_split") != "validation":
        raise ValueError("Selected adaptive policy must be selected from split validation.")
    parameters = selected.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Selected adaptive policy parameters must be a mapping.")
    provenance = selected.get("validation_artifacts")
    if not isinstance(provenance, dict):
        raise ValueError("Selected adaptive policy has no validation artifact provenance.")
    report_info = provenance.get("report")
    if not isinstance(report_info, dict) or not report_info.get("path") or not report_info.get("sha256"):
        raise ValueError("Selected adaptive policy is missing its validation report hash.")
    report_path = Path(report_info["path"])
    if not report_path.is_absolute():
        report_path = root / report_path
    if not report_path.is_file() or file_hash(report_path) != report_info["sha256"]:
        raise ValueError("Selected adaptive policy validation report is missing or modified.")
    report = read_json(report_path)
    if report.get("source_split") != "validation" or report.get("selected", {}).get("parameters") != parameters:
        raise ValueError("Selected adaptive policy does not match its validation report.")
    if report.get("execution") != provenance.get("execution"):
        raise ValueError("Selected adaptive policy execution provenance does not match its report.")
    if report.get("merge_manifests") != provenance.get("merges"):
        raise ValueError("Selected adaptive policy merge provenance does not match its report.")
    if report.get("checkpoints") != selected.get("checkpoints"):
        raise ValueError("Selected adaptive policy checkpoints do not match its validation report.")
    if not isinstance(provenance.get("execution"), str) or not provenance["execution"]:
        raise ValueError("Selected adaptive policy is missing validation execution provenance.")
    merges = provenance.get("merges")
    if not isinstance(merges, dict) or not all(isinstance(merges.get(name), dict) for name in ("author", "lora")):
        raise ValueError("Selected adaptive policy is missing validation merge provenance.")
    seed = report.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("Selected adaptive policy validation seed is invalid.")
    from research.slm_sql.artifacts import completed_artifact
    from research.slm_sql.pipeline import load_execution

    _, execution_manifest = load_execution(cfg, "validation", seed)
    if execution_manifest.get("sha256") != provenance["execution"]:
        raise ValueError("Selected adaptive policy validation execution artifact changed.")
    for variant in ("author", "lora"):
        for checkpoint, expected_hash in merges[variant].items():
            merge_path = artifact_path(cfg, "validation", f"{variant}.csc-k{checkpoint}", seed)
            _, manifest = completed_artifact(merge_path)
            if manifest.get("sha256") != expected_hash:
                raise ValueError("Selected adaptive policy validation merge artifact changed.")
    return selected, file_hash(selected_path)


def _contract(cfg, seeds, ks):
    from research.slm_sql.adaptation import _check_weights

    root = Path(cfg["experiment_dir"])
    training = read_json(root / "adapter/training_manifest.json")
    if training.get("status") != "complete" or training["spec"]["training"] != cfg["training"]:
        raise ValueError("Complete the configured adapter before freezing final evaluation.")
    _check_weights(root / "adapter/final", training["adapter_files"])
    prepared = read_json(root / "data/manifest.json")
    _, selected_policy_hash = load_selected_policy(cfg, required=cfg["adaptive"]["enabled"])
    return {
        "version": 1, "config": cfg, "inference_seeds": seeds, "k_values": ks,
        "data_signature": prepared["signature"],
        "evaluation_sha256": prepared["splits"]["evaluation"]["sha256"],
        "generator": read_json(root / "models/generator.json"),
        "merger": read_json(root / "models/merger.json"),
        "adapter_signature": training["signature"], "adapter_files": training["adapter_files"],
        "selected_policy": selected_policy_hash,
        "source": {key: value for key, value in source_state().items() if key != "git_dirty"},
    }


def freeze(cfg, seeds=(42, 43, 44), ks=None):
    seeds = list(seeds)
    ks = list(cfg["evaluation"]["sampling_k_values"] if ks is None else ks)
    if not seeds or len(seeds) != len(set(seeds)) or any(not isinstance(s, int) or isinstance(s, bool) or s < 0 for s in seeds):
        raise ValueError("Final seeds must be unique nonnegative integers.")
    declared_ks = set(cfg["evaluation"]["sampling_k_values"])
    if (
        not ks
        or len(ks) != len(set(ks))
        or any(
            not isinstance(k, int)
            or isinstance(k, bool)
            or not 1 <= k <= cfg["generation"]["evaluation_samples"]
            for k in ks
        )
        or not set(ks).issubset(declared_ks)
    ):
        raise ValueError("Final k must be declared in evaluation.sampling_k_values.")
    root = Path(cfg["experiment_dir"])
    path = root / "final_protocol.json"
    if not path.exists() and (
        list((root / "reports").glob("evaluation-*.comparison.json"))
        or list(root.glob("seed*/evaluation.lora.csc-*.jsonl"))
        or list(root.glob("seed*/evaluation.lora.adaptive-csc.jsonl"))
    ):
        raise ValueError("Final LoRA predictions already exist; use a new experiment to claim a prospective freeze.")
    contract = _contract(cfg, seeds, ks)
    if path.exists():
        saved = read_json(path)
        if saved["spec"] != contract or saved["signature"] != digest(contract):
            raise ValueError("Final protocol changed after freezing; use a separate experiment.")
        return {**saved, "reused": True}
    value = {"signature": digest(contract), "spec": contract}
    atomic_json(path, value)
    return {**value, "reused": False}


def verify_freeze(cfg, seed, k=None):
    path = Path(cfg["experiment_dir"]) / "final_protocol.json"
    if not path.exists():
        return None  # Exploratory single-seed runs remain supported and are labeled.
    frozen = read_json(path)
    spec = frozen["spec"]
    if seed not in spec["inference_seeds"] or (k is not None and k not in spec["k_values"]):
        raise ValueError("Seed/k was not declared in the final protocol.")
    if spec != _contract(cfg, spec["inference_seeds"], spec["k_values"]) or frozen["signature"] != digest(spec):
        raise ValueError("Config, data, source or adapter changed after the final freeze.")
    return frozen["signature"]


def aggregate_seeds(cfg, seeds=(42, 43, 44), k=None):
    """Score paired caches afresh; repeated seeds do not become independent questions."""
    from research.slm_sql.evaluation import score

    seeds = list(seeds)
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError("Provide unique inference seeds.")
    k = k or cfg["generation"]["evaluation_samples"]
    root = Path(cfg["experiment_dir"])
    if not (root / "final_protocol.json").exists():
        raise ValueError("Freeze before final inference to produce a multi-seed final report.")
    reports, sources = [], []
    for seed in seeds:
        verify_freeze(cfg, seed, k)
        result = score(cfg, seed=seed, k=k)
        path = Path(result["path"])
        reports.append(read_json(path))
        sources.append({"path": str(path), "sha256": file_hash(path)})
    ids = [r["id"] for r in reports[0]["records"]]
    if any([r["id"] for r in report["records"]] != ids for report in reports):
        raise ValueError("Every seed must evaluate the same questions in the same order.")
    summary, rows = {}, []
    for variant in ("sc", "author_csc", "lora_csc"):
        exs = [r["summary"][variant]["ex"] for r in reports]
        by_question = [sum(report["records"][i]["correct"][variant] for report in reports) / len(reports) for i in range(len(ids))]
        metrics = {
            "ex_mean": statistics.mean(exs),
            "ex_std": statistics.stdev(exs) if len(exs) > 1 else 0.0,
            "ex_by_seed": dict(zip(map(str, seeds), exs)),
            "seed_averaged_question_ci95": bootstrap_mean_ci(by_question, cfg["evaluation"]["bootstrap_samples"]),
        }
        summary[variant] = metrics
        rows.append({"variant": variant, "seeds": len(seeds), "questions": len(ids), "ex_mean": metrics["ex_mean"], "ex_std": metrics["ex_std"]})
    deltas = [sum(int(r["records"][i]["correct"]["lora_csc"]) - int(r["records"][i]["correct"]["author_csc"]) for r in reports) / len(reports) for i in range(len(ids))]
    report = {
        "seeds": seeds, "k": k, "questions": len(ids), "summary": summary,
        "paired_delta_ci95": bootstrap_mean_ci(deltas, cfg["evaluation"]["bootstrap_samples"]),
        "paired_tests_by_seed": {str(seed): r["comparison"] for seed, r in zip(seeds, reports)},
        "sources": sources, "final_protocol_signature": read_json(root / "final_protocol.json")["signature"],
        "notes": ["Inference seeds share one trained adapter; these are not three independent training runs.",
                  "Bootstrap unit is a question averaged over seeds; McNemar tests remain per seed."],
    }
    path = root / "reports" / f"evaluation-k{k}.multi-seed.json"
    atomic_json(path, report)
    with path.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {"path": str(path), "summary": summary, "paired_delta_ci95": report["paired_delta_ci95"]}
