"""Run `python -m research.slm_sql --help` for the canonical CSC research workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from research.slm_sql.artifacts import atomic_json, read_json
from research.slm_sql.config import DEFAULT_CONFIG, artifact_path, load_config


def plan(cfg: dict, seed=42) -> dict:
    root = Path(cfg["experiment_dir"])
    counts = {}
    manifest_path = root / "data" / "manifest.json"
    if manifest_path.exists():
        data = read_json(manifest_path)
        counts = {key: item["count"] for key, item in data["splits"].items()}
    stages = {}
    for split in ("train", "validation", "evaluation"):
        stages[split] = {}
        names = [
            "generation",
            "execution",
            f"author.csc-k{cfg['generation'][f'{split}_samples']}",
            f"lora.csc-k{cfg['generation'][f'{split}_samples']}",
        ]
        if split in ("validation", "evaluation"):
            names.extend(["adaptive-csc-eval"])
        if split == "validation" and cfg["adaptive"]["enabled"]:
            names.extend(
                f"{variant}.csc-k{k}"
                for k in cfg["adaptive"]["checkpoints"]
                if k <= cfg["generation"]["validation_samples"]
                for variant in ("author", "lora")
            )
        if split == "evaluation":
            names.extend(["greedy.generation", "greedy.execution"])
            names.extend(f"{variant}.csc-k{k}" for k in cfg["evaluation"]["sampling_k_values"]
                         for variant in ("author", "lora"))
            if cfg["adaptive"]["enabled"]:
                names.extend(["adaptive", "author.adaptive-csc", "lora.adaptive-csc"])
        for stage in dict.fromkeys(names):
            path = artifact_path(cfg, split, stage, seed)
            sidecar = path.with_suffix(path.suffix + ".manifest.json")
            state = read_json(sidecar) if sidecar.exists() else {}
            stages[split][stage] = (
                "complete (validated when used)" if state.get("complete") else "pending/resumable"
            )
    return {
        "experiment_dir": str(root),
        "generator": cfg["generation"]["model"],
        "merger": cfg["merger"]["model"],
        "seed": seed,
        "inputs": {
            key: {"path": cfg["data"][key], "exists": (Path(cfg["data"][key]) / name).is_file()}
            for key, name in (("train_dir", "train.json"), ("evaluation_dir", "dev.json"))
        },
        "prepared_questions": counts,
        "generation_completions": {
            split: counts.get(split, 0) * cfg["generation"][f"{split}_samples"] for split in counts
        },
        "maximum_merge_completions_evaluation": counts.get("evaluation", 0)
        * cfg["merger"]["samples"],
        "sampling_matrix": {
            "k_values": cfg["evaluation"]["sampling_k_values"],
            "greedy_completions": counts.get("evaluation", 0),
            "adaptive": cfg["adaptive"],
            "maximum_adaptive_completions": counts.get("evaluation", 0) * cfg["adaptive"]["max_k"]
            if cfg["adaptive"]["enabled"] else 0,
            "maximum_csc_sweep_completions_per_variant": counts.get("evaluation", 0) * cfg["merger"]["samples"]
            * (sum(k > 1 for k in cfg["evaluation"]["sampling_k_values"]) + int(cfg["adaptive"]["enabled"])),
            "maximum_author_csc_completions_including_training_baseline": counts.get("evaluation", 0)
            * cfg["merger"]["samples"]
            * (sum(k > 1 for k in set(cfg["evaluation"]["sampling_k_values"]
                                      + [cfg["generation"]["evaluation_samples"]]))
               + int(cfg["adaptive"]["enabled"])),
            "notes": "Greedy/adaptive use separate draws. Only pending stages load weights. CSC@1 has no merger call.",
        },
        "status": stages,
        "order": [
            "prepare",
            "sampling-baseline",
            "generate --split train",
            "execute --split train",
            "generate --split validation",
            "execute --split validation",
            "pairs",
            "train",
            "merge --split validation --variant author|lora --k <adaptive checkpoint>",
            "adaptive-sweep --split validation",
            "freeze --seeds ... --ks ...",
            "sampling-final",
            "report-suite --seeds ...",
        ],
        "downloads_or_gpu_started": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="Write a config without downloading data or weights")
    init.add_argument("--output", type=Path, required=True)
    init.add_argument(
        "--pilot",
        action="store_true",
        help="Small separate experiment: 80/30/20 questions, 4 generation / 2 CSC samples",
    )
    init.add_argument("--train-dir")
    init.add_argument("--evaluation-dir")
    prepare_parser = sub.add_parser(
        "prepare", help="Validate benchmark paths/gold SQL and freeze DB-disjoint splits (CPU)"
    )
    prepare_parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate author models using dev only; train splits can be added later without regenerating evaluation candidates",
    )
    for command in (
        "plan", "generate", "execute", "greedy", "adaptive", "baseline", "merge",
        "pairs", "train", "score", "suite", "sampling-baseline", "sampling-final",
        "adaptive-merge", "adaptive-csc-eval", "adaptive-sweep", "adaptive-policy-sweep",
    ):
        cmd = sub.add_parser(command)
        cmd.add_argument("--seed", type=int, default=42)
        if command in ("generate", "execute", "greedy", "adaptive", "baseline", "merge", "score", "suite",
                       "sampling-baseline", "sampling-final", "adaptive-merge", "adaptive-csc-eval",
                       "adaptive-sweep", "adaptive-policy-sweep"):
            cmd.add_argument(
                "--split", choices=("train", "validation", "evaluation"), default="evaluation"
            )
        if command in ("baseline", "merge", "score", "suite", "sampling-baseline", "sampling-final"):
            cmd.add_argument(
                "--k",
                type=int,
                help="Use the first k cached generator samples, without regenerating",
            )
        if command == "merge":
            cmd.add_argument("--variant", choices=("author", "lora"), default="author")
            cmd.add_argument("--source", choices=("fixed", "adaptive"), default="fixed")
        if command == "score":
            cmd.add_argument("--baseline-only", action="store_true")
        if command == "suite":
            mode = cmd.add_mutually_exclusive_group()
            mode.add_argument("--baseline-only", action="store_true")
            mode.add_argument("--generator-only", action="store_true", help="Score greedy, pass@k, SC@k and adaptive without merger outputs")
        if command == "adaptive-merge":
            cmd.add_argument("--variant", choices=("author", "lora"), default="author")
        if command == "adaptive-csc-eval":
            cmd.add_argument("--policy", choices=("beta", "margin", "router"))
        if command in ("adaptive-sweep", "adaptive-policy-sweep"):
            cmd.add_argument("--policy", choices=("beta", "margin", "router"))
            cmd.add_argument("--thresholds", help="Comma-separated thresholds for validation sweep")
    freeze_parser = sub.add_parser("freeze", help="Freeze adapter/config/source and final seeds before final LoRA inference (CPU)")
    freeze_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    freeze_parser.add_argument("--ks", nargs="+", type=int)
    report_parser = sub.add_parser("report", help="Aggregate paired inference seeds without pooling repeated questions (CPU)")
    report_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    report_parser.add_argument("--k", type=int)
    suite_report_parser = sub.add_parser("report-suite", help="Aggregate the full sampling matrix over paired inference seeds (CPU)")
    suite_report_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    suite_report_parser.add_argument("--k", type=int)
    suite_report_mode = suite_report_parser.add_mutually_exclusive_group()
    suite_report_mode.add_argument("--baseline-only", action="store_true")
    suite_report_mode.add_argument("--generator-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    seed = getattr(args, "seed", 42)
    if seed < 0:
        parser.error("seed must be nonnegative")
    if args.command == "init":
        if args.output.exists():
            parser.error(f"Config already exists: {args.output}")
        if args.train_dir:
            cfg["data"]["train_dir"] = str(Path(args.train_dir).resolve())
        if args.evaluation_dir:
            cfg["data"]["evaluation_dir"] = str(Path(args.evaluation_dir).resolve())
        if args.pilot:
            cfg["experiment_dir"] = str(Path("results/csc_slm_sql_pilot").resolve())
            cfg["data"].update(train_limit=80, validation_limit=30, evaluation_limit=20)
            for key in ("train_samples", "validation_samples", "evaluation_samples"):
                cfg["generation"][key] = 4
            cfg["merger"]["samples"] = 2
            cfg["training"]["epochs"] = 1
            cfg["evaluation"]["sampling_k_values"] = [1, 2, 4]
            cfg["adaptive"]["max_k"] = 4
            cfg["adaptive"]["min_k"] = 2
            cfg["adaptive"]["batch_size"] = 2
            cfg["adaptive"]["checkpoints"] = [2, 4]
        atomic_json(args.output, cfg)
        result = {"config": str(args.output), "pilot": args.pilot}
    elif args.command == "plan":
        result = plan(cfg, seed)
    elif args.command == "freeze":
        from research.slm_sql.protocol import freeze
        value = freeze(cfg, args.seeds, args.ks)
        result = {"signature": value["signature"], "reused": value["reused"]}
    elif args.command == "report":
        from research.slm_sql.protocol import aggregate_seeds
        result = aggregate_seeds(cfg, args.seeds, args.k)
    elif args.command == "report-suite":
        from research.slm_sql.sampling import aggregate_suites
        result = aggregate_suites(cfg, args.seeds, k=args.k,
                                  baseline_only=args.baseline_only, generator_only=args.generator_only)
    elif args.command == "prepare":
        from research.slm_sql.data import prepare

        prepared = prepare(cfg, evaluation_only=args.evaluation_only)
        result = {"splits": prepared["splits"], "reused": prepared["reused"]}
    elif args.command == "baseline":
        # Each GPU model lives in its own process and releases VRAM on exit.
        commands = [
            ("generate", []),
            ("execute", []),
            ("merge", ["--variant", "author"]),
            ("score", ["--baseline-only"]),
        ]
        for command, extra in commands:
            cmd = [
                sys.executable,
                "-m",
                "research.slm_sql",
                "--config",
                str(args.config.resolve()),
                command,
                "--split",
                args.split,
                "--seed",
                str(seed),
                *extra,
            ]
            if args.k is not None and command in ("merge", "score"):
                cmd += ["--k", str(args.k)]
            subprocess.run(cmd, check=True)
        result = {"baseline": "complete", "split": args.split, "seed": seed}
    elif args.command in ("sampling-baseline", "sampling-final"):
        from research.slm_sql.sampling import sampling_ks
        ks = sampling_ks(cfg, args.k, args.split)
        final = args.command == "sampling-final"
        if final and args.split == "evaluation":
            from research.slm_sql.protocol import verify_freeze
            if not (Path(cfg["experiment_dir"]) / "final_protocol.json").is_file():
                raise ValueError("Run freeze before sampling-final to lock seeds, k values and adaptive settings.")
            for k in ks:
                verify_freeze(cfg, seed, k)
        commands = []
        if not final:
            commands.extend([("generate", []), ("execute", []), ("greedy", [])])
            if cfg["adaptive"]["enabled"]:
                commands.append(("adaptive", []))
        # Training's canonical baseline requires full-n author CSC, even when
        # a user requests a smaller report sweep.
        merge_ks = sorted(set(ks + [cfg["generation"][f"{args.split}_samples"]]))
        for k in merge_ks:
            commands.append(("merge", ["--variant", "author", "--k", str(k)]))
            if final and k in ks:
                commands.append(("merge", ["--variant", "lora", "--k", str(k)]))
            if k in ks:
                commands.append(("score", ["--k", str(k)] + ([] if final else ["--baseline-only"])))
        if cfg["adaptive"]["enabled"]:
            adaptive_command = "adaptive-merge" if final else "merge"
            adaptive_source = [] if final else ["--source", "adaptive"]
            commands.append((adaptive_command, ["--variant", "author", *adaptive_source]))
            if final:
                commands.append((adaptive_command, ["--variant", "lora"]))
        commands.append(("suite", ([] if final else ["--baseline-only"])
                         + (["--k", str(args.k)] if args.k is not None else [])))
        for command, extra in commands:
            subprocess.run([
                sys.executable, "-m", "research.slm_sql", "--config", str(args.config.resolve()),
                command, "--split", args.split, "--seed", str(seed), *extra,
            ], check=True)
        result = {"sampling": "complete", "mode": "comparison" if final else "baseline",
                  "split": args.split, "seed": seed, "k_values": ks}
    elif args.command in ("adaptive-merge", "adaptive-csc-eval", "adaptive-sweep", "adaptive-policy-sweep"):
        if args.command == "adaptive-merge":
            from research.slm_sql.pipeline import merge_stage

            result = merge_stage(cfg, args.split, seed, variant=args.variant, source="adaptive")
        elif args.command == "adaptive-csc-eval":
            from research.slm_sql.adaptive_eval import evaluate_adaptive_csc

            parameters = {"policy": args.policy} if args.policy and args.split != "evaluation" else None
            result = evaluate_adaptive_csc(cfg, args.split, seed, policy_parameters=parameters)
        else:
            from research.slm_sql.adaptive_eval import adaptive_sweep

            values = None
            if args.thresholds:
                try:
                    values = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
                except ValueError as exc:
                    raise ValueError("--thresholds must be comma-separated numbers.") from exc
            result = adaptive_sweep(cfg, args.split, seed, policy_name=args.policy, thresholds=values)
    elif args.command in ("generate", "execute", "greedy", "adaptive", "merge"):
        from research.slm_sql.pipeline import (
            adaptive_stage,
            execute_stage,
            generate_stage,
            merge_stage,
        )

        if args.command == "generate":
            result = generate_stage(cfg, args.split, seed)
        elif args.command == "execute":
            result = execute_stage(cfg, args.split, seed)
        elif args.command == "greedy":
            generate_stage(cfg, args.split, seed, mode="greedy")
            result = execute_stage(cfg, args.split, seed, mode="greedy")
        elif args.command == "adaptive":
            result = adaptive_stage(cfg, args.split, seed)
        else:
            result = merge_stage(cfg, args.split, seed, variant=args.variant, k=args.k, source=args.source)
    elif args.command == "pairs":
        from research.slm_sql.adaptation import build_pairs

        pairs = build_pairs(cfg, seed)
        result = {"stats": pairs["stats"], "reused": pairs["reused"]}
    elif args.command == "train":
        from research.slm_sql.adaptation import train_adapter

        trained = train_adapter(cfg, seed)
        result = {
            "status": trained["status"],
            "adapter": str(Path(cfg["experiment_dir"]) / "adapter" / "final"),
            "reused": trained["reused"],
        }
    elif args.command == "suite":
        from research.slm_sql.sampling import evaluate_suite

        result = evaluate_suite(cfg, args.split, seed, k=args.k,
                                baseline_only=args.baseline_only, generator_only=args.generator_only)
    else:
        from research.slm_sql.evaluation import score

        result = score(cfg, args.split, seed, args.k, baseline_only=args.baseline_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
