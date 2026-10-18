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
        for stage in (
            "generation",
            "execution",
            f"author.csc-k{cfg['generation'][f'{split}_samples']}",
            f"lora.csc-k{cfg['generation'][f'{split}_samples']}",
        ):
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
        "status": stages,
        "order": [
            "prepare",
            "baseline",
            "generate --split train",
            "execute --split train",
            "generate --split validation",
            "execute --split validation",
            "pairs",
            "train",
            "merge --variant lora",
            "score",
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
    for command in ("plan", "generate", "execute", "baseline", "merge", "pairs", "train", "score"):
        cmd = sub.add_parser(command)
        cmd.add_argument("--seed", type=int, default=42)
        if command in ("generate", "execute", "baseline", "merge", "score"):
            cmd.add_argument(
                "--split", choices=("train", "validation", "evaluation"), default="evaluation"
            )
        if command in ("baseline", "merge", "score"):
            cmd.add_argument(
                "--k",
                type=int,
                help="Use the first k cached generator samples, without regenerating",
            )
        if command == "merge":
            cmd.add_argument("--variant", choices=("author", "lora"), default="author")
        if command == "score":
            cmd.add_argument("--baseline-only", action="store_true")
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
        atomic_json(args.output, cfg)
        result = {"config": str(args.output), "pilot": args.pilot}
    elif args.command == "plan":
        result = plan(cfg, seed)
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
    elif args.command in ("generate", "execute", "merge"):
        from research.slm_sql.pipeline import execute_stage, generate_stage, merge_stage

        if args.command == "generate":
            result = generate_stage(cfg, args.split, seed)
        elif args.command == "execute":
            result = execute_stage(cfg, args.split, seed)
        else:
            result = merge_stage(cfg, args.split, seed, variant=args.variant, k=args.k)
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
    else:
        from research.slm_sql.evaluation import score

        result = score(cfg, args.split, seed, args.k, baseline_only=args.baseline_only)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
