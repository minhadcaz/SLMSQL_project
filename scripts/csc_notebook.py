"""Small helpers shared by the three canonical CSC notebooks; no GPU imports."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from research.slm_sql.artifacts import atomic_json, file_hash, read_json
from research.slm_sql.config import DEFAULT_CONFIG, load_config


def configure(repo: str | Path) -> tuple[Path, dict]:
    repo = Path(repo)
    path = Path(os.environ.get("CSC_CONFIG", str(repo / "results" / "csc_run_config.json")))
    if path.exists():
        cfg = load_config(path)
        for env, actual in (
            ("CSC_TRAIN_DIR", cfg["data"]["train_dir"]),
            ("CSC_EVAL_DIR", cfg["data"]["evaluation_dir"]),
            ("CSC_EXPERIMENT_DIR", cfg["experiment_dir"]),
        ):
            if os.environ.get(env) and str(Path(os.environ[env]).resolve()) != actual:
                raise ValueError(
                    f"{env} differs from the frozen config. Use a new CSC_CONFIG and experiment directory."
                )
        return path, cfg
    cfg = load_config(DEFAULT_CONFIG)
    for key, env, filename, candidates in (
        (
            "train_dir",
            "CSC_TRAIN_DIR",
            "train.json",
            [
                repo / "data/bird/train",
                repo / "data/bird",
                Path("/kaggle/input/bird-train/train"),
                Path("/kaggle/input/bird/train"),
            ],
        ),
        (
            "evaluation_dir",
            "CSC_EVAL_DIR",
            "dev.json",
            [repo / "data/bird", Path("/kaggle/input/bird-dev"), Path("/kaggle/input/bird/dev")],
        ),
    ):
        configured = os.environ.get(env)
        if configured:
            cfg["data"][key] = str(Path(configured).resolve())
        else:
            found = next((p for p in candidates if (p / filename).is_file()), None)
            if found:
                cfg["data"][key] = str(found.resolve())
    if os.environ.get("CSC_PILOT", "1") == "1":
        cfg["data"].update(train_limit=80, validation_limit=30, evaluation_limit=20)
        for key in ("train_samples", "validation_samples", "evaluation_samples"):
            cfg["generation"][key] = 4
        cfg["merger"]["samples"] = 2
        cfg["training"]["epochs"] = 1
        cfg["experiment_dir"] = str((repo / "results/csc_slm_sql_pilot").resolve())
    if os.environ.get("CSC_EXPERIMENT_DIR"):
        cfg["experiment_dir"] = str(Path(os.environ["CSC_EXPERIMENT_DIR"]).resolve())
    atomic_json(path, cfg)
    return path, cfg


def run(repo: str | Path, config: str | Path, *arguments: str) -> None:
    command = [
        sys.executable,
        "-m",
        "research.slm_sql",
        "--config",
        str(Path(config).resolve()),
        *arguments,
    ]
    print("Stage:", " ".join(arguments), flush=True)
    subprocess.run(command, cwd=repo, check=True)


def install_profile(repo: str | Path, profile: str) -> None:
    if profile not in ("inference", "training"):
        raise ValueError("Unknown CSC dependency profile.")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            str(Path(repo) / f"requirements-csc-{profile}.txt"),
        ],
        check=True,
    )


def hub_credentials() -> tuple[str, str]:
    """Only read Kaggle secrets for explicitly requested Hub actions."""
    repo_id, token = (
        os.environ.get("HF_REPO_ID", "").strip(),
        os.environ.get("HF_TOKEN", "").strip(),
    )
    if not repo_id or not token:
        try:
            from kaggle_secrets import UserSecretsClient

            secrets = UserSecretsClient()
            repo_id = repo_id or secrets.get_secret("HF_REPO_ID")
            token = token or secrets.get_secret("HF_TOKEN")
        except (ImportError, KeyError, RuntimeError) as exc:
            raise ValueError(
                "Configure Kaggle Secrets HF_REPO_ID and HF_TOKEN (write access)."
            ) from exc
    if not repo_id or not token:
        raise ValueError("Both HF_REPO_ID and HF_TOKEN must be configured.")
    return repo_id, token


def publish_csc_adapter(cfg: dict) -> dict:
    """Publish only the verified LoRA adapter + provenance to a private model repo."""
    from huggingface_hub import HfApi

    from research.slm_sql.adaptation import _check_weights

    repo_id, token = hub_credentials()
    root = Path(cfg["experiment_dir"]) / "adapter"
    manifest = read_json(root / "training_manifest.json")
    if manifest.get("status") != "complete" or manifest["spec"]["training"] != cfg["training"]:
        raise ValueError("Adapter is incomplete or training settings changed.")
    _check_weights(root / "final", manifest["adapter_files"])
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id=repo_id).private:
        raise ValueError("Refusing to upload to an existing public model repository.")
    api.upload_folder(
        folder_path=str(root / "final"),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload CSC LoRA adapter",
    )
    api.upload_file(
        path_or_fileobj=str(root / "training_manifest.json"),
        path_in_repo=".csc/training_manifest.json",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Record CSC base checkpoint and adapter hashes",
    )
    return {"repo_id": repo_id, "adapter_dir": str(root / "final"), "private": True}


def restore_csc_adapter(cfg: dict) -> dict:
    """Restore matching adapter files; the full experiment cache is separate."""
    from huggingface_hub import HfApi, snapshot_download

    from research.slm_sql.adaptation import _check_weights

    repo_id, token = hub_credentials()
    if not HfApi(token=token).model_info(repo_id=repo_id).private:
        raise ValueError("CSC adapter restoration requires a private model repository.")
    downloaded = Path(snapshot_download(repo_id=repo_id, repo_type="model", token=token))
    manifest = read_json(downloaded / ".csc" / "training_manifest.json")
    if manifest.get("status") != "complete" or manifest["spec"]["training"] != cfg["training"]:
        raise ValueError("Hub adapter is incomplete or its training settings differ.")
    _check_weights(downloaded, manifest["adapter_files"])
    root = Path(cfg["experiment_dir"])
    pairs = read_json(root / "adaptation" / "manifest.json")
    if manifest["spec"]["pairs"] != pairs["signature"]:
        raise ValueError("Hub adapter does not match the restored training pairs.")
    target = root / "adapter"
    files = {
        **{name: downloaded / name for name in manifest["adapter_files"]},
        "training_manifest.json": downloaded / ".csc" / "training_manifest.json",
    }
    for name, source in files.items():
        existing = target / ("final" if name != "training_manifest.json" else "") / name
        if existing.exists() and file_hash(existing) != file_hash(source):
            raise ValueError(f"Existing adapter file differs; refusing overwrite: {existing}")
    for name, source in files.items():
        destination = target / ("final" if name != "training_manifest.json" else "") / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source, destination)
    _check_weights(target / "final", manifest["adapter_files"])
    return {"repo_id": repo_id, "adapter_dir": str(target / "final"), "restored": True}
