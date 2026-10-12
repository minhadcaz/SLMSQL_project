"""Small, dependency-free helpers shared by the Kaggle research notebooks.

The notebooks are intentionally runnable in separate Kaggle sessions.  This
module only resolves paths, prepares a validated BIRD development split, and
publishes an already-created checkpoint when the user explicitly configures a
Hugging Face token and repository.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_BIRD_DEV_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def _is_repo(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "benchmarks").is_dir()


def resolve_repo_dir(start: str | Path | None = None) -> Path:
    """Find the checked-out project without assuming a particular Kaggle path."""
    configured = os.environ.get("SLM_PRJ_PATH")
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    current = Path(start or Path.cwd()).resolve()
    candidates.extend([current, *current.parents])
    candidates.extend([
        Path("/kaggle/working/SLM-PRJ"),
        Path("/kaggle/working/SLMSQL_project"),
        Path("/content/SLM-PRJ"),
        Path("/content/SLMSQL_project"),
    ])
    for candidate in candidates:
        if _is_repo(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find the repository. Clone it into /kaggle/working or set SLM_PRJ_PATH."
    )


def prepare_repo_import(repo_dir: str | Path | None = None) -> Path:
    """Change into the repo and put it first on sys.path for notebook imports."""
    resolved = resolve_repo_dir(repo_dir)
    os.chdir(resolved)
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    return resolved


def ensure_repo(repo_url: str | None = None, destination: str | Path | None = None) -> Path:
    """Find the repo or clone it into the writable Kaggle workspace."""
    try:
        return prepare_repo_import()
    except FileNotFoundError:
        url = repo_url or os.environ.get(
            "SLM_PRJ_REPO_URL", "https://github.com/minhadcaz/SLMSQL_project.git"
        )
        target = Path(destination or "/kaggle/working/SLM-PRJ").expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and any(target.iterdir()):
            raise FileExistsError(
                f"Repository target is not empty and is not this project: {target}"
            )
        import subprocess

        subprocess.run(["git", "clone", url, str(target)], check=True)
        return prepare_repo_import(target)


def load_kaggle_secrets(names: tuple[str, ...] = ("HF_TOKEN", "HF_REPO_ID")) -> dict[str, bool]:
    """Load named Kaggle secrets into env without printing their values."""
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return {name: bool(os.environ.get(name, "").strip()) for name in names}
    client = UserSecretsClient()
    loaded = {}
    for name in names:
        if os.environ.get(name, "").strip():
            loaded[name] = True
            continue
        try:
            value = client.get_secret(name)
        except Exception:
            value = ""
        if value:
            os.environ[name] = value
        loaded[name] = bool(value)
    return loaded


def resolve_data_dir(repo_dir: str | Path | None = None) -> Path:
    """Return the writable data directory used by every notebook."""
    configured = os.environ.get("SLM_DATA_DIR")
    data_dir = Path(configured).expanduser() if configured else Path(repo_dir or resolve_repo_dir()) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir.resolve()


def _bird_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"BIRD questions file is empty or not a JSON list: {path}")
    return records


def validate_bird_dir(
    data_dir: str | Path, *, minimum_questions: int = 1000,
) -> tuple[bool, str]:
    """Validate the full BIRD dev layout, not just that directories exist.

    The minimum guards against accidentally reusing a Mini-Dev or a tiny pilot
    subset as the official full development split.
    """
    root = Path(data_dir)
    questions = root / "dev.json"
    databases = root / "dev_databases"
    if not questions.is_file():
        return False, f"missing {questions}"
    if not databases.is_dir():
        return False, f"missing {databases}"
    try:
        records = _bird_records(questions)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if len(records) < minimum_questions:
        return False, (
            f"only {len(records)} questions found; full BIRD dev should not be a Mini-Dev "
            f"or pilot subset (minimum guard: {minimum_questions})"
        )
    missing = []
    for record in records:
        db_id = record.get("db_id")
        if not db_id:
            return False, "a BIRD record has no db_id"
        db_path = databases / str(db_id) / f"{db_id}.sqlite"
        if not db_path.is_file():
            missing.append(str(db_path))
            if len(missing) >= 3:
                break
    if missing:
        return False, "missing database files: " + ", ".join(missing)
    return True, f"{len(records)} questions and {len({r['db_id'] for r in records})} databases"


def _safe_extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with zipfile.ZipFile(archive) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.filename}")
        handle.extractall(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    base = destination.resolve()
    with tarfile.open(archive) as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != base and base not in target.parents:
                raise ValueError(f"Unsafe path in archive: {member.name}")
        handle.extractall(destination)


def _extract_archive(archive: Path, destination: Path) -> None:
    if zipfile.is_zipfile(archive):
        _safe_extract_zip(archive, destination)
    elif tarfile.is_tarfile(archive):
        _safe_extract_tar(archive, destination)
    else:
        raise ValueError(f"Unsupported archive: {archive}")


def _find_first(root: Path, name: str) -> Path | None:
    matches = sorted(root.rglob(name))
    return matches[0] if matches else None


def _copy_bird_layout(source: Path, destination: Path) -> None:
    """Copy dev.json and dev_databases from an attached/downloaded dataset."""
    source = source.resolve()
    question_file = source / "dev.json"
    if not question_file.is_file():
        question_file = _find_first(source, "dev.json")
    if question_file is None:
        raise FileNotFoundError(f"dev.json not found under {source}")

    database_root = source / "dev_databases"
    if not database_root.is_dir():
        database_root = next((p for p in source.rglob("dev_databases") if p.is_dir()), None)
    if database_root is None:
        archive = _find_first(source, "dev_databases.zip")
        if archive is not None:
            extracted = source / "_dev_databases_extracted"
            _extract_archive(archive, extracted)
            database_root = next((p for p in extracted.rglob("dev_databases") if p.is_dir()), extracted)
    if database_root is None or not database_root.is_dir():
        raise FileNotFoundError(f"dev_databases/ not found under {source}")

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(question_file, destination / "dev.json")
    target_databases = destination / "dev_databases"
    if target_databases.exists():
        shutil.rmtree(target_databases)
    shutil.copytree(database_root, target_databases)


def _attached_bird_dir() -> Path | None:
    configured = os.environ.get("BIRD_INPUT_DIR")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([
        Path("/kaggle/input/bird-benchmark"),
        Path("/kaggle/input/bird-dev"),
        Path("/kaggle/input/bird-full-dev"),
        Path("/kaggle/input/bird-sql"),
    ])
    for candidate in candidates:
        if candidate.is_dir():
            ok, _ = validate_bird_dir(candidate)
            if ok:
                return candidate
    return None


def prepare_bird_dev(
    data_dir: str | Path,
    *,
    force_refresh: bool = False,
    url: str | None = None,
) -> dict[str, Any]:
    """Prepare the full BIRD dev split in ``data_dir/bird``.

    An attached Kaggle dataset is preferred.  If none is attached, the
    official development archive is downloaded.  Existing data is reused only
    after validating every database referenced by dev.json; set
    ``FORCE_REFRESH_DATA=1`` to deliberately replace it.
    """
    data_root = Path(data_dir)
    target = data_root / "bird"
    if not force_refresh:
        ok, detail = validate_bird_dir(target)
        if ok:
            return {"source": "existing_validated", "path": str(target), "detail": detail}

    attached = _attached_bird_dir()
    if attached is not None:
        _copy_bird_layout(attached, target)
        source = f"attached:{attached}"
    else:
        download_dir = data_root / "downloads"
        archive = download_dir / "bird_dev.zip"
        unpacked = download_dir / "bird_dev"
        if force_refresh and archive.exists():
            archive.unlink()
        if force_refresh and unpacked.exists():
            shutil.rmtree(unpacked)
        download_dir.mkdir(parents=True, exist_ok=True)
        download_url = url or os.environ.get("BIRD_DEV_URL", DEFAULT_BIRD_DEV_URL)
        if not archive.exists():
            print(f"Downloading full BIRD dev archive from {download_url}")
            urllib.request.urlretrieve(download_url, archive)
        if not unpacked.exists():
            _extract_archive(archive, unpacked)
        _copy_bird_layout(unpacked, target)
        source = download_url

    ok, detail = validate_bird_dir(target)
    if not ok:
        raise RuntimeError(f"BIRD dev preparation failed: {detail}")
    return {"source": source, "path": str(target), "detail": detail}


def resolve_synsql_raw(repo_dir: str | Path | None = None) -> Path | None:
    """Find an explicitly supplied synthetic training file without downloading it."""
    configured = os.environ.get("SYNSQL_RAW")
    candidates = [Path(configured).expanduser()] if configured else []
    root = Path(repo_dir or resolve_repo_dir())
    candidates.extend([
        root / "data" / "synsql" / "raw.jsonl",
        root / "data" / "synsql" / "syn_think.jsonl",
        Path("/kaggle/input/synsql/raw.jsonl"),
        Path("/kaggle/input/synsql-think/raw.jsonl"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def publish_checkpoint(
    checkpoint_dir: str | Path,
    *,
    repo_id: str | None = None,
    token_env: str = "HF_TOKEN",
    commit_message: str = "Upload Kaggle checkpoint",
) -> dict[str, Any]:
    """Upload a checkpoint only when both repo ID and token are configured."""
    checkpoint = Path(checkpoint_dir)
    repo_id = repo_id or os.environ.get("HF_REPO_ID", "").strip()
    token = os.environ.get(token_env, "").strip()
    if not repo_id:
        return {"uploaded": False, "reason": "HF_REPO_ID is not set"}
    if not token:
        return {"uploaded": False, "reason": f"{token_env} is not set"}
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(
        folder_path=str(checkpoint),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
    )
    return {"uploaded": True, "repo_id": repo_id, "checkpoint": str(checkpoint)}


def download_checkpoint(
    repo_id: str | None = None,
    *,
    local_dir: str | Path | None = None,
    token_env: str = "HF_TOKEN",
) -> Path:
    """Restore a model checkpoint into writable storage for a later session."""
    repo_id = repo_id or os.environ.get("HF_REPO_ID", "").strip()
    if not repo_id:
        raise ValueError("Set HF_REPO_ID to restore a checkpoint from Hugging Face.")
    from huggingface_hub import snapshot_download

    target = Path(local_dir or Path("/kaggle/working") / "slm-sql-checkpoint")
    target.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(target),
        token=os.environ.get(token_env) or None,
    )
    return Path(result)
