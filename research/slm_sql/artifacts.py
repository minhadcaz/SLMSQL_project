"""Atomic, resumable JSONL artifacts with strict provenance checks."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def file_hash(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@contextmanager
def exclusive_lock(path: str | Path):
    """Reject concurrent writers immediately instead of repeating expensive work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                if not path.stat().st_size:
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Another process is writing this stage: {path}") from exc
        yield


def locked_stage(name: str):
    def decorate(function):
        @wraps(function)
        def run(cfg, *args, **kwargs):
            with exclusive_lock(Path(cfg["experiment_dir"]) / "locks" / f"{name}.lock"):
                return function(cfg, *args, **kwargs)

        return run

    return decorate


class Journal:
    """Hold a writer lock; commit completed questions without rewriting the cache.

    An interrupted, unterminated last line is discarded only for incomplete runs.
    A completed artifact must pass its content hash before it can be reused.
    """

    def __init__(self, path: str | Path, spec: dict, ids: list[str]):
        self.path = Path(path)
        self.manifest_path = self.path.with_suffix(self.path.suffix + ".manifest.json")
        self.spec = spec
        self.ids = ids
        if len(ids) != len(set(ids)):
            raise ValueError("Artifact IDs must be unique.")
        self.records: list[dict] = []
        self.manifest: dict = {}
        self._lock = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.path.with_suffix(self.path.suffix + ".lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt

                self._lock.write(b"0")
                self._lock.flush()
                self._lock.seek(0)
                msvcrt.locking(self._lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._load()
            return self
        except BaseException:
            self._lock.close()
            raise

    def _load(self):
        if self.manifest_path.exists():
            self.manifest = read_json(self.manifest_path)
            if self.manifest["signature"] != digest(self.spec):
                raise ValueError(
                    f"Artifact configuration changed: {self.path}. Use a new experiment directory."
                )
        elif self.path.exists() and self.path.stat().st_size:
            raise ValueError(f"Missing provenance manifest: {self.path}")
        else:
            self.manifest = {"signature": digest(self.spec), "spec": self.spec, "complete": False}
            atomic_json(self.manifest_path, self.manifest)
        if self.manifest.get("complete"):
            if not self.path.exists() or file_hash(self.path) != self.manifest["sha256"]:
                raise ValueError(f"Completed artifact was modified: {self.path}")
        if self.path.exists():
            with self.path.open("r+b") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.endswith(b"\n"):
                        if self.manifest.get("complete"):
                            raise ValueError(f"Corrupt completed JSONL: {self.path}")
                        handle.truncate(offset)
                        break
                    self.records.append(json.loads(line))
        if [r["id"] for r in self.records] != self.ids[: len(self.records)] or len(
            self.records
        ) > len(self.ids):
            raise ValueError(f"Artifact question order or dataset changed: {self.path}")
        if self.manifest.get("complete") and len(self.records) != len(self.ids):
            raise ValueError(f"Incomplete question set in completed artifact: {self.path}")

    def append(self, record: dict) -> None:
        if (
            self.complete
            or len(self.records) >= len(self.ids)
            or record["id"] != self.ids[len(self.records)]
        ):
            raise ValueError("Unexpected or duplicate artifact record.")
        with self.path.open("ab") as handle:
            handle.write((json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)

    @property
    def complete(self) -> bool:
        return bool(self.manifest.get("complete"))

    def finish(self, **metadata) -> dict:
        if len(self.records) != len(self.ids):
            raise ValueError("Cannot finish an artifact with missing questions.")
        if not self.path.exists():
            self.path.touch()
        self.manifest.update(metadata)
        self.manifest.update(complete=True, count=len(self.records), sha256=file_hash(self.path))
        atomic_json(self.manifest_path, self.manifest)
        return self.manifest

    def __exit__(self, *_):
        self._lock.close()


def completed_artifact(path: str | Path) -> tuple[list[dict], dict]:
    path = Path(path)
    manifest = read_json(path.with_suffix(path.suffix + ".manifest.json"))
    if not manifest.get("complete") or file_hash(path) != manifest.get("sha256"):
        raise ValueError(f"Artifact is incomplete or modified: {path}")
    return read_jsonl(path), manifest
