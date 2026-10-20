"""Prospective freeze for the optional legacy 00-08 evaluation branch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research.reproducibility import source_state
from research.slm_sql.artifacts import atomic_json, digest, read_json


def _records_contract(records: list[dict[str, Any]]) -> dict[str, Any]:
    items = [
        (record.get("id", f"{record.get('db_id')}::{record.get('question')}"),
         record.get("db_id"), record.get("question"), record.get("gold_sql"))
        for record in records
    ]
    if len({item[0] for item in items}) != len(items):
        raise ValueError("Legacy evaluation records must have unique IDs.")
    return {
        "questions": len(items),
        "db_ids": sorted({item[1] for item in items}),
        "records_sha256": digest(sorted(items)),
    }


def freeze_legacy_protocol(
    path: str | Path,
    *,
    configs: dict[str, dict[str, Any]],
    seeds: list[int],
    k_values: list[int],
    datasets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Write or verify the final legacy evaluation contract before inference."""
    seeds = list(seeds)
    k_values = list(k_values)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Legacy evaluation seeds must be unique.")
    if not k_values or len(k_values) != len(set(k_values)):
        raise ValueError("Legacy evaluation k values must be unique.")
    contract = {
        "version": 1,
        "configs": configs,
        "seeds": seeds,
        "k_values": k_values,
        "datasets": {name: _records_contract(records) for name, records in sorted(datasets.items())},
        "source": {key: value for key, value in source_state().items() if key != "git_dirty"},
    }
    path = Path(path)
    if path.exists():
        saved = read_json(path)
        if saved.get("spec") != contract or saved.get("signature") != digest(contract):
            raise ValueError("Legacy final protocol changed; use a new results/final directory.")
        return {**saved, "reused": True}
    payload = {"signature": digest(contract), "spec": contract}
    atomic_json(path, payload)
    return {**payload, "reused": False}
