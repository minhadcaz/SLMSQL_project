"""One DB-disjoint calibration holdout shared by legacy notebooks 02–07.

This branch reports a held-out portion of BIRD dev, not full-dev paper scores.
The canonical 09–11 branch instead calibrates on BIRD train and retains full dev.
"""

from __future__ import annotations

import random
from pathlib import Path

from research.slm_sql.artifacts import atomic_json, digest, read_json


def split_legacy_dev(records: list[dict], manifest_path: str | Path, *, seed=42, fraction=0.2):
    if not 0 < fraction < 1:
        raise ValueError("Calibration fraction must be in (0, 1).")
    db_ids = sorted({r["db_id"] for r in records})
    if len(db_ids) < 2:
        raise ValueError("At least two databases are needed for a disjoint calibration holdout.")
    identities = [digest([r["db_id"], r["question"], r["gold_sql"]]) for r in records]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate benchmark questions in legacy split.")
    random.Random(seed).shuffle(db_ids)
    count = max(1, min(len(db_ids) - 1, round(len(db_ids) * fraction)))
    calibration_dbs = set(db_ids[:count])
    spec = {
        "version": 1, "seed": seed, "calibration_fraction": fraction,
        "source_sha256": digest(identities),
        "calibration_db_ids": sorted(calibration_dbs),
        "evaluation_db_ids": sorted(set(db_ids) - calibration_dbs),
    }
    path = Path(manifest_path)
    if path.exists():
        if read_json(path) != spec:
            raise ValueError("Legacy data split changed; use a separate results directory.")
    else:
        atomic_json(path, spec)
    return {
        "calibration": [r for r in records if r["db_id"] in calibration_dbs],
        "evaluation": [r for r in records if r["db_id"] not in calibration_dbs],
        "manifest": spec,
    }


def partition_cached_records(records: list[dict], full_split: dict) -> dict:
    """Apply an existing full-source split to a possibly smaller pilot cache."""
    allowed = {(r["db_id"], r["question"], r["gold_sql"]) for part in ("calibration", "evaluation")
               for r in full_split[part]}
    if any((r["db_id"], r["question"], r["gold_sql"]) not in allowed for r in records):
        raise ValueError("Candidate cache contains questions outside the declared benchmark source.")
    dbs = set(full_split["manifest"]["calibration_db_ids"])
    parts = {
        "calibration": [r for r in records if r["db_id"] in dbs],
        "evaluation": [r for r in records if r["db_id"] not in dbs],
        "manifest": full_split["manifest"],
    }
    if not parts["calibration"] or not parts["evaluation"]:
        raise ValueError("Pilot cache needs both calibration and evaluation DBs; regenerate a stratified pilot.")
    return parts


def sample_legacy_pilot(full_split: dict, limit: int = 0, seed: int = 42) -> list[dict]:
    from training.data_prep import stratified_sample

    if limit < 0:
        raise ValueError("BIRD_LIMIT must be nonnegative.")
    if limit == 0:
        return full_split["calibration"] + full_split["evaluation"]
    if limit < 2:
        raise ValueError("Use BIRD_LIMIT >= 2 so the pilot covers both disjoint partitions.")
    n_calibration = min(len(full_split["calibration"]), max(1, round(limit * 0.2)))
    n_evaluation = min(len(full_split["evaluation"]), limit - n_calibration)
    n_calibration = min(len(full_split["calibration"]), limit - n_evaluation)
    return (stratified_sample(full_split["calibration"], n_calibration, seed)
            + stratified_sample(full_split["evaluation"], n_evaluation, seed))
