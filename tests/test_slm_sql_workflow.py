"""CPU integration tests: real SQLite execution with deterministic fake GPU backends."""

from __future__ import annotations

import copy
import json
import shutil
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

from research.slm_sql.adaptation import build_pairs, focus_tags, tokenize_example, validate_adapter
from research.slm_sql.artifacts import (
    Journal,
    atomic_json,
    exclusive_lock,
    completed_artifact,
    digest,
    file_hash,
    read_json,
)
from research.slm_sql.config import artifact_path, load_config
from research.slm_sql.data import load_split, prepare
from research.slm_sql.evaluation import score
from research.slm_sql.execution import execute, result_fingerprint
from research.slm_sql.pipeline import execute_stage, generate_stage, merge_stage

CORRECT = "SELECT SUM(amount) FROM sales"
WRONG = "SELECT COUNT(*) FROM sales"


@pytest.fixture
def experiment(tmp_path):
    train_root, eval_root = tmp_path / "train", tmp_path / "dev"
    train_root.mkdir()
    eval_root.mkdir()
    for root, split, count in ((train_root, "train", 6), (eval_root, "dev", 2)):
        records = []
        for index in range(count):
            db_id = f"{split}_{index}"
            folder = root / f"{split}_databases" / db_id
            folder.mkdir(parents=True)
            with sqlite3.connect(folder / f"{db_id}.sqlite") as conn:
                conn.execute("CREATE TABLE sales (id INTEGER PRIMARY KEY, amount REAL)")
                conn.executemany(
                    "INSERT INTO sales VALUES (?, ?)",
                    [(1, 10 + index + (100 if split == "dev" else 0)), (2, 20 + index)],
                )
            for kind in ("disputed", "unanimous", "invalid"):
                records.append(
                    {
                        "db_id": db_id,
                        "question": f"{kind}: sum revenue in {db_id}",
                        "evidence": "revenue = SUM(amount)",
                        "SQL": CORRECT + " /* GOLD_LABEL_SECRET */",
                        "difficulty": "simple",
                    }
                )
        (root / f"{split}.json").write_text(json.dumps(records))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "experiment_dir": str(tmp_path / "experiment"),
                "data": {
                    "train_dir": str(train_root),
                    "evaluation_dir": str(eval_root),
                    "validation_fraction": 0.33,
                    "train_limit": 0,
                    "validation_limit": 0,
                },
                "generation": {
                    "train_samples": 4,
                    "validation_samples": 4,
                    "evaluation_samples": 4,
                    "batch_size": 2,
                },
                "merger": {"samples": 2, "batch_size": 2},
                "evaluation": {"bootstrap_samples": 40},
            }
        )
    )
    return load_config(config)


class FakeBackend:
    calls = []
    instances = []

    def __init__(self, settings, *, frozen_identity=None, adapter=None):
        self.settings = settings
        self.adapter = adapter
        self.identity = frozen_identity or {"model": settings["model"], "revision": "fake-commit"}
        self.versions = {"backend": "fake"}
        self.instances.append(settings["model"])

    def generate(self, messages, *, n, seeds):
        self.calls.extend(messages)
        assert len(messages) == len(seeds)
        assert all("GOLD_LABEL_SECRET" not in str(m) for m in messages)
        generated = []
        for message in messages:
            text = str(message)
            if "SLM-SQL-1.5B" in self.settings["model"]:
                sqls = (
                    [CORRECT] * n
                    if "unanimous:" in text
                    else ["not SQL"] * n
                    if "invalid:" in text
                    else [WRONG, CORRECT] * n
                )
            else:
                sqls = [CORRECT if self.adapter else WRONG] * n
            generated.append(
                [
                    {
                        "raw_text": f"<answer>{sql}</answer>",
                        "n_tokens": 12,
                        "finish_reason": "stop",
                        "prompt_tokens": 100,
                    }
                    for sql in sqls[:n]
                ]
            )
        return generated


def _generate_execute(cfg, split):
    generate_stage(cfg, split, backend_factory=FakeBackend)
    execute_stage(cfg, split)


def _baseline(cfg, split="evaluation"):
    _generate_execute(cfg, split)
    merge_stage(cfg, split, backend_factory=FakeBackend)


def _fake_adapter(cfg):
    """Produce tiny fixtures representing a completed PEFT save; no model is trained."""
    pairs = read_json(Path(cfg["experiment_dir"]) / "adaptation" / "manifest.json")
    _, baseline = completed_artifact(artifact_path(cfg, "evaluation", "author.csc-k4", 42))
    folder = Path(cfg["experiment_dir"]) / "adapter" / "final"
    folder.mkdir(parents=True)
    (folder / "adapter_model.safetensors").write_bytes(b"TEST ONLY - not model weights")
    atomic_json(
        folder / "adapter_config.json", {"r": 16, "base_model_name_or_path": cfg["merger"]["model"]}
    )
    spec = {
        "training": cfg["training"],
        "base": baseline["model_identity"],
        "pairs": pairs["signature"],
        "files": pairs["files"],
        "target_mode": "completion_only_answer",
        "version": 1,
    }
    metadata = {
        "status": "complete",
        "signature": digest(spec),
        "spec": spec,
        "base_identity": baseline["model_identity"],
        "adapter_files": {p.name: file_hash(p) for p in folder.iterdir()},
    }
    atomic_json(folder.parent / "training_manifest.json", metadata)
    return folder


def test_prepare_database_disjoint_and_reused(experiment):
    first = prepare(experiment)
    assert not first["reused"]
    sets = [
        set(first["splits"][split]["db_ids"]) for split in ("train", "validation", "evaluation")
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert prepare(experiment)["reused"]
    schema = Path(experiment["experiment_dir"]) / "data" / "schemas" / f"{next(iter(sets[0]))}.json"
    schema.write_text('{"text": "modified"}')
    with pytest.raises(ValueError, match="schema was modified"):
        load_split(experiment, "train")


def test_prepare_rejects_overlap_and_database_change(experiment):
    prepare(experiment)
    records, _ = load_split(experiment, "evaluation")
    with sqlite3.connect(records[0]["db_path"]) as conn:
        conn.execute("INSERT INTO sales VALUES (3, 50)")
    with pytest.raises(ValueError, match="Database changed"):
        load_split(experiment, "evaluation")


def test_prepare_rejects_shared_database_ids(experiment):
    root = Path(experiment["data"]["evaluation_dir"])
    records = json.loads((root / "dev.json").read_text())
    source = root / "dev_databases" / "dev_0" / "dev_0.sqlite"
    target = root / "dev_databases" / "train_0"
    target.mkdir()
    (target / "train_0.sqlite").write_bytes(source.read_bytes())
    records[0]["db_id"] = "train_0"
    (root / "dev.json").write_text(json.dumps(records))
    with pytest.raises(ValueError, match="share database IDs"):
        prepare(experiment)


def test_bird_set_semantics_and_exact_numbers(experiment):
    prepare(experiment)
    rows, _ = load_split(experiment, "evaluation")
    path = rows[0]["db_path"]
    assert (
        execute("SELECT id FROM sales ORDER BY id", path)["fingerprint"]
        == execute("SELECT id FROM sales UNION ALL SELECT id FROM sales ORDER BY id DESC", path)[
            "fingerprint"
        ]
    )
    assert result_fingerprint([(1,), (1.0,)]) == result_fingerprint([(1,)])
    number = 1e23
    assert result_fingerprint([(number,)]) == result_fingerprint([(int(number),)])
    assert result_fingerprint([(number,)]) != result_fingerprint([(10**23,)])
    assert result_fingerprint([(None,)]) != result_fingerprint([("None",)])
    assert (
        execute("SELECT id FROM sales WHERE 0", path)["fingerprint"]
        == execute("SELECT id, amount FROM sales WHERE 0", path)["fingerprint"]
    )
    assert execute("DELETE FROM sales", path)["fingerprint"] is None
    assert execute("SELECT * FROM sales", path, max_rows=1)["error"]


def test_generation_and_merge_resume_without_loading_model(experiment):
    prepare(experiment)
    _baseline(experiment)

    def forbidden(*args, **kwargs):
        raise AssertionError("A complete cache must not load a GPU model")

    assert generate_stage(experiment, "evaluation", backend_factory=forbidden)["reused"]
    assert execute_stage(experiment, "evaluation")["reused"]
    assert merge_stage(experiment, backend_factory=forbidden)["reused"]
    cli_config = Path(experiment["experiment_dir"]).parent / "cli_config.json"
    atomic_json(cli_config, experiment)
    result = subprocess.run(
        [sys.executable, "-m", "research.slm_sql", "--config", str(cli_config), "baseline"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert '"baseline": "complete"' in result.stdout
    changed = copy.deepcopy(experiment)
    changed["generation"]["temperature"] = 0.9
    with pytest.raises(ValueError, match="configuration changed"):
        generate_stage(changed, "evaluation", backend_factory=forbidden)


def test_resume_after_interrupted_request(experiment):
    prepare(experiment)

    class Interrupting(FakeBackend):
        count = 0

        def generate(self, *args, **kwargs):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("simulated interruption")
            return super().generate(*args, **kwargs)

    with pytest.raises(RuntimeError, match="interruption"):
        generate_stage(experiment, "evaluation", backend_factory=Interrupting)
    path = artifact_path(experiment, "evaluation", "generation", 42)
    with path.open("ab") as handle:
        handle.write(b'{"id": "incomplete')

    class UpgradedBackend(FakeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.versions = {"backend": "upgraded"}

    with pytest.raises(ValueError, match="Inference package versions changed"):
        generate_stage(experiment, "evaluation", backend_factory=UpgradedBackend)
    generate_stage(experiment, "evaluation", backend_factory=FakeBackend)
    records, _ = completed_artifact(path)
    prepared, _ = load_split(experiment, "evaluation")
    assert [r["id"] for r in records] == [r["id"] for r in prepared]


def test_pairs_keep_gold_out_of_inputs_and_augment_orders(experiment):
    prepare(experiment)
    for split in ("train", "validation"):
        _generate_execute(experiment, split)
    result = build_pairs(experiment)
    assert result["stats"]["train"]["examples"] == 2 * result["stats"]["train"]["focused"]
    from research.slm_sql.artifacts import read_jsonl

    train = read_jsonl(Path(experiment["experiment_dir"]) / "adaptation" / "train.jsonl")
    validation = read_jsonl(Path(experiment["experiment_dir"]) / "adaptation" / "validation.jsonl")
    assert not {r["db_id"] for r in train} & {r["db_id"] for r in validation}
    assert all(r["source_split"] == "train" for r in train)
    assert all("GOLD_LABEL_SECRET" not in str(r["messages"]) for r in train)
    assert all(r["target"] == f"<answer>{CORRECT}</answer>" for r in train)
    assert build_pairs(experiment)["reused"]


def test_paired_author_vs_lora_end_to_end(experiment):
    prepare(experiment)
    for split in ("train", "validation", "evaluation"):
        _generate_execute(experiment, split)
    merge_stage(experiment, backend_factory=FakeBackend)
    baseline = score(experiment, baseline_only=True)
    assert baseline["summary"]["author_csc"]["ex"] == pytest.approx(1 / 3)
    build_pairs(experiment)
    adapter = _fake_adapter(experiment)
    merge_stage(experiment, variant="lora", backend_factory=FakeBackend)
    report = score(experiment)
    assert report["comparison"]["delta_ex"] == pytest.approx(1 / 3)
    assert report["summary"]["lora_csc"]["csc_calls"] == 2
    assert report["comparison"]["mcnemar"]["lora_only_correct"] == 2
    assert report["summary"]["lora_csc"]["wrong_to_correct"] == 2
    raw_report = read_json(report["path"])
    assert (
        raw_report["provenance"]["generation"]["model_identity"]["model"]
        == experiment["generation"]["model"]
    )
    lora_manifest_path = artifact_path(experiment, "evaluation", "lora.csc-k4", 42).with_suffix(
        ".jsonl.manifest.json"
    )
    lora_manifest = read_json(lora_manifest_path)
    original_packages = lora_manifest["packages"]
    lora_manifest["packages"] = {"backend": "different inference runtime"}
    atomic_json(lora_manifest_path, lora_manifest)
    with pytest.raises(ValueError, match="identical inference package versions"):
        score(experiment)
    lora_manifest["packages"] = original_packages
    atomic_json(lora_manifest_path, lora_manifest)
    # A valid replacement adapter still cannot reuse predictions made by old weights.
    weights = adapter / "adapter_model.safetensors"
    weights.write_bytes(b"replacement test-only adapter")
    metadata_path = adapter.parent / "training_manifest.json"
    metadata = read_json(metadata_path)
    metadata["adapter_files"][weights.name] = file_hash(weights)
    atomic_json(metadata_path, metadata)
    validate_adapter(experiment, "evaluation", 42, 4)
    with pytest.raises(ValueError, match="different adapter weights"):
        score(experiment)
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="modified"):
        validate_adapter(experiment, "evaluation", 42, 4)


def test_invalid_merge_is_failure_not_silent_sc_fallback(experiment):
    prepare(experiment)
    _generate_execute(experiment, "evaluation")

    class InvalidMerge(FakeBackend):
        def generate(self, messages, *, n, seeds):
            return [
                [
                    {
                        "raw_text": "<think>unfinished",
                        "n_tokens": 1024,
                        "finish_reason": "length",
                        "prompt_tokens": 100,
                    }
                    for _ in range(n)
                ]
                for _ in messages
            ]

    merge_stage(experiment, backend_factory=InvalidMerge)
    records, _ = completed_artifact(artifact_path(experiment, "evaluation", "author.csc-k4", 42))
    assert all(r["selected_fp"] is None for r in records if r["csc_called"])
    result = score(experiment, baseline_only=True)
    assert result["summary"]["author_csc"]["merge_truncated"] == 4


def test_tokenization_masks_prompt_and_never_truncates_targets():
    class Tokenizer:
        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            text = "".join(m["role"] + ":" + m["content"] + ";" for m in messages)
            if add_generation_prompt:
                text += "assistant:"
            return list(text.encode())

    row = {
        "messages": [{"role": "user", "content": "schema"}],
        "target": "<answer>SELECT 1</answer>",
    }
    encoded = tokenize_example(Tokenizer(), row, 1000)
    assert -100 in encoded["labels"]
    prefix = encoded["labels"].count(-100)
    assert encoded["labels"][prefix:] == encoded["input_ids"][prefix:]
    assert tokenize_example(Tokenizer(), row, 10) is None


def test_structural_focus_detects_join_and_aggregation():
    assert focus_tags("SELECT * FROM a JOIN b ON a.id=b.id", "SELECT * FROM a") == [
        "join_disagreement"
    ]
    assert focus_tags("SELECT COUNT(*) FROM a", "SELECT SUM(x) FROM a") == [
        "aggregation_disagreement"
    ]
    assert focus_tags("SELECT * FROM a WHERE x=1", "SELECT * FROM a WHERE x=2") == []


def test_artifact_tampering_and_duplicate_ids_are_rejected(tmp_path):
    path = tmp_path / "cache.jsonl"
    with Journal(path, {"model": "x"}, ["a"]) as journal:
        journal.append({"id": "a", "prediction": "SQL"})
        journal.finish()
    path.write_text('{"id":"a","prediction":"tampered"}\n')
    with pytest.raises(ValueError, match="modified"):
        with Journal(path, {"model": "x"}, ["a"]):
            pass
    with pytest.raises(ValueError, match="unique"):
        Journal(tmp_path / "other", {}, ["same", "same"])


def test_legacy_csc_cannot_train_on_dev_labels(tmp_path):
    from research.csc_lora import build_csc_training_data, train_csc_lora

    with pytest.raises(ValueError, match="forbidden"):
        build_csc_training_data(
            [{"gold_sql": "SELECT 1", "source_split": "evaluation"}], None, tmp_path / "bad.jsonl"
        )
    with pytest.raises(RuntimeError, match="Legacy CSC training is disabled"):
        train_csc_lora(tmp_path / "unused", "wrong-generator", tmp_path / "output")


def test_prefix_k_reuses_generation_and_exposes_missing_timing(experiment):
    prepare(experiment)
    _generate_execute(experiment, "evaluation")
    merge_stage(experiment, k=2, backend_factory=FakeBackend)
    report = score(experiment, k=2, baseline_only=True)
    raw = read_json(report["path"])
    assert all(r["costs"]["author_csc"]["amortized_generation_s"] is None for r in raw["records"])


def test_plan_is_cpu_only_and_config_typos_fail(experiment, tmp_path):
    from research.slm_sql.__main__ import plan

    result = plan(experiment)
    assert not result["downloads_or_gpu_started"]
    bad = tmp_path / "bad.json"
    bad.write_text('{"training": {"epohs": 3}}')
    with pytest.raises(ValueError, match="Unknown config key"):
        load_config(bad)


def test_dev_only_baseline_can_add_train_splits_without_regeneration(experiment):
    source = Path(experiment["data"]["train_dir"]) / "train.json"
    backup = source.with_suffix(".saved")
    source.rename(backup)
    result = prepare(experiment, evaluation_only=True)
    assert set(result["splits"]) == {"evaluation"}
    _baseline(experiment)
    baseline = score(experiment, baseline_only=True)
    assert baseline["summary"]["author_csc"]["questions"] == 6
    backup.rename(source)
    full = prepare(experiment)
    assert set(full["splits"]) == {"train", "validation", "evaluation"}

    def forbidden(*args, **kwargs):
        raise AssertionError("Adding adaptation data must not regenerate final evaluation")

    assert generate_stage(experiment, "evaluation", backend_factory=forbidden)["reused"]
    assert merge_stage(experiment, backend_factory=forbidden)["reused"]


@pytest.mark.parametrize("stage", ["prepare", "build_pairs", "train_adapter"])
def test_concurrent_stage_rejected_before_expensive_work(experiment, stage):
    from research.slm_sql.adaptation import train_adapter

    functions = {"prepare": prepare, "build_pairs": build_pairs, "train_adapter": train_adapter}
    with exclusive_lock(Path(experiment["experiment_dir"]) / "locks" / f"{stage}.lock"):
        with pytest.raises(RuntimeError, match="Another process is writing"):
            functions[stage](experiment)


def test_private_hub_adapter_round_trip_preserves_pair_provenance(
    experiment, tmp_path, monkeypatch
):
    from scripts.csc_notebook import publish_csc_adapter, restore_csc_adapter

    prepare(experiment)
    for split in ("train", "validation"):
        _generate_execute(experiment, split)
    _baseline(experiment)
    build_pairs(experiment)
    adapter = _fake_adapter(experiment)
    uploaded = tmp_path / "hub"
    uploaded.mkdir()

    class FakeApi:
        private = True

        def __init__(self, token):
            assert token == "test-secret"

        def create_repo(self, **kwargs):
            assert kwargs["private"] and kwargs["repo_type"] == "model"

        def model_info(self, **kwargs):
            return SimpleNamespace(private=self.private)

        def upload_folder(self, *, folder_path, **kwargs):
            shutil.copytree(folder_path, uploaded, dirs_exist_ok=True)

        def upload_file(self, *, path_or_fileobj, path_in_repo, **kwargs):
            path = uploaded / path_in_repo
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path_or_fileobj, path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(HfApi=FakeApi, snapshot_download=lambda **kwargs: str(uploaded)),
    )
    monkeypatch.setenv("HF_REPO_ID", "owner/private-csc-adapter")
    monkeypatch.setenv("HF_TOKEN", "test-secret")

    FakeApi.private = False
    with pytest.raises(ValueError, match="public model repository"):
        publish_csc_adapter(experiment)
    assert not list(uploaded.iterdir())
    FakeApi.private = True
    published = publish_csc_adapter(experiment)
    assert published["private"]
    assert not (uploaded / "adaptation").exists()
    assert not (uploaded / "data").exists()
    assert (uploaded / ".csc" / "training_manifest.json").is_file()

    shutil.rmtree(adapter.parent)
    restored = restore_csc_adapter(experiment)
    assert restored["restored"]
    validate_adapter(experiment, "evaluation", 42, 4)
    assert restore_csc_adapter(experiment)["restored"]

    shutil.rmtree(adapter.parent)
    pairs_path = Path(experiment["experiment_dir"]) / "adaptation" / "manifest.json"
    pairs = read_json(pairs_path)
    pairs["signature"] = "another-experiment"
    atomic_json(pairs_path, pairs)
    with pytest.raises(ValueError, match="restored training pairs"):
        restore_csc_adapter(experiment)
    assert not adapter.exists()
