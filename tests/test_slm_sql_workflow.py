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
from research.slm_sql.config import artifact_path, load_config, merge_artifact_path
from research.slm_sql.data import _load, load_split, prepare
from research.slm_sql.evaluation import score
from research.slm_sql.execution import execute, result_fingerprint
from research.slm_sql.pipeline import (
    _execution_spec, _generation_spec, adaptive_stage, execute_stage, generate_stage,
    load_adaptive, load_generation, merge_stage,
)
from research.slm_sql.sampling import aggregate_suites, evaluate_suite

CORRECT = "SELECT SUM(amount) FROM sales"
WRONG = "SELECT COUNT(*) FROM sales"


def _shape(value):
    """Keep artifact contract tests independent of paths, hashes and timings."""
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    return type(value).__name__


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
                "evaluation": {"bootstrap_samples": 40, "sampling_k_values": [1, 2, 4]},
                "adaptive": {"min_k": 2, "max_k": 4, "batch_size": 2},
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


def _select_validation_policy(cfg, seed=42):
    from research.slm_sql.adaptive_eval import adaptive_sweep

    for variant in ("author", "lora"):
        for checkpoint in cfg["adaptive"]["checkpoints"]:
            merge_stage(cfg, split="validation", seed=seed, k=checkpoint,
                        variant=variant, backend_factory=FakeBackend)
    return adaptive_sweep(cfg, split="validation", seed=seed)


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


def test_bird_repeated_question_text_keeps_distinct_question_ids(experiment):
    root = Path(experiment["data"]["evaluation_dir"])
    records = json.loads((root / "dev.json").read_text())
    duplicate = {**records[0], "question_id": 863}
    second = {**duplicate, "question_id": 875}
    (root / "dev.json").write_text(json.dumps([duplicate, second]))

    loaded = _load(root, "dev")

    assert len(loaded) == 2
    assert loaded[0]["id"] != loaded[1]["id"]
    assert loaded[0]["question"] == loaded[1]["question"]
    assert loaded[0]["gold_sql"] == loaded[1]["gold_sql"]


def test_bird_duplicate_question_id_is_rejected(experiment):
    root = Path(experiment["data"]["evaluation_dir"])
    records = json.loads((root / "dev.json").read_text())
    duplicate = {**records[0], "question_id": 863}
    (root / "dev.json").write_text(json.dumps([duplicate, duplicate]))

    with pytest.raises(ValueError, match="Duplicate question_id"):
        _load(root, "dev")


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


def test_stage_specs_and_record_shapes_are_regression_locked(experiment):
    prepare(experiment)
    _, data_manifest = load_split(experiment, "evaluation")
    generation_spec = _generation_spec(experiment, "evaluation", 42, data_manifest)
    assert digest(_shape(generation_spec)) == (
        "768c26da1c27fcaad1151314f98e0af078601dd041aad2a7b504f7543044bae6"
    )

    generate_stage(experiment, "evaluation", backend_factory=FakeBackend)
    execute_stage(experiment, "evaluation")
    generated, generation_manifest = completed_artifact(
        artifact_path(experiment, "evaluation", "generation", 42)
    )
    executed, execution_manifest = completed_artifact(
        artifact_path(experiment, "evaluation", "execution", 42)
    )
    assert digest(_shape(generated[0])) == (
        "e89372d2d5db73f8611cc15529d790e09762c9b3cb1986cbd402cb6da0195d8f"
    )
    execution_spec = _execution_spec(experiment, "evaluation", 42, generation_manifest)
    assert digest(_shape(execution_spec)) == (
        "3557a60a91e2fbb6e043ffdd5cf7d63d376c70205db0e52609117de1f80260ce"
    )
    assert digest(_shape(executed[0])) == (
        "191c4106fdd68bcd3f71e0cda3acb455da28133356a59327a15efc0c903db161"
    )

    merge_stage(experiment, k=4, backend_factory=FakeBackend)
    merged, merge_manifest = completed_artifact(
        merge_artifact_path(experiment, "evaluation", 42, "author", 4)
    )
    assert digest(_shape(merge_manifest["spec"])) == (
        "3db79a2c4988fc5d26f82125e0244ce14056ff8b307c7f5a0ee5843c79b50267"
    )
    assert digest(_shape(merged[0])) == (
        "99cd45a6f85a632bfa597d605cff1fd22e2b11cdd5d9026d9cde5e413ee266f0"
    )

    report = read_json(score(experiment, k=4, baseline_only=True)["path"])
    assert digest(_shape(report["records"][0])) == (
        "64c254ae246aec4ae2dcfe47c860046d304e706a23677ea1ea2b60cb88b78fcb"
    )
    assert digest(_shape(report["summary"])) == (
        "7389a71239a6766609566cf02568592e150614a7b5ada2af234be341cfab0840"
    )


def _sampling_baseline(cfg, seed=42):
    generate_stage(cfg, "evaluation", seed, backend_factory=FakeBackend)
    execute_stage(cfg, "evaluation", seed)
    generate_stage(cfg, "evaluation", seed, mode="greedy", backend_factory=FakeBackend)
    execute_stage(cfg, "evaluation", seed, mode="greedy")
    if cfg["adaptive"]["enabled"]:
        adaptive_stage(cfg, seed=seed, backend_factory=FakeBackend)
    for k in cfg["evaluation"]["sampling_k_values"]:
        merge_stage(cfg, seed=seed, k=k, backend_factory=FakeBackend)
    if cfg["adaptive"]["enabled"]:
        merge_stage(cfg, seed=seed, source="adaptive", backend_factory=FakeBackend)


def test_author_sampling_suite_covers_greedy_fixed_and_adaptive(experiment):
    prepare(experiment)
    _sampling_baseline(experiment)
    result = evaluate_suite(experiment, baseline_only=True)
    report = read_json(result["path"])
    assert report["summary"]["greedy"]["questions"] == 6
    fixed = report["summary"]["fixed"]
    assert set(fixed) == {"1", "2", "4"}
    assert fixed["2"]["pass@k"] == pytest.approx(2 / 3)
    assert fixed["2"]["estimated_pass@k"] == pytest.approx(11 / 18)
    assert fixed["2"]["self_consistency@k"]["ex"] == pytest.approx(1 / 3)
    assert fixed["2"]["author_csc@k"]["questions"] == 6
    assert fixed["2"]["slm_sql@k"] == fixed["2"]["author_csc@k"]
    assert report["summary"]["adaptive"]["mean_samples"] == pytest.approx(4.0)
    assert "adaptive_author_csc" in report["summary"]["methods"]
    assert fixed["1"]["pass@k"] != fixed["1"]["estimated_pass@k"]
    for row in report["records"]:
        assert row["costs"]["sc@2"]["generation_s"] is None
        assert row["costs"]["author_csc@2"]["generation_merge_s"] is None
        cost = row["costs"]["author_csc@4"]
        assert cost["output_tokens"] == cost["generation_tokens"] + cost["merge_tokens"]
    assert Path(result["path"]).with_suffix(".csv").is_file()


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


def test_prospective_freeze_and_multi_seed_comparison(experiment):
    from research.slm_sql.protocol import freeze, verify_freeze, aggregate_seeds

    cfg = experiment
    prepare(cfg)
    _baseline(cfg)
    for split in ("train", "validation"):
        _generate_execute(cfg, split)
    build_pairs(cfg)
    _fake_adapter(cfg)
    _select_validation_policy(cfg)
    frozen = freeze(cfg, seeds=[42, 43], ks=[4])
    assert not frozen["reused"]
    assert freeze(cfg, seeds=[42, 43], ks=[4])["reused"]
    with pytest.raises(ValueError, match="not declared"):
        verify_freeze(cfg, 44, 4)
    for seed in (42, 43):
        generate_stage(cfg, "evaluation", seed=seed, backend_factory=FakeBackend)
        execute_stage(cfg, "evaluation", seed=seed)
        merge_stage(cfg, seed=seed, backend_factory=FakeBackend)
        merge_stage(cfg, seed=seed, variant="lora", backend_factory=FakeBackend)
    result = aggregate_seeds(cfg, seeds=[42, 43], k=4)
    report = read_json(result["path"])
    assert report["paired_delta_ci95"]["n"] == 6  # 6 questions, not 12 seed/question rows.
    assert len(report["paired_tests_by_seed"]) == 2
    assert report["summary"]["lora_csc"]["ex_mean"] > report["summary"]["author_csc"]["ex_mean"]
    single = read_json(Path(cfg["experiment_dir"]) / "reports/evaluation-seed42-k4.comparison.json")
    assert 0 <= single["oracle"]["pass_at_k"] <= 1
    assert single["summary"]["author_csc"]["by_database"]
    changed = {**cfg, "evaluation": {**cfg["evaluation"], "bootstrap_samples": 41}}
    with pytest.raises(ValueError, match="after the final freeze"):
        verify_freeze(changed, 42, 4)
    selected_path = Path(cfg["experiment_dir"]) / "adaptive" / "selected_policy.json"
    selected_path.write_bytes(selected_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="after the final freeze"):
        verify_freeze(cfg, 42, 4)


def test_adaptive_validation_selection_freeze_and_offline_evaluation(experiment):
    from research.slm_sql.adaptive_eval import adaptive_sweep, evaluate_adaptive_csc
    from research.slm_sql.protocol import freeze

    cfg = experiment
    prepare(cfg)
    with pytest.raises(ValueError, match="only on split validation"):
        adaptive_sweep(cfg, split="evaluation")
    _sampling_baseline(cfg)
    for split in ("train", "validation"):
        _generate_execute(cfg, split)
    build_pairs(cfg)
    _fake_adapter(cfg)
    selected = _select_validation_policy(cfg)
    assert selected["selected_policy"]
    with pytest.raises(ValueError, match="Run freeze"):
        evaluate_adaptive_csc(cfg, split="evaluation")
    freeze(cfg, seeds=[42], ks=[1, 2, 4])
    merge_stage(cfg, split="evaluation", variant="lora", k=4, backend_factory=FakeBackend)
    evaluated = evaluate_adaptive_csc(cfg, split="evaluation", seed=42)
    report = read_json(evaluated["path"])
    assert report["accuracy_basis"] == "offline_prefix_exact"
    assert report["policy_provenance"]["execution"]
    assert report["summary"]["adaptive"]["questions"] == 6
    for row in report["records"]:
        cost = row["costs"]["adaptive"]
        assert cost["output_tokens"] == cost["generation_tokens"] + cost["merge_tokens"]
        assert cost["generation_samples"] == row["adaptive"]["checkpoint"]


def test_freeze_cannot_be_claimed_after_final_lora_predictions(experiment):
    from research.slm_sql.protocol import freeze
    cfg = experiment
    prepare(cfg)
    _baseline(cfg)
    for split in ("train", "validation"):
        _generate_execute(cfg, split)
    build_pairs(cfg)
    _fake_adapter(cfg)
    merge_stage(cfg, variant="lora", backend_factory=FakeBackend)
    with pytest.raises(ValueError, match="already exist"):
        freeze(cfg)


@pytest.mark.parametrize("policy", ["beta", "margin"])
def test_adaptive_really_stops_early_from_nested_execution_votes(experiment, policy):
    cfg = copy.deepcopy(experiment)
    cfg["generation"]["evaluation_samples"] = 8
    cfg["adaptive"].update(policy=policy, min_k=4, max_k=8, batch_size=4)
    prepare(cfg)

    class Counting(FakeBackend):
        requests = []

        def generate(self, messages, *, n, seeds):
            self.requests.append((str(messages[0]), n, seeds[0]))
            return super().generate(messages, n=n, seeds=seeds)

    adaptive_stage(cfg, backend_factory=Counting)
    rows, _ = load_adaptive(cfg, "evaluation", 42)
    assert len(Counting.requests) == 10
    assert sum(n for _, n, _ in Counting.requests) == 40 < 6 * 8
    for row in rows:
        unanimous = "unanimous:" in row["question"]
        assert row["n_samples_used"] == (4 if unanimous else 8)
        assert row["stop_reason"] == ("consensus" if unanimous else "budget")
        if "invalid:" in row["question"]:
            assert row["selected_fp"] is None
            assert row["selected_sql"] == ""
        elif unanimous:
            assert row["selected_sql"] == CORRECT
            assert row["selected_fp"] == row["gold_execution"]["fingerprint"]
        else:
            assert row["selected_sql"] == WRONG  # first-generated group wins ties
        assert all(c["seed"] == r["seed"] for r in row["rounds"]
                   for c in row["candidates"][r["cumulative_samples"] - r["samples"]:r["cumulative_samples"]])

    def forbidden(*args, **kwargs):
        raise AssertionError("Complete online-adaptive cache must not load a model")

    assert adaptive_stage(cfg, backend_factory=forbidden)["reused"]
    assert not artifact_path(cfg, "evaluation", "generation", 42).exists()
    changed = copy.deepcopy(cfg)
    changed["adaptive"]["threshold"] = 0.99
    with pytest.raises(ValueError, match="no longer matches"):
        load_adaptive(changed, "evaluation", 42)
    changed = copy.deepcopy(cfg)
    changed["execution"]["timeout"] += 1
    with pytest.raises(ValueError, match="no longer matches"):
        load_adaptive(changed, "evaluation", 42)


def test_adaptive_resume_retries_only_incomplete_question_and_checks_packages(experiment):
    prepare(experiment)

    class Interrupting(FakeBackend):
        count = 0

        def generate(self, *args, **kwargs):
            self.count += 1
            if self.count == 3:
                raise RuntimeError("adaptive interruption")
            return super().generate(*args, **kwargs)

    with pytest.raises(RuntimeError, match="adaptive interruption"):
        adaptive_stage(experiment, backend_factory=Interrupting)
    path = artifact_path(experiment, "evaluation", "adaptive", 42)
    before = path.read_bytes()
    assert len(before.splitlines()) == 1

    class Upgraded(FakeBackend):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.versions = {"backend": "upgraded"}

    with pytest.raises(ValueError, match="Inference package versions changed"):
        adaptive_stage(experiment, backend_factory=Upgraded)
    adaptive_stage(experiment, backend_factory=FakeBackend)
    assert path.read_bytes().startswith(before)
    rows, _ = load_adaptive(experiment, "evaluation", 42)
    assert len(rows) == 6
    assert len({r["id"] for r in rows}) == 6


def test_router_adaptive_merge_only_loads_merger_for_call_csc(experiment):
    cfg = copy.deepcopy(experiment)
    cfg["adaptive"].update(policy="router", min_k=2, max_k=4, batch_size=2)
    prepare(cfg)
    adaptive_stage(cfg, backend_factory=FakeBackend)

    class MergerOnly(FakeBackend):
        def __init__(self, settings, **kwargs):
            if settings["model"] == cfg["generation"]["model"]:
                raise AssertionError("adaptive-merge must not load the generator")
            super().__init__(settings, **kwargs)

    merge_stage(cfg, source="adaptive", backend_factory=MergerOnly)
    rows, _ = completed_artifact(merge_artifact_path(cfg, "evaluation", 42, "author", source="adaptive"))
    adaptive_rows, _ = load_adaptive(cfg, "evaluation", 42)
    assert any(row["csc_called"] for row in rows)
    assert all(output["csc_called"] == (record["adaptive_action"] == "CALL_CSC")
               for output, record in zip(rows, adaptive_rows))


def test_budget_csc_runs_only_when_policy_has_not_reached_consensus(experiment):
    cfg = copy.deepcopy(experiment)
    cfg["adaptive"].update(
        policy="beta", min_k=4, max_k=4, batch_size=2, threshold=0.95,
        call_csc_at_budget=True,
    )
    prepare(cfg)
    adaptive_stage(cfg, backend_factory=FakeBackend)
    rows, _ = load_adaptive(cfg, "evaluation", 42)
    by_kind = {row["question"].split(":", 1)[0]: row for row in rows}

    assert by_kind["unanimous"]["adaptive_action"] == "STOP"
    assert by_kind["unanimous"]["stop_reason"] == "consensus"
    assert by_kind["disputed"]["adaptive_action"] == "CALL_CSC"
    assert by_kind["disputed"]["stop_reason"] == "budget_csc"
    assert by_kind["invalid"]["adaptive_action"] == "STOP"
    assert by_kind["invalid"]["stop_reason"] == "budget"

    merge_stage(cfg, source="adaptive", backend_factory=FakeBackend)
    merged, _ = completed_artifact(
        merge_artifact_path(cfg, "evaluation", 42, "author", source="adaptive")
    )
    merged_by_id = {row["id"]: row for row in merged}
    for kind, row in by_kind.items():
        assert merged_by_id[row["id"]]["csc_called"] is (kind == "disputed")


def test_offline_budget_csc_preserves_consensus_stop(experiment):
    from research.slm_sql.adaptive_eval import _score_policy

    cfg = copy.deepcopy(experiment)
    cfg["adaptive"].update(
        policy="beta", min_k=4, max_k=4, batch_size=2, threshold=0.95,
        call_csc_at_budget=True,
    )
    prepare(cfg)
    _baseline(cfg)
    author, manifest = completed_artifact(
        merge_artifact_path(cfg, "evaluation", 42, "author", 4)
    )
    merges = {
        "author": {4: (author, manifest)},
        "lora": {4: (author, manifest)},
    }
    result = _score_policy(
        cfg,
        "evaluation",
        42,
        {"policy": "beta", "threshold": 0.95, "min_k": 4, "max_k": 4},
        [4],
        merges,
    )
    by_kind = {row["question"].split(":", 1)[0]: row for row in result["records"]}

    assert by_kind["unanimous"]["adaptive"]["action"] == "STOP"
    assert by_kind["unanimous"]["costs"]["adaptive"]["csc_called"] is False
    assert by_kind["disputed"]["adaptive"]["action"] == "CALL_CSC"
    assert by_kind["disputed"]["costs"]["adaptive"]["csc_called"] is True
    assert by_kind["invalid"]["adaptive"]["action"] == "STOP_AT_BUDGET"
    assert by_kind["invalid"]["costs"]["adaptive"]["csc_called"] is False


def test_greedy_is_a_separate_temperature_zero_request(experiment):
    prepare(experiment)

    class TemperatureAware(FakeBackend):
        requests = []

        def generate(self, messages, *, n, seeds):
            self.requests.append((self.settings.copy(), n, seeds.copy()))
            outputs = super().generate(messages, n=n, seeds=seeds)
            if self.settings["temperature"] == 0:
                for message, completions in zip(messages, outputs):
                    if "disputed:" in str(message):
                        completions[0]["raw_text"] = f"<answer>{CORRECT}</answer>"
            return outputs

    generate_stage(experiment, "evaluation", backend_factory=TemperatureAware)
    generate_stage(experiment, "evaluation", mode="greedy", backend_factory=TemperatureAware)
    fixed, _ = load_generation(experiment, "evaluation", 42)
    greedy, manifest = load_generation(experiment, "evaluation", 42, mode="greedy")
    assert manifest["spec"]["sampling"]["temperature"] == 0
    assert manifest["spec"]["sampling"]["top_p"] == 1
    requests = TemperatureAware.requests
    assert len(requests) == 6  # 3 fixed batches and 3 independent greedy batches
    assert all(n == 4 and settings["temperature"] > 0 for settings, n, _ in requests[:3])
    assert all(n == 1 and settings["temperature"] == 0 for settings, n, _ in requests[3:])
    assert fixed[0]["candidates"][0]["sql"] == WRONG
    assert greedy[0]["candidates"][0]["sql"] == CORRECT
    assert fixed[0]["candidates"][0]["seed"] == requests[0][2][0]
    assert greedy[0]["candidates"][0]["seed"] == requests[3][2][0]
    assert requests[0][2] != requests[3][2]


def test_sampling_suite_rejects_missing_csc_and_mismatched_generator_packages(experiment):
    prepare(experiment)
    _sampling_baseline(experiment)
    path = merge_artifact_path(experiment, "evaluation", 42, "author", 1)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    saved = sidecar.read_bytes()
    sidecar.unlink()
    with pytest.raises(FileNotFoundError, match="csc-k1"):
        evaluate_suite(experiment, baseline_only=True)
    sidecar.write_bytes(saved)
    greedy = artifact_path(experiment, "evaluation", "greedy.generation", 42)
    sidecar = greedy.with_suffix(greedy.suffix + ".manifest.json")
    manifest = read_json(sidecar)
    manifest["packages"] = {"backend": "different"}
    atomic_json(sidecar, manifest)
    with pytest.raises(ValueError, match="identical generator weights and inference package"):
        evaluate_suite(experiment, generator_only=True)


def test_sampling_suite_supports_disabled_adaptive_and_generator_only(experiment):
    cfg = copy.deepcopy(experiment)
    cfg["adaptive"]["enabled"] = False
    prepare(cfg)
    _generate_execute(cfg, "evaluation")
    generate_stage(cfg, "evaluation", mode="greedy", backend_factory=FakeBackend)
    execute_stage(cfg, "evaluation", mode="greedy")
    result = evaluate_suite(cfg, generator_only=True)
    report = read_json(result["path"])
    assert set(report["summary"]["methods"]) == {"greedy", "sc@1", "sc@2", "sc@4"}
    assert report["summary"]["adaptive"] is None
    assert report["summary"]["fixed"]["4"]["author_csc@k"] is None


def test_sampling_comparison_all_k_adaptive_multi_seed_and_cached_cli(experiment):
    from research.slm_sql.protocol import freeze, verify_freeze

    cfg = experiment
    prepare(cfg)
    _sampling_baseline(cfg)
    for split in ("train", "validation"):
        _generate_execute(cfg, split)
    build_pairs(cfg)
    _fake_adapter(cfg)
    _select_validation_policy(cfg)
    frozen = freeze(cfg, seeds=[42, 43])
    assert frozen["spec"]["k_values"] == [1, 2, 4]
    assert verify_freeze(cfg, 42) == frozen["signature"]
    changed = copy.deepcopy(cfg)
    changed["adaptive"]["threshold"] = 0.99
    with pytest.raises(ValueError, match="after the final freeze"):
        verify_freeze(changed, 42)
    with pytest.raises(ValueError, match="not declared"):
        verify_freeze(cfg, 44)
    with pytest.raises(ValueError, match="not declared"):
        verify_freeze(cfg, 42, 3)
    for seed in (42, 43):
        if seed == 43:
            _sampling_baseline(cfg, seed)
        for k in (1, 2, 4):
            merge_stage(cfg, seed=seed, k=k, variant="lora", backend_factory=FakeBackend)
        merge_stage(cfg, seed=seed, source="adaptive", variant="lora", backend_factory=FakeBackend)
    result = aggregate_suites(cfg, seeds=[42, 43])
    report = read_json(result["path"])
    methods = report["summary"]
    assert methods["lora_csc@4"]["ex_mean"] > methods["author_csc@4"]["ex_mean"]
    assert methods["adaptive_lora_csc"]["ex_mean"] == methods["adaptive_author_csc"]["ex_mean"]
    assert methods["lora_csc@1"]["ex_mean"] == methods["sc@1"]["ex_mean"]
    assert methods["lora_csc@4"]["seed_averaged_question_ci95"]["n"] == 6
    pair = report["paired_comparisons"]["adaptive_lora_csc_vs_adaptive_author_csc"]
    assert pair["paired_delta_ci95"]["n"] == 6  # not 12 seed/question observations
    assert len(pair["tests_by_seed"]) == 2
    for variant in ("author", "lora"):
        rows, manifest = completed_artifact(merge_artifact_path(cfg, "evaluation", 42, variant, 1))
        assert all(not r["csc_called"] for r in rows)
        assert "model_identity" not in manifest  # no unnecessary merger load at k=1
    cli_config = Path(cfg["experiment_dir"]).parent / "sampling_cli.json"
    atomic_json(cli_config, cfg)
    for command in ("sampling-baseline", "sampling-final"):
        cached = subprocess.run(
            [sys.executable, "-m", "research.slm_sql", "--config", str(cli_config), command],
            capture_output=True, text=True,
        )
        assert cached.returncode == 0, cached.stderr
        assert '"sampling": "complete"' in cached.stdout
    sidecar = merge_artifact_path(cfg, "evaluation", 42, "lora", source="adaptive")
    sidecar = sidecar.with_suffix(sidecar.suffix + ".manifest.json")
    metadata = read_json(sidecar)
    metadata["packages"] = {"backend": "other"}
    atomic_json(sidecar, metadata)
    with pytest.raises(ValueError, match="identical inference package"):
        evaluate_suite(cfg)


def test_sampling_config_legacy_budget_and_invalid_values(experiment, tmp_path):
    path = tmp_path / "small.json"
    atomic_json(path, {"generation": {"evaluation_samples": 3}})
    cfg = load_config(path)
    assert cfg["evaluation"]["sampling_k_values"] == [1, 2]
    assert cfg["adaptive"]["max_k"] == cfg["adaptive"]["min_k"] == 3
    for overrides, match in (
        ({"evaluation": {"sampling_k_values": [1, 1]}}, "unique positive"),
        ({"evaluation": {"sampling_k_values": [65]}}, "cannot exceed"),
        ({"adaptive": {"enabled": "true"}}, "boolean"),
        ({"adaptive": {"threshold": 0}}, "threshold"),
    ):
        atomic_json(path, overrides)
        with pytest.raises(ValueError, match=match):
            load_config(path)
    atomic_json(path, {"adaptive": {"policy": "router"}})
    assert load_config(path)["adaptive"]["policy"] == "router"
