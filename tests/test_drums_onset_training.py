from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.drums_onset_training import (
    EXPERIMENT_FORMAT,
    DrumsTrainingError,
    DrumsTrainingOptions,
    _experiment_payload,
    run_drums_onset_training,
)


def test_training_options_are_strict_and_do_not_accept_catalog_locations(tmp_path: Path) -> None:
    options = DrumsTrainingOptions.from_mapping({"model_id": "drums-local-v1", "epochs": 1})

    assert options.profile == "onset_classifier_v2"
    assert options.epochs == 1
    with pytest.raises(DrumsTrainingError, match="unsupported"):
        DrumsTrainingOptions.from_mapping(
            {"model_id": "drums-local-v1", "catalog_root": str(tmp_path)}
        )
    with pytest.raises(DrumsTrainingError, match="epochs"):
        DrumsTrainingOptions.from_mapping({"model_id": "drums-local-v1", "epochs": True})


def test_experiment_ledger_links_task_preprocessing_and_checkpoint_without_paths(
    tmp_path: Path,
) -> None:
    config = tmp_path / "training-config.yaml"
    config.write_text("training: {}\n")
    checkpoint = tmp_path / "checkpoints" / "best_f1.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    task_view = {
        "format": "strum-drums-onset-catalog-manifest/v1",
        "catalog": {"catalog_id": "curated", "content_sha256": "c" * 64},
        "task": {"pipeline_id": "drums.onset-classifier", "pipeline_version": 1},
        "songs": [],
    }
    index = tmp_path / "train_index.json"
    index.write_text("{}")
    options = DrumsTrainingOptions.from_mapping({"model_id": "drums-local-v1", "epochs": 1})

    payload = _experiment_payload(
        task_view=task_view,
        options=options,
        indexes=[
            {
                "split": "train",
                "index_name": index.name,
                "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
                "total_onsets": 1,
                "lineage": {"catalog_id": "curated", "source_ids": ["octave-src-a"]},
            }
        ],
        config_path=config,
        checkpoint=checkpoint,
        metrics={"overall_f1": 0.5, "val_loss": 1.0, "num_samples": 1},
    )

    assert payload["format"] == EXPERIMENT_FORMAT
    assert payload["task_view"]["catalog_id"] == "curated"
    assert payload["checkpoint"]["name"] == "checkpoints/best_f1.pt"
    assert payload["checkpoint"]["deployment_status"] == "requires_profile_packaging"
    assert str(tmp_path) not in json.dumps(payload)


def test_worker_training_records_catalog_and_checkpoint_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_view = tmp_path / "task-view.json"
    task_view.write_text(
        json.dumps(
            {
                "format": "strum-drums-onset-catalog-manifest/v1",
                "catalog": {"catalog_id": "curated", "content_sha256": "c" * 64},
                "task": {"pipeline_id": "drums.onset-classifier", "pipeline_version": 1},
                "songs": [],
            }
        )
    )
    output = tmp_path / "experiment"

    def fake_prepare(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        return [
            {
                "split": split,
                "index_name": f"{split}_index.json",
                "index_sha256": split * 16,
                "total_onsets": 1,
                "lineage": {"catalog_id": "curated", "source_ids": ["octave-src-a"]},
            }
            for split in ("train", "val", "test")
        ]

    def fake_config(result_output: Path, _options: DrumsTrainingOptions) -> object:
        (result_output / "training-config.yaml").write_text("training: {}\n")
        return object()

    def fake_train(_config: object, _options: DrumsTrainingOptions) -> dict[str, object]:
        checkpoint = output / "checkpoints" / "best_f1.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        return {
            "checkpoint": str(checkpoint),
            "metrics": {"overall_f1": 0.5, "val_loss": 1.0, "num_samples": 1},
        }

    monkeypatch.setattr("src.drums_onset_training._prepare_cache", fake_prepare)
    monkeypatch.setattr("src.drums_onset_training._load_training_config", fake_config)
    monkeypatch.setattr("src.drums_onset_training._run_existing_trainer", fake_train)

    result = run_drums_onset_training(
        task_view,
        output,
        {"model_id": "drums-local-v1", "epochs": 1},
        catalog_root=tmp_path,
    )

    ledger = json.loads((output / "experiment.json").read_text())
    assert result["experiment_name"] == "experiment.json"
    assert ledger["task_view"]["catalog_id"] == "curated"
    assert ledger["preprocessing"]["splits"][0]["split"] == "train"
    assert ledger["checkpoint"]["name"] == "checkpoints/best_f1.pt"
