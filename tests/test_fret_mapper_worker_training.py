from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.train_fret_mapper import load_cache
from src.fret_mapper_worker_training import FretMapperTrainingError, _read_task_view
from src.model_bundle import load_model_bundle
from src.worker import PIPELINES, prepare_dataset_request, run_training_request


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    path = root / "assets" / "sha256" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _catalog_with_train_and_val(root: Path) -> None:
    records: list[dict[str, object]] = []
    for index in range(32):
        source_id = f"octave-src-{index:08x}"
        records.append(
            {
                "source_id": source_id,
                "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
                "rights": {
                    "training_use": "allowed",
                    "provenance": "Reviewed",
                    "license": "test-only",
                },
                "metadata": {"name": f"Fixture {index}"},
                "chart": {
                    "notes_midi": _asset(root, f"midi-{index}".encode(), "notes.mid"),
                    "instruments": {
                        "guitar": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART GUITAR"],
                        }
                    },
                },
                "audio": {"guitar": _asset(root, f"audio-{index}".encode(), "guitar.ogg")},
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "fret-mapper-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_guitar_fret_mapper_worker_uses_catalog_splits_and_packages_no_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog_with_train_and_val(tmp_path)
    task_view = tmp_path / "views" / "guitar-mapper.json"
    prepare_request = tmp_path / "prepare.json"
    pipeline_id = "strum.fret-mapper/guitar/v1"
    prepare_request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": pipeline_id,
                "output": str(task_view),
                "options": {},
            }
        )
    )
    prepare_dataset_request(prepare_request)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "fret_mapper_guitar"
    assert prepared["task"]["label_schema"]["id"] == "five-lane-fret-mapper-midi/v1"

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("build_mapper_dataset.py"):
            manifest = json.loads(
                Path(command[command.index("--catalog-manifest") + 1]).read_text()
            )
            cache_dir = Path(command[command.index("--cache-dir") + 1])
            cache_dir.mkdir(parents=True)
            for song in manifest["songs"]:
                np.savez_compressed(
                    cache_dir / f"{song['source_id']}.npz",
                    X=np.zeros((2, 95), dtype=np.float32),
                    Y=np.zeros((2, 5), dtype=np.float32),
                    song_id=song["source_id"],
                    split=song["split"],
                )
        elif command[1].endswith("train_fret_mapper.py"):
            checkpoint = Path(command[command.index("--out") + 1])
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"fret-mapper-weights")
            metrics = Path(command[command.index("--metrics-out") + 1])
            metrics.write_text(
                json.dumps(
                    {
                        "best_val_f1": 0.42,
                        "train_song_count": 20,
                        "val_song_count": 4,
                        "feature_dimension": 95,
                    }
                )
            )

    monkeypatch.setattr("src.fret_mapper_worker_training._run_script", fake_run)
    monkeypatch.setattr(
        "src.fret_mapper_worker_training.importlib.util.find_spec", lambda _: object()
    )
    output = tmp_path / "experiments" / "guitar-mapper-v1"
    train_request = tmp_path / "train.json"
    train_request.write_text(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": "catalog-guitar-mapper-v1", "epochs": 1, "device": "cpu"},
            }
        )
    )

    result = run_training_request(train_request)

    assert result["status"] == "completed"
    assert result["deployment_status"] == "requires_fret_mapper_profile_evaluation_and_packaging"
    assert [component["id"] for component in result["components"]] == ["fret_mapper.guitar"]
    assert "build_mapper_dataset.py" in commands[0][1]
    assert "train_fret_mapper.py" in commands[1][1]
    assert "--use-catalog-splits" in commands[1]
    assert str(tmp_path) not in (output / "bundle" / "strum-model-bundle.json").read_text()
    experiment = json.loads((output / "experiment.json").read_text())
    assert str(tmp_path) not in json.dumps(experiment)
    assert experiment["task_view"]["catalog_id"] == "fret-mapper-training-fixture"
    assert experiment["preprocessing"]["cache_counts"]["train"] > 0
    assert experiment["preprocessing"]["cache_counts"]["val"] > 0
    bundle = load_model_bundle(output / "bundle", check_files=True)
    assert set(bundle.components) == {"fret_mapper.guitar"}
    assert not bundle.profiles


def test_fret_mapper_rejects_the_wrong_catalog_task_kind(tmp_path: Path) -> None:
    _catalog_with_train_and_val(tmp_path)
    task_view = tmp_path / "wrong.json"
    task_view.write_text(
        json.dumps(
            {
                "format": "strum-catalog-task-manifest/v1",
                "task": {
                    "kind": "section_guitar",
                    "pipeline_id": "strum.section-classifier/guitar/v1",
                },
            }
        )
    )
    with pytest.raises(FretMapperTrainingError, match="exact catalog task view"):
        _read_task_view(task_view, tmp_path, "strum.fret-mapper/guitar/v1")


def test_mapper_trainer_uses_catalog_song_splits_without_reshuffling(tmp_path: Path) -> None:
    for source_id, split, value in (("train-song", "train", 1.0), ("val-song", "val", 2.0)):
        np.savez_compressed(
            tmp_path / f"{source_id}.npz",
            X=np.full((1, 95), value, dtype=np.float32),
            Y=np.zeros((1, 5), dtype=np.float32),
            song_id=source_id,
            split=split,
        )

    train_x, _train_y, val_x, _val_y, train_files, val_files = load_cache(
        tmp_path, use_catalog_splits=True
    )

    assert [path.name for path in train_files] == ["train-song.npz"]
    assert [path.name for path in val_files] == ["val-song.npz"]
    assert train_x[0, 0] == 1.0
    assert val_x[0, 0] == 2.0


@pytest.mark.parametrize(
    ("pipeline_id", "component"),
    [
        ("strum.fret-mapper/guitar/v1", "fret_mapper.guitar"),
        ("strum.fret-mapper/bass/v1", "fret_mapper.bass"),
    ],
)
def test_fret_mapper_pipelines_advertise_strict_worker_training(
    pipeline_id: str, component: str
) -> None:
    descriptor = next(item for item in PIPELINES if item.id == pipeline_id)

    assert descriptor.training_status == "available"
    assert descriptor.train_schema is not None
    assert descriptor.train_schema["required"] == ["model_id"]
    assert "catalog_root" not in descriptor.train_schema["properties"]
    assert descriptor.checkpoint_outputs == (component,)
    assert descriptor.inference_capability is None
    assert descriptor.training_requirements == (
        "strum_pitch_extra",
        "instrument_specific_profile_evaluation",
        "instrument_specific_profile_packaging",
    )


@pytest.mark.parametrize(
    ("pipeline_id", "required_gap"),
    [
        ("strum.instrument-chart/pro-guitar/v1", "pro_string_fret_target_encoder"),
        ("strum.instrument-chart/pro-bass/v1", "pro_bass_training_architecture"),
        ("strum.instrument-chart/pro-keys/v1", "pro_keys_pitch_target_encoder"),
        ("strum.section-classifier/guitar/v1", "section_training_worker"),
        ("strum.section-classifier/bass/v1", "section_runtime_integration"),
    ],
)
def test_prepare_only_pro_and_section_pipelines_publish_their_actual_gaps(
    pipeline_id: str, required_gap: str
) -> None:
    descriptor = next(item for item in PIPELINES if item.id == pipeline_id)

    assert descriptor.preparation_status == "available"
    assert descriptor.training_status == "planned"
    assert descriptor.train_schema is None
    assert required_gap in descriptor.training_requirements
    assert required_gap in descriptor.as_json()["training_requirements"]
