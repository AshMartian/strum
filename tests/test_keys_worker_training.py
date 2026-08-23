from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest
import yaml

from src.catalog_task_manifest import build_catalog_task_manifest
from src.keys_worker_training import KeysTrainingError, _read_task_view
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
                "metadata": {"name": f"Keys fixture {index}"},
                "chart": {
                    "notes_midi": _asset(root, f"midi-{index}".encode(), "notes.mid"),
                    "instruments": {
                        "keys": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART KEYS"],
                        }
                    },
                },
                "audio": {"keys": _asset(root, f"audio-{index}".encode(), "keys.ogg")},
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "keys-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_keys_worker_trains_from_part_keys_task_view_and_packages_experiment_only_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog_with_train_and_val(tmp_path)
    task_view = tmp_path / "views" / "keys.json"
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "keys.onset-fret/v1",
                "output": str(task_view),
                "options": {},
            }
        )
    )
    prepare_dataset_request(request)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "keys_onset_fret"
    assert prepared["task"]["label_schema"]["track_prefixes"] == ["PART KEYS"]
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_guitar_v1.py"):
            config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
            for subdir in ("keys_v1_onset", "keys_v1_fret"):
                stage = checkpoint_dir / subdir
                stage.mkdir(parents=True, exist_ok=True)
                (stage / "best.pt").write_bytes(f"{subdir}-weights".encode())
                (stage / "history.json").write_text(
                    json.dumps([{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.8}])
                )

    monkeypatch.setattr("src.keys_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "keys-v1"
    train_request = tmp_path / "train.json"
    train_request.write_text(
        json.dumps(
            {
                "pipeline_id": "keys.onset-fret/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "catalog-keys-v1",
                    "epochs": 1,
                    "batch_size": 2,
                    "device": "cpu",
                },
            }
        )
    )
    result = run_training_request(train_request)

    assert result["status"] == "completed"
    assert result["pipeline_id"] == "keys.onset-fret/v1"
    assert result["deployment_status"] == "requires_keys_profile_evaluation_and_packaging"
    assert [component["id"] for component in result["components"]] == ["keys.onset", "keys.fret"]
    assert commands[0][commands[0].index("--instrument") + 1] == "keys"
    bundle = output / "bundle"
    manifest = (bundle / "strum-model-bundle.json").read_text()
    experiment = (output / "experiment.json").read_text()
    assert str(tmp_path) not in manifest
    assert str(tmp_path) not in experiment
    assert "guitar.onset" not in manifest
    assert "bass.onset" not in manifest
    config = json.loads((bundle / "configs" / "keys-training-config.json").read_text())
    assert config["format"] == "strum-keys-neural-model-config/v1"
    assert config["instrument"] == "keys"
    assert set(load_model_bundle(bundle, check_files=True).components) == {
        "keys.onset",
        "keys.fret",
    }


def test_keys_worker_rejects_generic_keys_or_bass_task_views(tmp_path: Path) -> None:
    _catalog_with_train_and_val(tmp_path)
    generic = build_catalog_task_manifest(tmp_path, "keys")
    generic_path = tmp_path / "generic-keys.json"
    generic_path.write_text(json.dumps(generic))
    with pytest.raises(KeysTrainingError, match="Keys onset/fret catalog task view"):
        _read_task_view(generic_path, tmp_path)

    exact = build_catalog_task_manifest(tmp_path, "keys_onset_fret")
    exact["task"]["label_schema"] = {"track_prefixes": ["PART BASS"]}
    exact_path = tmp_path / "wrong-track.json"
    exact_path.write_text(json.dumps(exact))
    with pytest.raises(KeysTrainingError, match="Keys onset/fret catalog task view"):
        _read_task_view(exact_path, tmp_path)


def test_keys_pipeline_advertises_strict_training_without_inference() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "keys.onset-fret/v1")
    assert descriptor.training_status == "available"
    assert descriptor.train_schema is not None
    assert descriptor.train_schema["required"] == ["model_id"]
    assert "catalog_root" not in descriptor.train_schema["properties"]
    assert descriptor.checkpoint_outputs == ("keys.onset", "keys.fret")
    assert descriptor.inference_capability is None


def test_five_lane_preprocessor_reads_part_keys_not_guitar(tmp_path: Path) -> None:
    from scripts.preprocess_guitar_windows import parse_onsets_from_manifest

    source = mido.MidiFile(ticks_per_beat=480)
    guitar = mido.MidiTrack()
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    keys = mido.MidiTrack()
    keys.append(mido.MetaMessage("track_name", name="PART KEYS", time=0))
    keys.append(mido.Message("note_on", note=100, velocity=100, time=240))
    source.tracks.extend((guitar, keys))
    midi_path = tmp_path / "notes.mid"
    source.save(midi_path)

    assert parse_onsets_from_manifest(midi_path, label_track="PART KEYS") == [(250.0, {4})]
