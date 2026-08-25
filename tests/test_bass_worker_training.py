from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest
import yaml

from src.bass_worker_training import BassTrainingError, _read_task_view
from src.catalog_task_manifest import build_catalog_task_manifest
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
                        "bass": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART BASS"],
                        }
                    },
                },
                "audio": {"bass": _asset(root, f"audio-{index}".encode(), "bass.ogg")},
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "bass-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_bass_worker_trains_from_its_catalog_task_view_and_packages_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog_with_train_and_val(tmp_path)
    monkeypatch.setattr(
        "src.catalog_task_manifest.classify_five_lane_runtime_source", lambda *_args, **_kwargs: None
    )
    task_view = tmp_path / "views" / "bass.json"
    prepare_request = tmp_path / "prepare.json"
    prepare_request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "bass.onset-fret/v1",
                "output": str(task_view),
                "options": {},
            }
        )
    )
    prepare_dataset_request(prepare_request)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "bass_onset_fret"
    assert prepared["task"]["label_schema"]["track_names"] == ["PART BASS"]
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_guitar_v1.py"):
            config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
            for subdir in ("bass_v1_onset", "bass_v1_fret"):
                stage = checkpoint_dir / subdir
                stage.mkdir(parents=True, exist_ok=True)
                (stage / "best.pt").write_bytes(f"{subdir}-weights".encode())
                (stage / "history.json").write_text(
                    json.dumps([{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.8}])
                )

    monkeypatch.setattr("src.bass_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "bass-v1"
    train_request = tmp_path / "train.json"
    train_request.write_text(
        json.dumps(
            {
                "pipeline_id": "bass.onset-fret/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "catalog-bass-v1",
                    "epochs": 1,
                    "batch_size": 2,
                    "device": "cpu",
                },
            }
        )
    )

    result = run_training_request(train_request)

    assert result["status"] == "completed"
    assert result["pipeline_id"] == "bass.onset-fret/v1"
    assert result["model_id"] == "catalog-bass-v1"
    assert result["deployment_status"] == "requires_bass_profile_evaluation_and_packaging"
    assert [component["id"] for component in result["components"]] == ["bass.onset", "bass.fret"]
    assert "preprocess_guitar_windows.py" in commands[0][1]
    assert commands[0][commands[0].index("--instrument") + 1] == "bass"
    assert "train_guitar_v1.py" in commands[1][1]
    bundle = output / "bundle"
    manifest = (bundle / "strum-model-bundle.json").read_text()
    experiment = (output / "experiment.json").read_text()
    assert str(tmp_path) not in manifest
    assert str(tmp_path) not in experiment
    assert "guitar.onset" not in manifest
    assert "guitar.fret" not in manifest
    experiment_data = json.loads(experiment)
    assert experiment_data["task_view"]["catalog_id"] == "bass-training-fixture"
    assert experiment_data["deployment_status"] == "requires_bass_profile_evaluation_and_packaging"
    portable_config = json.loads((bundle / "configs" / "bass-training-config.json").read_text())
    assert portable_config["format"] == "strum-bass-neural-model-config/v1"
    assert portable_config["instrument"] == "bass"
    assert portable_config["model_implementation"] == "five-lane-crnn/v1"
    loaded = load_model_bundle(bundle, check_files=True)
    assert set(loaded.components) == {"bass.onset", "bass.fret"}


def test_bass_worker_rejects_generic_bass_or_guitar_task_views(tmp_path: Path) -> None:
    _catalog_with_train_and_val(tmp_path)
    generic_bass = build_catalog_task_manifest(tmp_path, "bass")
    generic_path = tmp_path / "generic-bass.json"
    generic_path.write_text(json.dumps(generic_bass))
    with pytest.raises(BassTrainingError, match="Bass onset/fret catalog task view"):
        _read_task_view(generic_path, tmp_path)


def test_bass_pipeline_advertises_a_strict_worker_training_schema() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "bass.onset-fret/v1")

    assert descriptor.training_status == "available"
    assert descriptor.train_schema is not None
    assert descriptor.train_schema["required"] == ["model_id"]
    assert "catalog_root" not in descriptor.train_schema["properties"]
    assert descriptor.catalog_requirements["label_schema"] == "five-lane-midi/v2"
    assert descriptor.catalog_requirements["label_tracks"] == ["PART BASS"]
    assert descriptor.checkpoint_outputs == ("bass.onset", "bass.fret")
    assert descriptor.inference_capability == "bass.neural-v1-expert/v1"


def test_five_lane_preprocessor_reads_part_bass_not_part_guitar(tmp_path: Path) -> None:
    from scripts.preprocess_guitar_windows import parse_onsets_from_manifest

    midi_path = tmp_path / "notes.mid"
    source = mido.MidiFile(ticks_per_beat=480)
    guitar = mido.MidiTrack()
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    guitar.append(mido.Message("note_off", note=96, velocity=0, time=120))
    bass = mido.MidiTrack()
    bass.append(mido.MetaMessage("track_name", name="PART BASS", time=0))
    bass.append(mido.Message("note_on", note=99, velocity=100, time=240))
    bass.append(mido.Message("note_off", note=99, velocity=0, time=120))
    bass_alt = mido.MidiTrack()
    bass_alt.append(mido.MetaMessage("track_name", name="PART BASS ALT", time=0))
    bass_alt.append(mido.Message("note_on", note=96, velocity=100, time=0))
    bass_alt.append(mido.Message("note_off", note=96, velocity=0, time=120))
    source.tracks.extend((guitar, bass_alt, bass))
    source.save(midi_path)

    assert parse_onsets_from_manifest(midi_path, label_track="PART BASS") == [(250.0, {3})]
