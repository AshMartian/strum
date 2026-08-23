from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path

import mido
import numpy as np
import pytest
import soundfile as sf

from scripts.preprocess_vocal_lyric_alignment import (
    TOKEN_TO_ID,
    prepare_vocal_lyric_alignment,
    tokenize_lyric,
)
from scripts.preprocess_vocal_phrase_boundaries import prepare_vocal_phrase_boundaries
from scripts.preprocess_vocal_talky_activity import prepare_vocal_talky_activity
from scripts.preprocess_vocals_frames import (
    PITCH_CLASS_COUNT,
    VocalsPreprocessError,
    parse_vocal_events,
    prepare_vocals_frames,
)
from scripts.train_vocal_lyric_alignment import TrainingSettings as LyricTrainingSettings
from scripts.train_vocal_lyric_alignment import train_vocal_lyric_alignment
from scripts.train_vocal_talky_activity import TrainingSettings as TalkyTrainingSettings
from scripts.train_vocal_talky_activity import train_vocal_talky_activity
from src.catalog_task_manifest import build_catalog_task_manifest
from src.model_bundle import load_model_bundle
from src.vocals_worker_training import VocalsTrainingError, _read_task_view
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
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/wav",
    }


def _vocal_midi() -> bytes:
    source = mido.MidiFile(ticks_per_beat=480)
    guitar = mido.MidiTrack()
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    guitar.append(mido.Message("note_off", note=96, velocity=0, time=480))
    vocals = mido.MidiTrack()
    vocals.append(mido.MetaMessage("track_name", name="PART VOCALS", time=0))
    vocals.append(mido.Message("note_on", note=105, velocity=100, time=0))
    vocals.append(mido.MetaMessage("lyrics", text="hello", time=0))
    vocals.append(mido.Message("note_on", note=60, velocity=100, time=0))
    vocals.append(mido.Message("note_off", note=60, velocity=0, time=480))
    vocals.append(mido.Message("note_on", note=106, velocity=100, time=0))
    vocals.append(mido.Message("note_on", note=96, velocity=100, time=0))
    vocals.append(mido.Message("note_off", note=96, velocity=0, time=240))
    source.tracks.extend((guitar, vocals))
    stream = BytesIO()
    source.save(file=stream)
    return stream.getvalue()


def _wav() -> bytes:
    stream = BytesIO()
    sf.write(stream, np.full(22_050, 0.01, dtype=np.float32), 22_050, format="WAV")
    return stream.getvalue()


def _catalog(root: Path, *, materialize: bool = False) -> None:
    records: list[dict[str, object]] = []
    midi = _vocal_midi()
    audio = _wav() if materialize else b"fixture-audio"
    for index in range(32):
        records.append(
            {
                "source_id": f"octave-src-{index:08x}",
                "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
                "rights": {
                    "training_use": "allowed",
                    "provenance": "Reviewed",
                    "license": "test-only",
                },
                "metadata": {"name": f"Fixture {index}"},
                "chart": {
                    "notes_midi": _asset(root, midi, f"notes-{index}.mid"),
                    "instruments": {
                        "vocals": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART VOCALS"],
                        }
                    },
                },
                "audio": {"vocals": _asset(root, audio, f"vocals-{index}.wav")},
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "vocals-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_vocals_worker_packages_a_catalog_backed_experiment_without_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path)
    task_view = tmp_path / "views" / "vocals.json"
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "vocals.note-activity/v1",
                "output": str(task_view),
                "options": {"split_ratios": [50, 50, 0]},
            }
        )
    )
    prepare_dataset_request(request)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "vocals_activity"
    assert prepared["task"]["label_schema"]["track_prefixes"] == ["PART VOCALS"]
    assert prepared["task"]["split_ratios"] == [50, 50, 0]
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_vocals_activity.py"):
            checkpoints = Path(command[command.index("--checkpoint-dir") + 1])
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "best.pt").write_bytes(b"vocal-frame-weights")
            (checkpoints / "history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "train_loss": 0.5,
                            "val_loss": 0.4,
                            "val_activity_f1": 0.8,
                            "val_pitch_accuracy": 0.7,
                        }
                    ]
                )
            )

    monkeypatch.setattr("src.vocals_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "vocals-v1"
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps(
            {
                "pipeline_id": "vocals.note-activity/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": "catalog-vocals-v1", "epochs": 1, "device": "cpu"},
            }
        )
    )
    result = run_training_request(train)

    assert result["status"] == "completed"
    assert result["deployment_status"] == "requires_vocals_profile_evaluation_and_packaging"
    assert [component["id"] for component in result["components"]] == [
        "vocals.frame_activity_pitch"
    ]
    assert "preprocess_vocals_frames.py" in commands[0][1]
    assert "train_vocals_activity.py" in commands[1][1]
    bundle = output / "bundle"
    assert str(tmp_path) not in (bundle / "strum-model-bundle.json").read_text()
    assert str(tmp_path) not in (output / "experiment.json").read_text()
    model = load_model_bundle(bundle, check_files=True)
    assert set(model.components) == {"vocals.frame_activity_pitch"}
    assert model.profiles == {}
    config = json.loads((bundle / "configs" / "vocals-frame-activity.json").read_text())
    assert config["excluded_outputs"] == ["lyrics", "phrases", "talkies", "harmonies", "chart"]


def test_vocals_worker_rejects_the_generic_vocals_task(tmp_path: Path) -> None:
    _catalog(tmp_path)
    generic = tmp_path / "generic-vocals.json"
    generic.write_text(json.dumps(build_catalog_task_manifest(tmp_path, "vocals")))
    with pytest.raises(VocalsTrainingError, match="Vocal activity catalog task view"):
        _read_task_view(generic, tmp_path)


def test_vocals_pipeline_exposes_strict_private_catalog_training_contract() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "vocals.note-activity/v1")
    assert descriptor.training_status == "available"
    assert descriptor.inference_capability is None
    assert descriptor.checkpoint_outputs == ("vocals.frame_activity_pitch",)
    assert descriptor.train_schema is not None
    assert "catalog_root" not in descriptor.train_schema["properties"]
    assert descriptor.prepare_schema["properties"]["split_ratios"] == {
        "type": "array",
        "items": {"type": "integer"},
    }
    phrase = next(item for item in PIPELINES if item.id == "vocals.phrase-boundaries/v1")
    assert phrase.training_status == "available"
    assert phrase.inference_capability is None
    assert phrase.checkpoint_outputs == ("vocals.phrase_boundaries",)
    assert phrase.private_request_fields == ("catalog_root",)
    planned = next(item for item in PIPELINES if item.id == "strum.instrument-chart/vocals/v1")
    assert planned.training_status == "planned"
    assert planned.inference_capability is None
    assert planned.private_request_fields == ("catalog_root",)
    contract = planned.as_json()["training_contract"]
    assert contract["format"] == "strum-planned-training-contract/v1"
    assert contract["execution"] == {
        "status": "not_available",
        "inference_capability": None,
        "required_handler": "vocal_chart_profile_handler/v1",
        "fallback": "forbidden",
    }
    assert contract["label_source"]["tracks"] == ["PART VOCALS"]
    assert contract["label_source"]["excluded_source_tracks"] == ["HARM1", "HARM2", "HARM3"]
    assert set(contract["available_experiment_components"]) == {
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
    }
    assert contract["available_source_policies"]["harmony"] == {
        "pipeline_id": "vocals.harmony-source-policy/v1",
        "task_format": "strum-vocal-harmony-source-task/v1",
        "status": "prepare_only",
        "shared_vocal_or_mix_fallback": False,
    }
    assert contract["catalog_admission"]["partition"] == {
        "unit": "source_id",
        "required_splits": ["train", "val", "test"],
        "cross_component_split_assignment": "identical-by-source-id/v1",
        "test_source_ids_forbidden_in_training_or_calibration": True,
    }
    assert contract["catalog_admission"]["harmony"] == {
        "tracks": ["HARM1", "HARM2", "HARM3"],
        "selection": "exact-declared-harmony-track/v1",
        "source_task_format": "strum-vocal-harmony-source-task/v1",
        "required_audio_policy": "isolated-harmony-stem-only/v1",
        "forbidden_audio_roles": ["vocals", "mix"],
        "required_provenance": [
            "isolated_source_stem/v1",
            "isolated_separation_output/v1",
        ],
    }
    composition = contract["composition_contract"]
    assert composition["format"] == "strum-vocal-chart-composition-contract/v1"
    assert composition["status"] == "not_available"
    assert composition["profile_graph_format"] == "strum-profile-composition/v1"
    assert [component["id"] for component in composition["required_components"]] == [
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
        "vocals.harmony_model",
    ]
    assert composition["required_components"][-1] == {
        "id": "vocals.harmony_model",
        "producer_pipeline": "not_implemented",
        "required_outputs": ["HARM1", "HARM2", "HARM3"],
        "source_policy": "vocals.harmony-source-policy/v1",
    }
    assert composition["compatibility"] == {
        "component_task_lineage": "same-catalog-control-and-source-partition/v1",
        "audio_binding": "same-catalog-audio-identity-or-pinned-alignment/v1",
        "clock": "same-master-timeline/v1",
        "lead_track": "PART VOCALS",
        "harmony_tracks_must_remain_distinct": True,
        "no_implicit_lyrics_or_harmony": True,
    }
    assert composition["forbidden_shortcuts"] == [
        "raw_component_as_profile",
        "shared_vocals_or_mix_as_harmony_supervision",
        "external_lyrics_as_chart_labels",
        "legacy_vocals_charter_fallback",
    ]
    evaluation = contract["held_out_evaluation_contract"]
    assert evaluation["format"] == "strum-vocal-held-out-chart-evaluation-contract/v1"
    assert evaluation["status"] == "not_available"
    assert evaluation["split"] == "test"
    assert evaluation["source_partition"] == "source-id-disjoint-from-train-and-val/v1"
    assert evaluation["evidence"]["recomputed_by"] == "strum"
    assert evaluation["evidence"]["quality_thresholds"] == (
        "explicit-profile-policy-not-yet-defined"
    )
    assert contract["packaging_contract"] == {
        "format": "strum-vocal-profile-package-contract/v1",
        "status": "not_available",
        "requires": [
            "completed-strum-recomputed-held-out-report",
            "hash-verified-required-components-and-configurations",
            "validated-strum-profile-composition-v1-graph",
            "registered-vocal-chart-execution-handler",
        ],
        "raw_component_bundle_deployment_status": "not_deployable",
    }
    assert contract["execution"] == {
        "status": "not_available",
        "inference_capability": None,
        "required_handler": "vocal_chart_profile_handler/v1",
        "fallback": "forbidden",
    }


def test_vocal_lyric_pipeline_packages_exact_part_vocals_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path)
    task_view = tmp_path / "views" / "vocals-lyrics.json"
    prepare = tmp_path / "prepare-lyrics.json"
    prepare.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "vocals.lyric-alignment/v1",
                "output": str(task_view),
                "options": {"split_ratios": [50, 50, 0]},
            }
        )
    )
    prepare_dataset_request(prepare)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "vocals_lyric_alignment"
    assert prepared["task"]["label_schema"]["track_names"] == ["PART VOCALS"]
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_vocal_lyric_alignment.py"):
            checkpoints = Path(command[command.index("--checkpoint-dir") + 1])
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "best.pt").write_bytes(b"vocal-lyric-weights")
            (checkpoints / "history.json").write_text(
                json.dumps([{"epoch": 1, "train_ctc_loss": 0.5, "val_ctc_loss": 0.4}])
            )

    monkeypatch.setattr("src.vocals_lyric_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "vocal-lyrics-v1"
    train = tmp_path / "train-lyrics.json"
    train.write_text(
        json.dumps(
            {
                "pipeline_id": "vocals.lyric-alignment/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": "catalog-vocal-lyrics-v1", "epochs": 1, "device": "cpu"},
            }
        )
    )
    result = run_training_request(train)
    assert (
        result["deployment_status"] == "requires_vocal_chart_composition_evaluation_and_packaging"
    )
    assert [component["id"] for component in result["components"]] == ["vocals.lyric_alignment"]
    assert "preprocess_vocal_lyric_alignment.py" in commands[0][1]
    assert "train_vocal_lyric_alignment.py" in commands[1][1]
    bundle = output / "bundle"
    assert str(tmp_path) not in (bundle / "strum-model-bundle.json").read_text()
    assert str(tmp_path) not in (output / "experiment.json").read_text()
    model = load_model_bundle(bundle, check_files=True)
    assert set(model.components) == {"vocals.lyric_alignment"}
    assert model.profiles == {}
    config = json.loads((bundle / "configs" / "vocals-lyric-alignment.json").read_text())
    assert config["excluded_outputs"] == ["pitch", "phrases", "talkies", "harmonies", "chart"]
    assert config["tokenizer"]["vocabulary"][0] == "<blank>"


def test_vocal_lyric_preprocessor_uses_observed_part_vocals_meta_events(tmp_path: Path) -> None:
    _catalog(tmp_path, materialize=True)
    task_view = tmp_path / "vocals-lyrics.json"
    task_view.write_text(
        json.dumps(build_catalog_task_manifest(tmp_path, "vocals_lyric_alignment"))
    )
    summary = prepare_vocal_lyric_alignment(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "lyric-cache",
        splits=("train", "val"),
        limit_songs=1,
    )
    assert summary["tokenizer"]["vocabulary_size"] > 2
    assert summary["splits"]["train"]["song_count"] == 1
    targets = np.load(tmp_path / "lyric-cache" / "train_targets.npy")
    assert targets[0, :5].tolist() == tokenize_lyric("hello")
    assert TOKEN_TO_ID["<blank>"] == 0


def test_vocal_lyric_trainer_runs_one_cpu_batch_from_catalog_targets(tmp_path: Path) -> None:
    _catalog(tmp_path, materialize=True)
    task_view = tmp_path / "vocals-lyrics.json"
    task_view.write_text(
        json.dumps(build_catalog_task_manifest(tmp_path, "vocals_lyric_alignment"))
    )
    prepare_vocal_lyric_alignment(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "lyric-cache",
        splits=("train", "val"),
        limit_songs=1,
    )
    metrics = train_vocal_lyric_alignment(
        LyricTrainingSettings(
            cache_dir=tmp_path / "lyric-cache",
            checkpoint_dir=tmp_path / "lyric-checkpoints",
            epochs=1,
            batch_size=1,
            learning_rate=0.0003,
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            seed=7,
        )
    )
    assert set(metrics) == {"train_ctc_loss", "val_ctc_loss"}
    assert (tmp_path / "lyric-checkpoints" / "best.pt").is_file()


def test_vocal_preprocessor_reads_part_vocals_not_guitar(tmp_path: Path) -> None:
    path = tmp_path / "notes.mid"
    path.write_bytes(_vocal_midi())
    events = parse_vocal_events(path)
    assert events["notes"] == [{"start": 0.0, "end": 0.5, "pitch": 60}]
    assert events["lyric_event_count"] == 1
    assert events["lyric_events"] == [{"time": 0.0, "text": "hello", "message_type": "lyrics"}]
    assert events["phrase_marker_count"] == 2
    assert events["phrase_start_events"] == [0.0]
    assert events["phrase_end_events"] == [0.5]
    assert events["talky_spans"] == [{"start": 0.5, "end": 0.75}]
    with pytest.raises(VocalsPreprocessError, match="PART VOCALS"):
        parse_vocal_events(path, label_track="HARM1")


def test_vocal_preprocessor_materializes_activity_and_pitch_from_catalog(tmp_path: Path) -> None:
    _catalog(tmp_path, materialize=True)
    task_view = tmp_path / "vocals.json"
    task_view.write_text(json.dumps(build_catalog_task_manifest(tmp_path, "vocals_activity")))
    summary = prepare_vocals_frames(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "cache",
        splits=("train", "val"),
        limit_songs=1,
    )
    assert summary["splits"]["train"]["song_count"] == 1
    assert summary["splits"]["val"]["song_count"] == 1
    pitch = np.load(tmp_path / "cache" / "train_pitch.npy")
    assert pitch.max() == 60 - 36 + 1
    assert pitch.max() < PITCH_CLASS_COUNT


def test_vocal_phrase_worker_packages_a_catalog_backed_experiment_without_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path)
    task_view = tmp_path / "views" / "vocals-phrases.json"
    prepare = tmp_path / "prepare-phrases.json"
    prepare.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "vocals.phrase-boundaries/v1",
                "output": str(task_view),
                "options": {"split_ratios": [50, 50, 0]},
            }
        )
    )
    prepare_dataset_request(prepare)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "vocals_phrase_boundaries"
    assert prepared["task"]["label_schema"]["difficulty_encoding"] == (
        "vocal-phrase-boundary-events/v1"
    )
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_vocal_phrase_boundaries.py"):
            checkpoints = Path(command[command.index("--checkpoint-dir") + 1])
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "best.pt").write_bytes(b"vocal-phrase-weights")
            (checkpoints / "history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "train_loss": 0.5,
                            "val_loss": 0.4,
                            "val_phrase_start_f1": 0.8,
                            "val_phrase_end_f1": 0.7,
                        }
                    ]
                )
            )

    monkeypatch.setattr("src.vocals_phrase_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "vocal-phrases-v1"
    train = tmp_path / "train-phrases.json"
    train.write_text(
        json.dumps(
            {
                "pipeline_id": "vocals.phrase-boundaries/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": "catalog-vocal-phrases-v1", "epochs": 1, "device": "cpu"},
            }
        )
    )
    result = run_training_request(train)

    assert result["status"] == "completed"
    assert (
        result["deployment_status"] == "requires_vocal_chart_composition_evaluation_and_packaging"
    )
    assert result["metrics"] == {
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_phrase_start_f1": 0.8,
        "val_phrase_end_f1": 0.7,
    }
    assert [component["id"] for component in result["components"]] == ["vocals.phrase_boundaries"]
    assert "preprocess_vocal_phrase_boundaries.py" in commands[0][1]
    assert "train_vocal_phrase_boundaries.py" in commands[1][1]
    bundle = output / "bundle"
    assert str(tmp_path) not in (bundle / "strum-model-bundle.json").read_text()
    assert str(tmp_path) not in (output / "experiment.json").read_text()
    model = load_model_bundle(bundle, check_files=True)
    assert set(model.components) == {"vocals.phrase_boundaries"}
    assert model.profiles == {}
    config = json.loads((bundle / "configs" / "vocals-phrase-boundaries.json").read_text())
    assert config["excluded_outputs"] == ["lyrics", "talkies", "harmonies", "chart"]


def test_vocal_phrase_preprocessor_materializes_observed_boundaries_from_catalog(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path, materialize=True)
    task_view = tmp_path / "vocals-phrases.json"
    task_view.write_text(
        json.dumps(build_catalog_task_manifest(tmp_path, "vocals_phrase_boundaries"))
    )
    summary = prepare_vocal_phrase_boundaries(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "phrase-cache",
        splits=("train", "val"),
        limit_songs=1,
    )
    assert summary["splits"]["train"]["song_count"] == 1
    assert summary["splits"]["val"]["song_count"] == 1
    starts = np.load(tmp_path / "phrase-cache" / "train_phrase_start.npy")
    ends = np.load(tmp_path / "phrase-cache" / "train_phrase_end.npy")
    assert starts.max() == 1
    assert ends.max() == 1


def test_vocal_talky_worker_packages_exact_note_96_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog(tmp_path)
    task_view = tmp_path / "views" / "vocals-talkies.json"
    prepare = tmp_path / "prepare-talkies.json"
    prepare.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "vocals.talky-activity/v1",
                "output": str(task_view),
                "options": {"split_ratios": [50, 50, 0]},
            }
        )
    )
    prepare_dataset_request(prepare)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == "vocals_talky_activity"
    assert prepared["task"]["label_schema"]["difficulty_encoding"] == (
        "vocal-pitchless-talky-note-96-spans/v1"
    )
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_vocal_talky_activity.py"):
            checkpoints = Path(command[command.index("--checkpoint-dir") + 1])
            checkpoints.mkdir(parents=True, exist_ok=True)
            (checkpoints / "best.pt").write_bytes(b"vocal-talky-weights")
            (checkpoints / "history.json").write_text(
                json.dumps(
                    [
                        {
                            "epoch": 1,
                            "train_loss": 0.5,
                            "val_loss": 0.4,
                            "val_talky_activity_f1": 0.7,
                        }
                    ]
                )
            )

    monkeypatch.setattr("src.vocals_talky_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "vocal-talky-v1"
    train = tmp_path / "train-talkies.json"
    train.write_text(
        json.dumps(
            {
                "pipeline_id": "vocals.talky-activity/v1",
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": "catalog-vocal-talky-v1", "epochs": 1, "device": "cpu"},
            }
        )
    )
    result = run_training_request(train)
    assert result["metrics"] == {
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_talky_activity_f1": 0.7,
    }
    assert (
        result["deployment_status"] == "requires_vocal_chart_composition_evaluation_and_packaging"
    )
    assert [component["id"] for component in result["components"]] == ["vocals.talky_activity"]
    assert "preprocess_vocal_talky_activity.py" in commands[0][1]
    assert "train_vocal_talky_activity.py" in commands[1][1]
    bundle = output / "bundle"
    assert str(tmp_path) not in (bundle / "strum-model-bundle.json").read_text()
    assert str(tmp_path) not in (output / "experiment.json").read_text()
    model = load_model_bundle(bundle, check_files=True)
    assert set(model.components) == {"vocals.talky_activity"}
    assert model.profiles == {}
    config = json.loads((bundle / "configs" / "vocals-talky-activity.json").read_text())
    assert config["outputs"] == ["pitchless_talky_activity"]
    assert config["excluded_outputs"] == ["pitch", "lyrics", "phrases", "harmonies", "chart"]


def test_vocal_talky_preprocessor_and_trainer_require_observed_note_96_targets(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path, materialize=True)
    task_view = tmp_path / "vocals-talkies.json"
    task_view.write_text(json.dumps(build_catalog_task_manifest(tmp_path, "vocals_talky_activity")))
    summary = prepare_vocal_talky_activity(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "talky-cache",
        splits=("train", "val"),
        limit_songs=1,
    )
    assert summary["splits"]["train"]["talky_span_count"] == 1
    assert summary["splits"]["val"]["talky_span_count"] == 1
    labels = np.load(tmp_path / "talky-cache" / "train_talky.npy")
    assert labels.max() == 1
    metrics = train_vocal_talky_activity(
        TalkyTrainingSettings(
            cache_dir=tmp_path / "talky-cache",
            checkpoint_dir=tmp_path / "talky-checkpoints",
            epochs=1,
            batch_size=1,
            learning_rate=0.0003,
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            seed=7,
        )
    )
    assert set(metrics) == {"train_loss", "val_loss", "val_talky_activity_f1"}
    assert (tmp_path / "talky-checkpoints" / "best.pt").is_file()
    np.save(tmp_path / "talky-cache" / "val_talky.npy", np.zeros_like(labels))
    with pytest.raises(ValueError, match="observed note-96 spans in both train and val"):
        train_vocal_talky_activity(
            TalkyTrainingSettings(
                cache_dir=tmp_path / "talky-cache",
                checkpoint_dir=tmp_path / "talky-negative-checkpoints",
                epochs=1,
                batch_size=1,
                learning_rate=0.0003,
                device="cpu",
                max_train_batches=1,
                max_val_batches=1,
                seed=7,
            )
        )
