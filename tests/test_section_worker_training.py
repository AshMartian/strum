from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import mido
import numpy as np
import pytest

from src.catalog_task_manifest import build_catalog_task_manifest
from src.model_bundle import load_model_bundle
from src.section_worker_training import SectionTrainingError, _read_task_view
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


def _five_lane_midi(*track_names: str) -> bytes:
    midi = mido.MidiFile()
    for track_name in track_names:
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("track_name", name=track_name, time=0))
        track.append(mido.Message("note_on", note=96, velocity=100, time=0))
        track.append(mido.Message("note_off", note=96, velocity=0, time=480))
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _catalog(root: Path) -> None:
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
                "metadata": {"name": f"Section fixture {index}"},
                "chart": {
                    "notes_midi": _asset(
                        root,
                        _five_lane_midi(
                            "PART GUITAR",
                            "PART GUITAR ALT",
                            "PART BASS",
                            "PART BASS ALT",
                        ),
                        f"notes-{index}.mid",
                    ),
                    "instruments": {
                        "guitar": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART GUITAR", "PART GUITAR ALT"],
                        },
                        "bass": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART BASS", "PART BASS ALT"],
                        },
                    },
                },
                "audio": {
                    "guitar": _asset(root, f"guitar-{index}".encode(), f"guitar-{index}.wav"),
                    "bass": _asset(root, f"bass-{index}".encode(), f"bass-{index}.wav"),
                },
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "section-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def _fake_section_scripts(commands: list[list[str]]) -> object:
    def fake_run(command: list[str]) -> None:
        commands.append(command)
        script = Path(command[1]).name
        if script == "build_catalog_section_labels.py":
            manifest = json.loads(Path(command[command.index("--manifest") + 1]).read_text())
            out = Path(command[command.index("--out") + 1])
            records = [
                {
                    "source_id": song["source_id"],
                    "split": song["split"],
                    "t_start_s": 0.0,
                    "t_end_s": 2.0,
                    "label": "single_notes",
                }
                for song in manifest["songs"]
            ]
            out.write_text(
                json.dumps(
                    {
                        "format": "strum-section-labels/v1",
                        "lineage": {"pipeline_id": manifest["task"]["pipeline_id"]},
                        "records": records,
                    }
                )
            )
        elif script == "preprocess_section_windows.py":
            labels = json.loads(Path(command[command.index("--labels") + 1]).read_text())
            cache_dir = Path(command[command.index("--cache-dir") + 1])
            cache_dir.mkdir(parents=True, exist_ok=True)
            for split in ("train", "val"):
                records = [record for record in labels["records"] if record["split"] == split]
                np.save(
                    cache_dir / f"{split}_section_mel.npy",
                    np.zeros((len(records), 128, 87), np.float32),
                )
                np.save(cache_dir / f"{split}_section_label.npy", np.full(len(records), 4, np.int8))
                (cache_dir / f"{split}_section_meta.json").write_text(
                    json.dumps(
                        [
                            {
                                "source_id": record["source_id"],
                                "t_start_s": record["t_start_s"],
                                "label": record["label"],
                            }
                            for record in records
                        ]
                    )
                )
        elif script == "train_section_classifier.py":
            checkpoint_dir = Path(command[command.index("--ckpt-dir") + 1])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "best.pt").write_bytes(b"section-weights")
            metrics = Path(command[command.index("--metrics-out") + 1])
            metrics.write_text(
                json.dumps(
                    {
                        "best_val_accuracy": 0.8,
                        "best_per_class_accuracy": {"single_notes": 0.8},
                        "last_train_loss": 0.2,
                        "train_record_count": 25,
                        "val_record_count": 4,
                        "device": "cpu",
                    }
                )
            )

    return fake_run


@pytest.mark.parametrize(
    ("pipeline_id", "task_kind", "component"),
    [
        ("strum.section-classifier/guitar/v1", "section_guitar", "section_classifier.guitar"),
        ("strum.section-classifier/bass/v1", "section_bass", "section_classifier.bass"),
    ],
)
def test_section_worker_packages_a_revalidated_catalog_experiment_without_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pipeline_id: str,
    task_kind: str,
    component: str,
) -> None:
    _catalog(tmp_path)
    task_view = tmp_path / "views" / f"{task_kind}.json"
    prepare = tmp_path / "prepare.json"
    prepare.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": pipeline_id,
                "output": str(task_view),
                "options": {"split_ratios": [80, 20, 0]},
            }
        )
    )
    prepare_dataset_request(prepare)
    prepared = json.loads(task_view.read_text())
    assert prepared["task"]["kind"] == task_kind
    label_track = "PART GUITAR" if task_kind == "section_guitar" else "PART BASS"
    assert prepared["task"]["label_schema"]["track_names"] == [label_track]
    assert {tuple(song["label_tracks"]) for song in prepared["songs"]} == {(label_track,)}
    assert str(tmp_path) not in json.dumps(prepared)

    commands: list[list[str]] = []
    monkeypatch.setattr("src.section_worker_training._run_script", _fake_section_scripts(commands))
    output = tmp_path / "experiments" / task_kind
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps(
            {
                "pipeline_id": pipeline_id,
                "task_view": str(task_view),
                "output": str(output),
                "catalog_root": str(tmp_path),
                "options": {"model_id": f"catalog-{task_kind}", "epochs": 1, "device": "cpu"},
            }
        )
    )
    result = run_training_request(train)

    assert result["status"] == "completed"
    assert result["deployment_status"] == "requires_section_profile_evaluation"
    assert [item[1].rsplit("/", 1)[-1] for item in commands] == [
        "build_catalog_section_labels.py",
        "preprocess_section_windows.py",
        "train_section_classifier.py",
    ]
    assert [item["id"] for item in result["components"]] == [component]
    bundle = output / "bundle"
    manifest = (bundle / "strum-model-bundle.json").read_text()
    experiment = (output / "experiment.json").read_text()
    assert str(tmp_path) not in manifest
    assert str(tmp_path) not in experiment
    model = load_model_bundle(bundle, check_files=True)
    assert set(model.components) == {component}
    assert model.profiles == {}
    config = json.loads(next((bundle / "configs").glob("*.json")).read_text())
    packaged_experiment = json.loads((output / "experiment.json").read_text())
    assert config["instrument"] == task_kind.removeprefix("section_")
    assert config["format"] == "strum-section-classifier-model-config/v1"
    assert config["preprocessing"] == "section-logmel-librosa-router-windows/v1"
    assert config["feature_extractor"]["backend"] == "librosa"
    assert config["task_view_sha256"] == hashlib.sha256(task_view.read_bytes()).hexdigest()
    assert packaged_experiment["task_view"]["sha256"] == config["task_view_sha256"]
    assert config["runtime_profile"] == {
        "format": "strum-section-router-deployment-requirements/v1",
        "status": "not_packageable",
        "reason": "section_router_execution_and_held_out_evaluation_not_proven",
        "requirements": [
            "section_router_profile_loader_tensor_only",
            "held_out_section_calibration_evaluation",
            "held_out_chart_impact_ablation",
            f"composed_{task_kind.removeprefix('section_')}_chart_profile_contract",
        ],
    }
    assert packaged_experiment["held_out_evaluation"] == {
        "status": "not_run",
        "reason": "trainer_did_not_report_held_out_evaluation",
    }


def test_section_worker_rejects_wrong_task_or_label_schema(tmp_path: Path) -> None:
    _catalog(tmp_path)
    wrong = build_catalog_task_manifest(tmp_path, "section_bass")
    path = tmp_path / "wrong.json"
    path.write_text(json.dumps(wrong))
    with pytest.raises(SectionTrainingError, match="exact catalog task view"):
        _read_task_view(path, tmp_path, "strum.section-classifier/guitar/v1")

    wrong = build_catalog_task_manifest(tmp_path, "section_guitar")
    wrong["task"]["label_schema"]["track_names"] = ["PART BASS"]
    path.write_text(json.dumps(wrong))
    with pytest.raises(SectionTrainingError, match="exact catalog task view"):
        _read_task_view(path, tmp_path, "strum.section-classifier/guitar/v1")


def test_section_worker_requires_reprepare_for_retired_prefix_view(tmp_path: Path) -> None:
    _catalog(tmp_path)
    retired = build_catalog_task_manifest(tmp_path, "section_guitar")
    retired["task"]["label_schema"] = {
        "id": "midi-section-events/v1",
        "track_prefixes": ["PART GUITAR"],
        "difficulty_encoding": "not-applicable",
    }
    path = tmp_path / "retired.json"
    path.write_text(json.dumps(retired))

    with pytest.raises(SectionTrainingError, match="retired prefix.*re-prepare"):
        _read_task_view(path, tmp_path, "strum.section-classifier/guitar/v1")


@pytest.mark.parametrize(
    ("pipeline_id", "component"),
    [
        ("strum.section-classifier/guitar/v1", "section_classifier.guitar"),
        ("strum.section-classifier/bass/v1", "section_classifier.bass"),
    ],
)
def test_section_descriptors_expose_private_catalog_training_without_inference(
    pipeline_id: str, component: str
) -> None:
    descriptor = next(item for item in PIPELINES if item.id == pipeline_id)
    assert descriptor.training_status == "available"
    assert descriptor.inference_capability is None
    assert descriptor.checkpoint_outputs == (component,)
    assert descriptor.private_request_fields == ("catalog_root",)
    assert descriptor.catalog_inspection_option_keys == (
        "audio_role",
        "fallback_audio_role",
        "disable_fallback",
        "required_difficulty",
    )
    assert descriptor.train_schema is not None
    assert "catalog_root" not in descriptor.train_schema["properties"]
    expected = (
        "section_router_profile_loader_tensor_only",
        "held_out_section_calibration_evaluation",
        "held_out_chart_impact_ablation",
        "composed_guitar_chart_profile_contract"
        if pipeline_id.endswith("guitar/v1")
        else "composed_bass_chart_profile_contract",
    )
    assert descriptor.training_requirements == expected
