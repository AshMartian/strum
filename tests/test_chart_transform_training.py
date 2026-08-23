import hashlib
import json
import math
import shutil
import wave
from array import array
from pathlib import Path

import pytest
import torch

from scripts.infer_chart_transform import load_source_events, predict
from scripts.train_chart_transform import (
    ChartEvent,
    DatasetValidationError,
    TrainingConfig,
    _parse_events,
    train,
)
from src.chart_transform_profile import (
    ChartTransformPromotionError,
    evaluate_chart_transform_candidate,
    package_chart_transform_profile,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.models.chart_audio import AudioFeatureError, event_audio_features
from src.models.chart_transform import EventTransformMLP
from src.worker import inspect_model_bundle, preflight_chart_request


def _unsafe_checkpoint_reducer() -> None:
    raise AssertionError("unsafe checkpoint payload was deserialized")


class _UnsafeCheckpointPayload:
    def __reduce__(self) -> tuple[object, tuple[object, ...]]:
        return _unsafe_checkpoint_reducer, ()


def _write_test_song(path: Path, frequency_hz: float) -> None:
    sample_rate = 16_000
    samples = array(
        "h",
        (
            round(12_000 * math.sin(2 * math.pi * frequency_hz * index / sample_rate))
            for index in range(sample_rate)
        ),
    )
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _catalog_task_dataset(path: Path) -> Path:
    """Create the smallest genuine, hash-bound catalog task-view fixture."""
    records = [
        {
            "song_id": "train-song",
            "source_id": "train-song",
            "notes_midi_sha256": "a" * 64,
            "split": "train",
            "instrument": "guitar",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}],
        },
        {
            "song_id": "held-out-song",
            "source_id": "held-out-song",
            "notes_midi_sha256": "b" * 64,
            "split": "validation",
            "instrument": "guitar",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [1]}],
            "target_events": [{"time_ms": 0, "lanes": [1]}],
        },
    ]
    task_view = {
        "pipeline": {"id": "chart_transform.five_lane", "version": 1},
        "catalog": {
            "catalog_id": "promotion-fixture-catalog",
            "manifest_sha256": "c" * 64,
            "records_sha256": "d" * 64,
        },
        "source_inputs": [
            {"source_id": item["source_id"], "notes_midi_sha256": item["notes_midi_sha256"]}
            for item in records
        ],
        "split": {
            "algorithm": "sha256-source-id-rank/v1",
            "seed": 7,
            "validation_fraction": 0.5,
            "assignments": {item["source_id"]: item["split"] for item in records},
        },
        "preprocessing": {"config_sha256": "e" * 64},
    }
    task_view["task_view_id"] = hashlib.sha256(
        json.dumps(task_view, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    (path / "pairs.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n")
    manifest = path / "dataset-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "promotion-fixture",
                "records": "pairs.jsonl",
                "provenance": "synthetic test fixture",
                "license": "test-only",
                "instrument": "guitar",
                "task_view": task_view,
            }
        )
    )
    return manifest


def test_transform_requires_held_out_evaluation_and_immutable_promotion(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    manifest = _catalog_task_dataset(dataset)
    candidate = tmp_path / "candidate"
    train(
        TrainingConfig(
            dataset_manifest=str(manifest),
            output_dir=str(candidate),
            model_id="promotion-fixture",
            source_difficulty="Expert",
            target_difficulty="Hard",
            hidden_dim=4,
            epochs=1,
            device="cpu",
        )
    )
    candidate_config = json.loads((candidate / "configs" / "training-config.json").read_text())
    assert candidate_config["lineage"]["dataset"]["dataset_id"] == "promotion-fixture"
    assert candidate_config["lineage"]["task_view"]["task_view_id"]
    assert candidate_config["lineage"]["split"]["validation_song_ids"] == ["held-out-song"]
    assert str(dataset) not in json.dumps(candidate_config["lineage"])
    assert inspect_model_bundle(candidate)["deployment_status"] == "not_deployable"
    report = tmp_path / "held-out.json"
    result = evaluate_chart_transform_candidate(
        bundle_root=candidate, dataset_manifest=manifest, output_path=report
    )
    assert result["split"] == "validation"
    assert result["records_evaluated"] == 1
    promoted = tmp_path / "promoted"
    packaged = package_chart_transform_profile(
        experiment_dir=candidate,
        evaluation_path=report,
        dataset_manifest=manifest,
        output_dir=promoted,
        profile_id="difficulty-transform-guitar-promoted",
    )
    assert packaged["status"] == "promoted"
    assert inspect_model_bundle(promoted)["deployment_status"] == "ready"
    request = tmp_path / "preflight.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(promoted),
                "profile_id": "difficulty-transform-guitar-promoted",
                "difficulty_policy": "learned:chart_transform.guitar.expert_to_hard",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )
    assert preflight_chart_request(request)["execution"] == "available"
    report_data = json.loads(report.read_text())
    report_data["metrics"]["loss"] += 1.0
    report.write_text(json.dumps(report_data))
    with pytest.raises(ChartTransformPromotionError, match="independently recomputed"):
        package_chart_transform_profile(
            experiment_dir=candidate,
            evaluation_path=report,
            dataset_manifest=manifest,
            output_dir=tmp_path / "altered-metrics",
            profile_id="altered-metrics",
        )
    report_data = json.loads((promoted / "evaluations" / "held-out.json").read_text())
    report_data["held_out_song_ids"] = ["different-song"]
    report.write_text(json.dumps(report_data))
    with pytest.raises(ChartTransformPromotionError, match="invalid"):
        package_chart_transform_profile(
            experiment_dir=candidate,
            evaluation_path=report,
            dataset_manifest=manifest,
            output_dir=tmp_path / "altered-held-out",
            profile_id="altered-held-out",
        )
    report_data = json.loads((promoted / "evaluations" / "held-out.json").read_text())
    report_data["dataset_manifest_sha256"] = "0" * 64
    report.write_text(json.dumps(report_data))
    with pytest.raises(ChartTransformPromotionError, match="invalid"):
        package_chart_transform_profile(
            experiment_dir=candidate,
            evaluation_path=report,
            dataset_manifest=manifest,
            output_dir=tmp_path / "altered-dataset-lineage",
            profile_id="altered-dataset-lineage",
        )
    report_data = json.loads((promoted / "evaluations" / "held-out.json").read_text())
    report_data.pop("candidate_lineage_sha256")
    report.write_text(json.dumps(report_data))
    with pytest.raises(ChartTransformPromotionError, match="invalid"):
        package_chart_transform_profile(
            experiment_dir=candidate,
            evaluation_path=report,
            dataset_manifest=manifest,
            output_dir=tmp_path / "missing-evidence",
            profile_id="missing-evidence",
        )
    report.write_text((promoted / "evaluations" / "held-out.json").read_text())
    original_manifest = manifest.read_text()
    altered_manifest = json.loads(original_manifest)
    altered_manifest["provenance"] = "altered after candidate training"
    manifest.write_text(json.dumps(altered_manifest))
    with pytest.raises(ChartTransformPromotionError, match="does not match candidate lineage"):
        package_chart_transform_profile(
            experiment_dir=candidate,
            evaluation_path=report,
            dataset_manifest=manifest,
            output_dir=tmp_path / "altered-input-lineage",
            profile_id="altered-input-lineage",
        )
    manifest.write_text(original_manifest)
    forged = tmp_path / "manually-fabricated-profile"
    shutil.copytree(promoted, forged)
    forged_profile = forged / "profiles" / "difficulty-transform-guitar-promoted.json"
    forged_config = json.loads(forged_profile.read_text())
    forged_config.pop("lineage")
    forged_profile.write_text(json.dumps(forged_config))
    forged_manifest_path = forged / MANIFEST_FILENAME
    forged_manifest = json.loads(forged_manifest_path.read_text())
    profile_entry = forged_manifest["profiles"]["difficulty-transform-guitar-promoted"]
    profile_entry["configuration_sha256"] = hashlib.sha256(forged_profile.read_bytes()).hexdigest()
    profile_entry["configuration_byte_length"] = forged_profile.stat().st_size
    forged_manifest_path.write_text(json.dumps(forged_manifest))
    request.write_text(
        json.dumps(
            {
                "model_root": str(forged),
                "profile_id": "difficulty-transform-guitar-promoted",
                "difficulty_policy": "learned:chart_transform.guitar.expert_to_hard",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )
    with pytest.raises(BundleValidationError, match="promotion configuration"):
        preflight_chart_request(request)


@pytest.mark.parametrize("time_ms", [-1, float("nan"), float("inf")])
def test_chart_event_parsers_reject_invalid_timestamps(tmp_path: Path, time_ms: float) -> None:
    with pytest.raises(DatasetValidationError, match="numeric time_ms"):
        _parse_events([{"time_ms": time_ms, "lanes": [0]}], 5, 1, "source_events")

    source_events = tmp_path / "events.json"
    source_events.write_text(json.dumps({"source_events": [{"time_ms": time_ms, "lanes": [0]}]}))
    with pytest.raises(DatasetValidationError, match="numeric time_ms"):
        load_source_events(source_events, 5)


def test_train_revalidates_audio_overrides_before_reading_the_dataset(tmp_path: Path) -> None:
    config = TrainingConfig(
        dataset_manifest=str(tmp_path / "not-read.json"),
        output_dir=str(tmp_path / "output"),
        model_id="invalid-audio-config",
        source_difficulty="Expert",
        target_difficulty="Hard",
        audio_manifest=str(tmp_path / "audio-manifest.json"),
    )
    with pytest.raises(DatasetValidationError, match="audio_manifest requires"):
        train(config)


def test_cpu_chart_pair_training_writes_valid_model_bundle(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "instrument": "bass",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1, 2]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}, {"time_ms": 500, "lanes": [1]}],
        },
        {
            "song_id": "song-b",
            "instrument": "bass",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [3]}, {"time_ms": 400, "lanes": [4]}],
            "target_events": [{"time_ms": 0, "lanes": [3]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "local-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic test fixture; replace with documented local chart provenance",
                "license": "test-only",
                "instrument": "bass",
            }
        )
    )
    output_dir = tmp_path / "bundle"
    result = train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            seed=7,
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            strum_revision="test-revision",
        )
    )

    bundle = load_model_bundle(output_dir, check_files=True)
    metadata = json.loads((output_dir / "training-metadata.json").read_text())
    experiment = json.loads((output_dir / "experiment.json").read_text())

    assert bundle.model_id == "test-expert-to-hard"
    component_name = "chart_transform.bass.expert_to_hard"
    component = bundle.component(component_name)
    assert component is not None
    assert component.byte_length and component.architecture == "EventTransformMLP/v1"
    assert component.config_sha256 and component.config_byte_length
    assert json.loads(component.config.read_text())["instrument"] == "bass"
    assert bundle.profiles == {}
    assert bundle.validate(check_files=True, verify_hashes=True) == []
    assert metadata["dataset"]["provenance"].startswith("synthetic")
    assert metadata["dataset"]["license"] == "test-only"
    assert metadata["dataset"]["instrument"] == "bass"
    assert experiment["format"] == "strum-experiment/v1"
    assert experiment["pipeline"] == {"id": "chart_transform.five_lane", "version": 1}
    assert experiment["checkpoint_mode"] == "fresh"
    assert experiment["deployment_status"] == "requires_transform_profile_evaluation_and_promotion"
    assert experiment["model_bundle"]["manifest_sha256"]
    assert str(dataset_dir) not in json.dumps(experiment)
    assert set(metadata["split"]["train_song_ids"]).isdisjoint(
        metadata["split"]["validation_song_ids"]
    )
    assert result["metrics"]["validation"]["loss"] >= 0

    fine_tune_output = tmp_path / "fine-tuned-bundle"
    fine_tuned = train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(fine_tune_output),
            model_id="test-expert-to-hard-fine-tuned",
            source_difficulty="Expert",
            target_difficulty="Hard",
            seed=7,
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            init_checkpoint=str(output_dir / "weights" / "chart_transform.pt"),
        )
    )
    fine_tuned_metadata = json.loads((fine_tune_output / "training-metadata.json").read_text())

    assert fine_tuned["metrics"]["validation"]["loss"] >= 0
    assert "checkpoint_sha256" in fine_tuned_metadata["initialization"]

    unsafe_checkpoint = tmp_path / "unsafe-parent.pt"
    torch.save({"payload": _UnsafeCheckpointPayload()}, unsafe_checkpoint)
    with pytest.raises(DatasetValidationError, match="safe tensor-only checkpoint"):
        train(
            TrainingConfig(
                dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
                output_dir=str(tmp_path / "unsafe-output"),
                model_id="unsafe-test",
                source_difficulty="Expert",
                target_difficulty="Hard",
                validation_fraction=0.5,
                hidden_dim=4,
                epochs=1,
                device="cpu",
                init_checkpoint=str(unsafe_checkpoint),
            )
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is unavailable")
def test_audio_conditioned_training_and_inference_keep_song_paths_local(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 100, "lanes": [0]}, {"time_ms": 500, "lanes": [1]}],
            "target_events": [{"time_ms": 100, "lanes": [0]}],
        },
        {
            "song_id": "song-b",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 200, "lanes": [2]}, {"time_ms": 600, "lanes": [3]}],
            "target_events": [{"time_ms": 200, "lanes": [2]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "audio-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic audio fixture",
                "license": "test-only",
            }
        )
    )
    _write_test_song(dataset_dir / "song-a.wav", 220.0)
    _write_test_song(dataset_dir / "song-b.wav", 440.0)
    audio_manifest = dataset_dir / "audio-manifest.json"
    audio_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-local-audio-assets/v1",
                "assets": [
                    {"song_id": "song-a", "audio": "song-a.wav"},
                    {"song_id": "song-b", "audio": "song-b.wav"},
                ],
            }
        )
    )
    output_dir = tmp_path / "audio-bundle"
    train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="audio-test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cpu",
            audio_feature_mode="rms_onset_v1",
            audio_manifest=str(audio_manifest),
        )
    )

    checkpoint_path = output_dir / "weights" / "chart_transform.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = json.loads((output_dir / "training-metadata.json").read_text())
    portable_config = (output_dir / "configs" / "training-config.json").read_text()
    source_events = (ChartEvent(100.0, (0,)), ChartEvent(500.0, (1,)))

    assert checkpoint["audio_feature_dim"] == 2
    assert checkpoint["instrument"] == "guitar"
    assert metadata["audio_conditioning"]["audio_manifest_sha256"]
    assert str(audio_manifest) not in portable_config
    with pytest.raises(ValueError, match="requires --song"):
        predict(checkpoint_path, source_events, song_path=None, device_name="cpu", threshold=0.5)
    assert isinstance(
        predict(
            checkpoint_path,
            source_events,
            song_path=dataset_dir / "song-a.wav",
            device_name="cpu",
            threshold=0.5,
        ),
        list,
    )
    with pytest.raises(AudioFeatureError, match="duration limit"):
        event_audio_features(
            dataset_dir / "song-a.wav",
            [1_201_000.0],
            sample_rate=16_000,
            window_ms=50.0,
            max_duration_seconds=1_200.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_chart_pair_training_uses_cuda_and_saves_portable_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    records = [
        {
            "song_id": "song-a",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [0]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}],
        },
        {
            "song_id": "song-b",
            "source_difficulty": "Expert",
            "target_difficulty": "Hard",
            "source_events": [{"time_ms": 0, "lanes": [1]}],
            "target_events": [{"time_ms": 0, "lanes": [1]}],
        },
    ]
    (dataset_dir / "pairs.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n"
    )
    (dataset_dir / "dataset-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "cuda-test-pairs",
                "records": "pairs.jsonl",
                "provenance": "synthetic CUDA test fixture",
                "license": "test-only",
            }
        )
    )
    output_dir = tmp_path / "bundle"
    observed_devices: list[tuple[torch.device, torch.device]] = []
    original_forward = EventTransformMLP.forward

    def record_forward(self: EventTransformMLP, features: torch.Tensor) -> torch.Tensor:
        observed_devices.append((next(self.parameters()).device, features.device))
        return original_forward(self, features)

    monkeypatch.setattr(EventTransformMLP, "forward", record_forward)

    train(
        TrainingConfig(
            dataset_manifest=str(dataset_dir / "dataset-manifest.json"),
            output_dir=str(output_dir),
            model_id="cuda-test-expert-to-hard",
            source_difficulty="Expert",
            target_difficulty="Hard",
            validation_fraction=0.5,
            hidden_dim=4,
            epochs=1,
            device="cuda:0",
        )
    )

    metadata = json.loads((output_dir / "training-metadata.json").read_text())
    checkpoint = torch.load(
        output_dir / "weights" / "chart_transform.pt", map_location="cpu", weights_only=False
    )

    assert metadata["runtime"]["device"]["requested"] == "cuda:0"
    assert metadata["runtime"]["device"]["resolved"] == "cuda:0"
    assert metadata["runtime"]["device"]["cuda_device_name"]
    assert observed_devices
    assert all(
        model_device.type == features_device.type == "cuda"
        for model_device, features_device in observed_devices
    )
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["model_state_dict"].values())
