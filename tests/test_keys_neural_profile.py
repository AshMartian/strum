from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest
import torch

from src.inference.guitar_neural import GuitarEvent
from src.inference.keys_neural_profile import (
    CAPABILITY,
    EVALUATION_FORMAT,
    KeysNeuralCharter,
    load_keys_neural_expert_profile,
)
from src.keys_profile_packaging import KeysProfilePackagingError, package_keys_profile
from src.profile_quality_policy import profile_quality_policy, profile_quality_policy_sha256
from src.model_bundle import MANIFEST_FILENAME, load_model_bundle
from src.models.guitar_v1 import (
    FretClassifierConfig,
    GuitarFretClassifier,
    GuitarOnsetCRNN,
    OnsetCRNNConfig,
)
from src.worker import WorkerRequestError, preflight_chart_request, run_chart_request


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _component(root: Path, path: Path, config: Path, architecture: str) -> dict[str, object]:
    return {
        "checkpoint": path.as_posix(),
        "sha256": _sha256(root / path),
        "byte_length": (root / path).stat().st_size,
        "config": config.as_posix(),
        "config_sha256": _sha256(root / config),
        "config_byte_length": (root / config).stat().st_size,
        "architecture": architecture,
        "preprocessing": "keys-logmel-windows/v1",
    }


def _keys_experiment(root: Path) -> tuple[Path, Path]:
    experiment = root / "experiment"
    bundle = experiment / "bundle"
    config = {
        "schema_version": 1,
        "format": "strum-keys-neural-model-config/v1",
        "instrument": "keys",
        "model_implementation": "five-lane-crnn/v1",
        "preprocessing": "keys-logmel-windows/v1",
        "audio": {
            "sample_rate": 22050,
            "n_mels": 128,
            "n_fft": 2048,
            "hop_length": 512,
            "fmin": 30.0,
            "fmax": 8000.0,
        },
        "onset_model": {
            "n_mels": 128,
            "in_channels": 1,
            "cnn_channels": [2, 2, 2, 2],
            "rnn_hidden": 2,
            "rnn_layers": 1,
            "rnn_dropout": 0.0,
            "head_hidden": 2,
            "head_dropout": 0.0,
        },
        "fret_model": {
            "n_mels": 128,
            "n_frames": 22,
            "in_channels": 1,
            "cnn_channels": [2, 2, 2, 2],
            "head_hidden": 2,
            "dropout": 0.0,
            "n_frets": 5,
            "aux_chord_head": True,
        },
        "onset_inference": {"peak_threshold": 0.4, "peak_min_distance_frames": 3},
    }
    config_path = bundle / "configs" / "keys-training-config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, sort_keys=True))
    onset_path = bundle / "weights" / "keys-onset.pt"
    fret_path = bundle / "weights" / "keys-fret.pt"
    onset_path.parent.mkdir()
    torch.save(
        {"state_dict": GuitarOnsetCRNN(OnsetCRNNConfig(**config["onset_model"])).state_dict()},
        onset_path,
    )
    torch.save(
        {
            "state_dict": GuitarFretClassifier(
                FretClassifierConfig(**config["fret_model"])
            ).state_dict()
        },
        fret_path,
    )
    manifest = {
        "schema_version": 1,
        "model_id": "catalog-keys-v1",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            "keys.onset": _component(
                bundle,
                Path("weights/keys-onset.pt"),
                Path("configs/keys-training-config.json"),
                "FiveLaneOnsetCRNN/v1",
            ),
            "keys.fret": _component(
                bundle,
                Path("weights/keys-fret.pt"),
                Path("configs/keys-training-config.json"),
                "FiveLaneFretClassifier/v1",
            ),
        },
    }
    manifest_path = bundle / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest))
    experiment.mkdir(exist_ok=True)
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-experiment/v1",
                "lifecycle": "completed",
                "pipeline": {"id": "keys.onset-fret", "version": 1},
                "deployment_status": "requires_keys_profile_evaluation_and_packaging",
                "model_bundle": {
                    "model_id": "catalog-keys-v1",
                    "manifest_sha256": _sha256(manifest_path),
                },
            }
        )
    )
    return experiment, bundle


def _evaluation(bundle: Path, output: Path) -> Path:
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": EVALUATION_FORMAT,
                "model_id": "catalog-keys-v1",
                "bundle_manifest_sha256": _sha256(bundle / MANIFEST_FILENAME),
                "task_view_sha256": "a" * 64,
                "split": "val",
                "records_evaluated": 2,
                "alignment_tolerance_ms": 50.0,
                "quality_policy": profile_quality_policy(),
                "quality_policy_sha256": profile_quality_policy_sha256(),
                "metrics": {"onset_f1": 0.8, "fret_f1": 0.8, "event_f1": 0.7},
            }
        )
    )
    return output


def _package(tmp_path: Path) -> Path:
    experiment, bundle = _keys_experiment(tmp_path)
    output = tmp_path / "deployable"
    result = package_keys_profile(
        experiment_dir=experiment,
        evaluation_path=_evaluation(bundle, tmp_path / "evaluation.json"),
        output_dir=output,
        profile_id="keys-v1-expert",
    )
    assert result["capability"] == CAPABILITY
    return output


def test_keys_profile_is_separate_and_preflights_for_keys_only(tmp_path: Path) -> None:
    output = _package(tmp_path)
    profile = load_keys_neural_expert_profile(
        load_model_bundle(output, check_files=True), "keys-v1-expert"
    )
    assert profile.onset_component == "keys.onset"
    request = tmp_path / "preflight.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "keys-v1-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["keys"],
                "device": "cpu",
            }
        )
    )
    plan = preflight_chart_request(request)
    assert plan["capability"] == CAPABILITY
    assert plan["execution"] == "available"
    assert plan["instrument_results"]["keys"]["stages"]["expert_chart"]["status"] == "ready"
    request.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "keys-v1-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )
    with pytest.raises(WorkerRequestError, match="does not cover"):
        preflight_chart_request(request)


def test_keys_runtime_writes_part_keys_not_part_guitar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = _package(tmp_path)
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"safe-audio-input")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "keys-v1-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["keys"],
                "device": "cpu",
            }
        )
    )
    request = tmp_path / "run.json"
    request.write_text(
        json.dumps(
            {
                "preflight_request": str(preflight),
                "audio_path": str(audio),
                "output_dir": str(tmp_path / "result"),
            }
        )
    )
    import scripts.preprocess_guitar_windows as preprocess

    monkeypatch.setattr(preprocess, "load_audio_mono_22050", lambda _: object())
    monkeypatch.setattr(
        KeysNeuralCharter,
        "from_bundle_profile",
        classmethod(
            lambda cls, _bundle, _profile, *, device: SimpleNamespace(
                transcribe=lambda *_args, **_kwargs: [
                    GuitarEvent(0.1, (0, 2), 0.9, (0.9, 0.1, 0.9, 0.1, 0.1))
                ]
            )
        ),
    )
    result = run_chart_request(request)
    assert result["expert_event_count"] == 1
    midi = mido.MidiFile(tmp_path / "result" / "notes.mid")
    assert midi.tracks[0].name == "PART KEYS"
    assert "guitar" not in result["instrument_results"]


def test_keys_packaging_rejects_wrong_experiment_or_report(tmp_path: Path) -> None:
    experiment, bundle = _keys_experiment(tmp_path)
    data = json.loads((experiment / "experiment.json").read_text())
    data["pipeline"] = {"id": "bass.onset-fret", "version": 1}
    (experiment / "experiment.json").write_text(json.dumps(data))
    with pytest.raises(KeysProfilePackagingError, match="packageable"):
        package_keys_profile(
            experiment_dir=experiment,
            evaluation_path=_evaluation(bundle, tmp_path / "evaluation.json"),
            output_dir=tmp_path / "deployable",
            profile_id="keys-v1-expert",
        )
    assert not (tmp_path / "deployable").exists()
