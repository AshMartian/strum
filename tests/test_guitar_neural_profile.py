from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import src.worker as worker_module
from src.guitar_profile_packaging import GuitarProfilePackagingError, package_guitar_profile
from src.inference.guitar_neural import GuitarEvent, GuitarNeuralCharter
from src.inference.guitar_neural_profile import (
    CAPABILITY,
    EVALUATION_FORMAT,
    load_guitar_neural_expert_profile,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.models.guitar_v1 import (
    FretClassifierConfig,
    GuitarFretClassifier,
    GuitarOnsetCRNN,
    OnsetCRNNConfig,
)
from src.profile_quality_policy import profile_quality_policy, profile_quality_policy_sha256
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
        "preprocessing": "guitar-logmel-windows/v1",
    }


def _worker_experiment(root: Path) -> tuple[Path, Path]:
    """Create the same portable layout produced by the Guitar worker."""
    experiment = root / "experiment"
    bundle = experiment / "bundle"
    config = {
        "schema_version": 1,
        "format": "strum-guitar-neural-model-config/v1",
        "preprocessing": "guitar-logmel-windows/v1",
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
    config_path = bundle / "configs" / "guitar-training-config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, sort_keys=True))
    onset_path = bundle / "weights" / "guitar-onset.pt"
    fret_path = bundle / "weights" / "guitar-fret.pt"
    onset_path.parent.mkdir()
    torch.save(
        {
            "state_dict": GuitarOnsetCRNN(OnsetCRNNConfig(**config["onset_model"])).state_dict(),
            "epoch": 1,
            "val_f1": 0.8,
        },
        onset_path,
    )
    torch.save(
        {
            "state_dict": GuitarFretClassifier(
                FretClassifierConfig(**config["fret_model"])
            ).state_dict(),
            "epoch": 1,
            "val_f1": 0.8,
        },
        fret_path,
    )
    manifest = {
        "schema_version": 1,
        "model_id": "catalog-guitar-v1",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            "guitar.onset": _component(
                bundle,
                Path("weights/guitar-onset.pt"),
                Path("configs/guitar-training-config.json"),
                "GuitarOnsetCRNN/v1",
            ),
            "guitar.fret": _component(
                bundle,
                Path("weights/guitar-fret.pt"),
                Path("configs/guitar-training-config.json"),
                "GuitarFretClassifier/v1",
            ),
        },
    }
    manifest_path = bundle / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest))
    experiment.mkdir(exist_ok=True)
    experiment_data = {
        "schema_version": 1,
        "format": "strum-experiment/v1",
        "lifecycle": "completed",
        "pipeline": {"id": "guitar.onset-fret", "version": 1},
        "deployment_status": "requires_profile_packaging",
        "model_bundle": {
            "model_id": "catalog-guitar-v1",
            "manifest_sha256": _sha256(manifest_path),
        },
    }
    (experiment / "experiment.json").write_text(json.dumps(experiment_data))
    return experiment, bundle


def _evaluation(bundle: Path, output: Path, *, onset_f1: float = 0.8) -> Path:
    manifest = bundle / MANIFEST_FILENAME
    output.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": EVALUATION_FORMAT,
                "model_id": "catalog-guitar-v1",
                "bundle_manifest_sha256": _sha256(manifest),
                "task_view_sha256": "a" * 64,
                "split": "val",
                "records_evaluated": 2,
                "alignment_tolerance_ms": 50.0,
                "quality_policy": profile_quality_policy(),
                "quality_policy_sha256": profile_quality_policy_sha256(),
                "metrics": {"onset_f1": onset_f1, "fret_f1": 0.8, "event_f1": 0.7},
            }
        )
    )
    return output


def test_worker_experiment_requires_evaluation_before_deployable_profile(tmp_path: Path) -> None:
    experiment, bundle = _worker_experiment(tmp_path)
    with pytest.raises(GuitarProfilePackagingError, match="evaluation"):
        package_guitar_profile(
            experiment_dir=experiment,
            evaluation_path=tmp_path / "missing.json",
            output_dir=tmp_path / "deployable",
            profile_id="guitar-v1",
        )
    assert not (bundle / "profiles").exists()


def test_packaged_profile_loads_real_worker_components_and_preflights(tmp_path: Path) -> None:
    experiment, bundle = _worker_experiment(tmp_path)
    evaluation = _evaluation(bundle, tmp_path / "evaluation.json")
    output = tmp_path / "deployable"

    result = package_guitar_profile(
        experiment_dir=experiment,
        evaluation_path=evaluation,
        output_dir=output,
        profile_id="guitar-v1-expert",
    )

    assert result["deployment_status"] == "deployable"
    assert not (bundle / "profiles").exists()
    packaged = load_model_bundle(output, check_files=True)
    profile = load_guitar_neural_expert_profile(packaged, "guitar-v1-expert")
    assert profile.fret_component == "guitar.fret"
    charter = GuitarNeuralCharter.from_bundle_profile(packaged, profile, device="cpu")
    assert charter.default_onset_thr == 0.4
    request = tmp_path / "preflight.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "guitar-v1-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )
    plan = preflight_chart_request(request)
    assert plan["capability"] == CAPABILITY
    assert plan["execution"] == "available"


def test_profile_runtime_rejects_non_tensor_checkpoint_state(tmp_path: Path) -> None:
    experiment, bundle = _worker_experiment(tmp_path)
    evaluation = _evaluation(bundle, tmp_path / "evaluation.json")
    output = tmp_path / "deployable"
    package_guitar_profile(
        experiment_dir=experiment,
        evaluation_path=evaluation,
        output_dir=output,
        profile_id="guitar-v1-expert",
    )
    bad_checkpoint = output / "weights" / "guitar-onset.pt"
    torch.save({"state_dict": {"not-a-tensor": "unsafe"}}, bad_checkpoint)
    manifest = json.loads((output / MANIFEST_FILENAME).read_text())
    manifest["components"]["guitar.onset"]["sha256"] = _sha256(bad_checkpoint)
    manifest["components"]["guitar.onset"]["byte_length"] = bad_checkpoint.stat().st_size
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    packaged = load_model_bundle(output, check_files=True)
    profile = load_guitar_neural_expert_profile(packaged, "guitar-v1-expert")
    with pytest.raises(BundleValidationError, match="state dict"):
        GuitarNeuralCharter.from_bundle_profile(packaged, profile, device="cpu")


def test_chart_run_uses_only_typed_neural_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, bundle = _worker_experiment(tmp_path)
    output_bundle = tmp_path / "deployable"
    package_guitar_profile(
        experiment_dir=experiment,
        evaluation_path=_evaluation(bundle, tmp_path / "evaluation.json"),
        output_dir=output_bundle,
        profile_id="guitar-v1-expert",
    )
    audio = tmp_path / "song.wav"
    audio.write_bytes(b"safe-audio-input")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output_bundle),
                "profile_id": "guitar-v1-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
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
        GuitarNeuralCharter,
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
    preflight_plan = preflight_chart_request(preflight)
    expected_manifest_sha256 = _sha256(output_bundle / MANIFEST_FILENAME)
    assert preflight_plan["manifest_sha256"] == expected_manifest_sha256
    assert result["manifest_sha256"] == expected_manifest_sha256
    run_manifest = json.loads((tmp_path / "result" / "run.json").read_text())
    assert run_manifest["manifest_sha256"] == expected_manifest_sha256
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(run_manifest)
    assert (tmp_path / "result" / "notes.mid").is_file()

    original_preflight = worker_module.preflight_chart_request

    def replace_manifest_after_preflight(path: Path) -> dict[str, object]:
        plan = original_preflight(path)
        manifest_path = output_bundle / MANIFEST_FILENAME
        replaced = json.loads(manifest_path.read_text())
        replaced["model_id"] = "catalog-guitar-replaced-v1"
        manifest_path.write_text(json.dumps(replaced, indent=2, sort_keys=True) + "\n")
        return plan

    monkeypatch.setattr(worker_module, "preflight_chart_request", replace_manifest_after_preflight)
    with pytest.raises(WorkerRequestError, match="bundle identity does not match preflight"):
        run_chart_request(request)


def test_chart_run_keeps_the_preflight_profile_during_same_bundle_profile_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rewritten preflight file cannot select another valid profile mid-run."""
    experiment, bundle = _worker_experiment(tmp_path)
    output_bundle = tmp_path / "deployable"
    package_guitar_profile(
        experiment_dir=experiment,
        evaluation_path=_evaluation(bundle, tmp_path / "evaluation.json"),
        output_dir=output_bundle,
        profile_id="guitar-v1-expert-a",
    )
    manifest_path = output_bundle / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"]["guitar-v1-expert-b"] = dict(manifest["profiles"]["guitar-v1-expert-a"])
    manifest_path.write_text(json.dumps(manifest))
    # Both profiles are independently executable and share the same immutable
    # bundle. The test therefore isolates profile authority from manifest race
    # protection.
    packaged = load_model_bundle(output_bundle, check_files=True)
    load_guitar_neural_expert_profile(packaged, "guitar-v1-expert-a")
    load_guitar_neural_expert_profile(packaged, "guitar-v1-expert-b")

    audio = tmp_path / "song.wav"
    audio.write_bytes(b"safe-audio-input")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output_bundle),
                "profile_id": "guitar-v1-expert-a",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
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

    real_preflight = worker_module.preflight_chart_request

    def preflight_then_swap_profile(path: Path) -> dict[str, object]:
        plan = real_preflight(path)
        preflight.write_text(
            json.dumps(
                {
                    "model_root": str(output_bundle),
                    "profile_id": "guitar-v1-expert-b",
                    "difficulty_policy": "expert_only",
                    "instruments": ["guitar"],
                    "device": "cpu",
                }
            )
        )
        return plan

    monkeypatch.setattr(worker_module, "preflight_chart_request", preflight_then_swap_profile)
    import scripts.preprocess_guitar_windows as preprocess

    monkeypatch.setattr(preprocess, "load_audio_mono_22050", lambda _: object())
    selected_profile_ids: list[str] = []
    monkeypatch.setattr(
        GuitarNeuralCharter,
        "from_bundle_profile",
        classmethod(
            lambda cls, _bundle, profile, *, device: (
                selected_profile_ids.append(profile.profile_id)
                or SimpleNamespace(
                    transcribe=lambda *_args, **_kwargs: [
                        GuitarEvent(0.1, (0,), 0.9, (0.9, 0.1, 0.1, 0.1, 0.1))
                    ]
                )
            )
        ),
    )

    result = run_chart_request(request)

    assert result["profile_id"] == "guitar-v1-expert-a"
    assert selected_profile_ids == ["guitar-v1-expert-a"]
