from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from src.drums_onset_profile_package import (
    COMPONENT_ID,
    PROFILE_ID,
    DrumsProfilePackagingError,
    package_drums_onset_experiment,
)
from src.inference import drums_onset_classifier_runtime
from src.inference.drums_onset_classifier_profile import (
    CAPABILITY,
    DrumsOnsetClassifierEvaluationProfile,
    load_drums_onset_classifier_evaluation_profile,
)
from src.inference.drums_onset_classifier_runtime import (
    DrumsOnsetClassifierRuntime,
    DrumsOnsetClassifierRuntimeError,
)
from src.model_bundle import BundleValidationError, load_model_bundle
from src.worker import package_checkpoint_request, preflight_chart_request


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _experiment(root: Path, *, deployment_status: str = "requires_profile_packaging") -> Path:
    root.mkdir()
    config = root / "training-config.yaml"
    config.write_text(
        "model:\n"
        "  num_classes: 8\n"
        "  branch_channels: [1, 32, 64, 128, 256]\n"
        "  context_size: 4\n"
        "  context_hidden: 64\n"
        "  spectral_dim: 32\n"
        "  classifier_hidden: 512\n"
        "  dropout: 0.3\n"
        "  use_freq_attn: false\n"
    )
    checkpoint = root / "checkpoints" / "best_f1.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"training checkpoint fixture")
    ledger = {
        "format": "strum-drums-onset-experiment/v1",
        "pipeline": {"id": "drums.onset-classifier/v1", "version": 1},
        "model": {
            "id": "catalog-v2",
            "profile": "onset_classifier_v2",
            "architecture": "OnsetClassifier/v2",
        },
        "task_view": {
            "format": "task",
            "sha256": "a" * 64,
            "catalog_id": "catalog",
            "catalog_content_sha256": "b" * 64,
        },
        "preprocessing": {"id": "drums-onset-windows/v1", "splits": []},
        "training": {"config_name": config.name, "config_sha256": _sha256(config)},
        "checkpoint": {
            "name": "checkpoints/best_f1.pt",
            "sha256": _sha256(checkpoint),
            "byte_length": checkpoint.stat().st_size,
            "format": "torch-training-checkpoint/v1",
            "deployment_status": deployment_status,
        },
        "metrics": {"test_overall_f1": 0.5, "test_loss": 1.0, "test_num_samples": 8},
    }
    (root / "experiment.json").write_text(json.dumps(ledger))
    return root


def test_package_worker_experiment_as_evaluation_only_bundle(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "experiment")
    output = tmp_path / "bundle"

    result = package_drums_onset_experiment(experiment, output)

    bundle = load_model_bundle(output, check_files=True)
    assert result["deployment_status"] == "evaluation_only_not_auto_chart_deployable"
    assert result["capability"] == CAPABILITY
    assert str(experiment) not in json.dumps(result)
    assert bundle.validate(check_files=True, verify_hashes=True) == []
    profile = load_drums_onset_classifier_evaluation_profile(bundle, PROFILE_ID)
    assert profile.component_id == COMPONENT_ID
    assert str(experiment) not in (output / "lineage.json").read_text()


def test_packaged_v2_profile_is_explicitly_not_auto_chart_executable(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "experiment")
    output = tmp_path / "bundle"
    package_drums_onset_experiment(experiment, output)
    request = tmp_path / "preflight.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": PROFILE_ID,
                "difficulty_policy": "evaluation_only",
                "instruments": ["drums"],
                "device": "cpu",
            }
        )
    )

    plan = preflight_chart_request(request)

    assert plan["capability"] == CAPABILITY
    assert plan["execution"] == "not_available"


def test_package_requires_a_genuine_worker_experiment_boundary(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "experiment", deployment_status="drums.v14-expert/v1")

    with pytest.raises(DrumsProfilePackagingError, match="assets do not match|supported V2"):
        package_drums_onset_experiment(experiment, tmp_path / "bundle")


def test_worker_package_request_has_no_renderer_path_in_response(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "experiment")
    request = tmp_path / "package.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "drums.onset-classifier/v1",
                "experiment_root": str(experiment),
                "output": str(tmp_path / "bundle"),
            }
        )
    )

    result = package_checkpoint_request(request)

    assert result["status"] == "packaged"
    assert str(tmp_path) not in json.dumps(result)


def test_profile_rejects_auto_chart_claim(tmp_path: Path) -> None:
    experiment = _experiment(tmp_path / "experiment")
    output = tmp_path / "bundle"
    package_drums_onset_experiment(experiment, output)
    config = output / "profiles" / "drums-onset-classifier-evaluation.json"
    raw = json.loads(config.read_text())
    raw["auto_chart_status"] = "supported"
    config.write_text(json.dumps(raw))
    manifest = output / "strum-model-bundle.json"
    bundle_raw = json.loads(manifest.read_text())
    bundle_raw["profiles"][PROFILE_ID]["configuration_sha256"] = _sha256(config)
    bundle_raw["profiles"][PROFILE_ID]["configuration_byte_length"] = config.stat().st_size
    manifest.write_text(json.dumps(bundle_raw))

    bundle = load_model_bundle(output, check_files=True)
    with pytest.raises(BundleValidationError, match="unsupported fields"):
        load_drums_onset_classifier_evaluation_profile(bundle, PROFILE_ID)


def test_runtime_only_classifies_prepared_windows() -> None:
    profile = DrumsOnsetClassifierEvaluationProfile(
        profile_id=PROFILE_ID,
        component_id=COMPONENT_ID,
        model_parameters={},
        configuration_sha256="a" * 64,
    )

    class Model(torch.nn.Module):
        def forward(
            self, fine: torch.Tensor, coarse: torch.Tensor, context: torch.Tensor
        ) -> torch.Tensor:
            assert fine.shape == (2, 1, 128, 87)
            assert coarse.shape == (2, 1, 128, 44)
            assert context.shape == (2, 64)
            return torch.zeros((2, 8))

    runtime = DrumsOnsetClassifierRuntime(profile, Model(), torch.device("cpu"))
    result = runtime.classify_windows(
        torch.zeros((2, 1, 128, 87)), torch.zeros((2, 1, 128, 44)), torch.zeros((2, 64))
    )

    assert len(result) == 2
    assert result[0].probabilities == (0.5,) * 8
    with pytest.raises(DrumsOnsetClassifierRuntimeError, match="fine mel"):
        runtime.classify_windows(
            torch.zeros((2, 128, 87)), torch.zeros((2, 1, 128, 44)), torch.zeros((2, 64))
        )


def test_runtime_loader_uses_safe_checkpoint_and_exact_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = DrumsOnsetClassifierEvaluationProfile(
        profile_id=PROFILE_ID,
        component_id=COMPONENT_ID,
        model_parameters={"num_classes": 8, "use_freq_attn": False},
        configuration_sha256="a" * 64,
    )
    seen: dict[str, object] = {}

    class Loaded(torch.nn.Module):
        def __init__(self, **parameters: object) -> None:
            super().__init__()
            seen["parameters"] = parameters

        def load_state_dict(self, state_dict: object, strict: bool = True):
            seen["state_dict"] = state_dict
            seen["strict"] = strict
            return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr(drums_onset_classifier_runtime, "OnsetClassifier", Loaded)
    monkeypatch.setattr(
        drums_onset_classifier_runtime.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {"weight": torch.tensor([1.0])}},
    )

    DrumsOnsetClassifierRuntime.from_profile(
        profile, checkpoint_path=tmp_path / "v2.pt", device="cpu"
    )

    assert seen["parameters"] == {
        "num_classes": 8,
        "branch_channels": [1, 32, 64, 128, 256],
        "context_size": 4,
        "context_hidden": 64,
        "classifier_hidden": 512,
        "spectral_dim": 32,
        "dropout": 0.3,
        "use_freq_attn": False,
        "use_hpss": False,
        "enhanced_spectral": False,
        "use_contrastive": False,
        "use_aux_head": False,
        "projection_dim": 128,
        "use_dual_head": False,
        "tom_head_hidden": 256,
        "use_lowfreq_branch": False,
        "use_lowfreq_spectral": False,
        "use_crash_flux": False,
        "crash_flux_dim": 32,
    }
    assert seen["strict"] is True
