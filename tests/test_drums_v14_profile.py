from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.inference.drums_v14_profile import load_drums_v14_expert_profile
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.worker import preflight_chart_request


def _bundle(root: Path, *, postprocess: str = "none") -> Path:
    checkpoint = root / "weights" / "v14.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"verified V14 checkpoint")
    model_config = root / "configs" / "drums-v14.yaml"
    model_config.parent.mkdir()
    model_config.write_text("model: drums-v14\n")
    profile_config = root / "profiles" / "drums-v14.json"
    profile_config.parent.mkdir()
    profile_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-drums-v14-expert-profile/v1",
                "model_architecture": "TwoStageDrumsCRNN/v14",
                "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                "segment_duration_seconds": 10,
                "overlap": 0.5,
                "onset_threshold": 0.4,
                "class_thresholds": [0.3, 0.25, 0.35, 0.12, 0.28, 0.12, 0.35, 0.12],
                "min_distance_ms": 20,
                "postprocess": postprocess,
                "class_to_midi": [96, 97, 98, 98, 99, 99, 100, 100],
                "model_parameters": {
                    "n_mels": 128, "conv_channels": [64, 128, 256, 512],
                    "freq_subbands": [32, 64, 96, 128], "subband_proj_dim": 256,
                    "lstm_hidden": 640, "lstm_layers": 3, "attention_heads": 10,
                    "attention_type": "flash", "attention_window": 512, "dropout": 0.0,
                    "onset_detector_hidden": 320, "classifier_hidden": 640,
                    "num_classes": 8, "predict_velocity": True
                },
            }
        )
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "drums-v14-fixture",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    "drums.v14": {
                        "checkpoint": "weights/v14.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "byte_length": checkpoint.stat().st_size,
                        "config": "configs/drums-v14.yaml",
                        "config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
                        "config_byte_length": model_config.stat().st_size,
                        "architecture": "TwoStageDrumsCRNN/v14",
                        "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                    }
                },
                "profiles": {
                    "drums-v14-expert": {
                        "capability": "drums.v14-expert/v1",
                        "instruments": ["drums"],
                        "required_components": ["drums.v14"],
                        "difficulty_policies": ["expert_only"],
                        "configuration": "profiles/drums-v14.json",
                        "configuration_sha256": hashlib.sha256(profile_config.read_bytes()).hexdigest(),
                        "configuration_byte_length": profile_config.stat().st_size,
                    }
                },
            }
        )
    )
    return root


def test_drums_v14_profile_requires_single_verified_expert_path(tmp_path: Path) -> None:
    bundle = load_model_bundle(_bundle(tmp_path), check_files=True)

    profile = load_drums_v14_expert_profile(bundle, "drums-v14-expert")

    assert profile.component_id == "drums.v14"
    assert profile.class_to_midi == (96, 97, 98, 98, 99, 99, 100, 100)
    assert profile.configuration_sha256


def test_drums_v14_profile_rejects_legacy_postprocess(tmp_path: Path) -> None:
    bundle = load_model_bundle(_bundle(tmp_path, postprocess="legacy"), check_files=True)

    with pytest.raises(BundleValidationError, match="unsupported fields"):
        load_drums_v14_expert_profile(bundle, "drums-v14-expert")


def test_drums_v14_profile_is_preflightable_for_direct_execution(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    request = tmp_path / "preflight.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(root),
                "profile_id": "drums-v14-expert",
                "difficulty_policy": "expert_only",
                "instruments": ["drums"],
                "device": "cuda",
            }
        )
    )

    plan = preflight_chart_request(request)

    assert plan["capability"] == "drums.v14-expert/v1"
    assert plan["execution"] == "available"
    assert plan["profile_configuration_sha256"]
