from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.inference.guitar_hybrid_profile import load_guitar_hybrid_rule_profile
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle


def _write_profile_bundle(root: Path, *, max_chord_size: int = 3) -> Path:
    checkpoint = root / "weights" / "onset.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"verified onset checkpoint")
    model_config = root / "configs" / "guitar.yaml"
    model_config.parent.mkdir()
    model_config.write_text("onset: {}\n")
    profile_config = root / "profiles" / "guitar-rule.json"
    profile_config.parent.mkdir()
    profile_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-guitar-hybrid-rule-profile/v1",
                "onset_threshold": 0.4,
                "latency_offset_ms": 25,
                "min_pitch_amplitude": 0.3,
                "min_pitch": 36,
                "max_pitch": 88,
                "snap_window_ms": 75,
                "sustain_min_duration_ms": 400,
                "max_chord_size": max_chord_size,
                "voice_filter": True,
            }
        )
    )
    manifest = {
        "schema_version": 1,
        "model_id": "guitar-rule-fixture",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            "guitar.onset": {
                "checkpoint": "weights/onset.pt",
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "byte_length": checkpoint.stat().st_size,
                "config": "configs/guitar.yaml",
                "config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
                "config_byte_length": model_config.stat().st_size,
                "architecture": "GuitarOnsetCRNN/v2",
            }
        },
        "profiles": {
            "guitar-rule": {
                "capability": "guitar.hybrid-v2-rule/v1",
                "instruments": ["guitar"],
                "required_components": ["guitar.onset"],
                "difficulty_policies": ["expert_only"],
                "configuration": "profiles/guitar-rule.json",
            }
        },
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    return root


def test_guitar_hybrid_profile_requires_verified_explicit_policy(tmp_path: Path) -> None:
    bundle = load_model_bundle(_write_profile_bundle(tmp_path), check_files=True)

    profile = load_guitar_hybrid_rule_profile(bundle, "guitar-rule")

    assert profile.onset_component == "guitar.onset"
    assert profile.max_chord_size == 3
    assert profile.configuration_sha256


def test_guitar_hybrid_profile_rejects_invalid_execution_setting(tmp_path: Path) -> None:
    bundle = load_model_bundle(_write_profile_bundle(tmp_path, max_chord_size=0), check_files=True)

    with pytest.raises(BundleValidationError, match="out of range"):
        load_guitar_hybrid_rule_profile(bundle, "guitar-rule")
