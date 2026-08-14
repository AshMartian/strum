"""Typed, fail-closed configuration for the first bundle-backed chart slice."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from src.model_bundle import BundleValidationError, ModelBundle

FORMAT = "strum-guitar-hybrid-rule-profile/v1"
CAPABILITY = "guitar.hybrid-v2-rule/v1"


@dataclass(frozen=True)
class GuitarHybridRuleProfile:
    """Verified Guitar onset + Basic Pitch + rule mapper execution settings."""

    profile_id: str
    onset_component: str
    onset_threshold: float
    latency_offset_ms: float
    min_pitch_amplitude: float
    min_pitch: int
    max_pitch: int
    snap_window_ms: float
    sustain_min_duration_ms: float
    max_chord_size: int
    voice_filter: bool
    configuration_sha256: str


def load_guitar_hybrid_rule_profile(
    bundle: ModelBundle, profile_id: str
) -> GuitarHybridRuleProfile:
    """Resolve one safe Guitar hybrid profile; never infer values from env/defaults."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.capability != CAPABILITY:
        raise BundleValidationError("profile is not a guitar.hybrid-v2-rule/v1 profile")
    if profile.instruments != ("guitar",) or profile.difficulty_policies != ("expert_only",):
        raise BundleValidationError(
            "guitar hybrid rule profile must declare Guitar Expert-only output"
        )
    if len(profile.required_components) != 1:
        raise BundleValidationError(
            "guitar hybrid rule profile requires exactly one onset component"
        )
    component_name = profile.required_components[0]
    component = bundle.component(component_name)
    if (
        component is None
        or component.checkpoint is None
        or component.config is None
        or component.config_sha256 is None
        or component.config_byte_length is None
        or component.architecture != "GuitarOnsetCRNN/v2"
        or profile.configuration is None
    ):
        raise BundleValidationError(
            "guitar hybrid rule profile has incomplete onset/configuration assets"
        )
    try:
        raw = json.loads(profile.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("guitar hybrid rule configuration is unreadable") from error
    required = {
        "schema_version",
        "format",
        "onset_threshold",
        "latency_offset_ms",
        "min_pitch_amplitude",
        "min_pitch",
        "max_pitch",
        "snap_window_ms",
        "sustain_min_duration_ms",
        "max_chord_size",
        "voice_filter",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != FORMAT
    ):
        raise BundleValidationError("guitar hybrid rule configuration has unsupported fields")
    numeric = (
        "onset_threshold",
        "latency_offset_ms",
        "min_pitch_amplitude",
        "snap_window_ms",
        "sustain_min_duration_ms",
    )
    if not all(
        isinstance(raw[key], (int, float)) and not isinstance(raw[key], bool) for key in numeric
    ):
        raise BundleValidationError("guitar hybrid rule configuration has invalid numeric settings")
    if not all(
        isinstance(raw[key], int) and not isinstance(raw[key], bool)
        for key in ("min_pitch", "max_pitch", "max_chord_size")
    ) or not isinstance(raw["voice_filter"], bool):
        raise BundleValidationError(
            "guitar hybrid rule configuration has invalid discrete settings"
        )
    if (
        not 0 < raw["onset_threshold"] <= 1
        or not 0 <= raw["min_pitch_amplitude"] <= 1
        or raw["min_pitch"] > raw["max_pitch"]
        or raw["snap_window_ms"] <= 0
        or raw["sustain_min_duration_ms"] < 0
        or raw["max_chord_size"] < 1
    ):
        raise BundleValidationError("guitar hybrid rule configuration settings are out of range")
    return GuitarHybridRuleProfile(
        profile_id=profile_id,
        onset_component=component_name,
        onset_threshold=float(raw["onset_threshold"]),
        latency_offset_ms=float(raw["latency_offset_ms"]),
        min_pitch_amplitude=float(raw["min_pitch_amplitude"]),
        min_pitch=raw["min_pitch"],
        max_pitch=raw["max_pitch"],
        snap_window_ms=float(raw["snap_window_ms"]),
        sustain_min_duration_ms=float(raw["sustain_min_duration_ms"]),
        max_chord_size=raw["max_chord_size"],
        voice_filter=raw["voice_filter"],
        configuration_sha256=hashlib.sha256(profile.configuration.read_bytes()).hexdigest(),
    )
