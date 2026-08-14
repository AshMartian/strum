"""Fail-closed configuration for direct Expert Drums V14 execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping

from src.model_bundle import BundleValidationError, ModelBundle

FORMAT = "strum-drums-v14-expert-profile/v1"
CAPABILITY = "drums.v14-expert/v1"
ARCHITECTURE = "TwoStageDrumsCRNN/v14"
PREPROCESSING = "drums-logmel-44100-2048-512-128-v1"
CLASS_NAMES = (
    "kick",
    "snare",
    "hihat",
    "high_tom",
    "ride",
    "low_tom",
    "crash",
    "floor_tom",
)


@dataclass(frozen=True)
class DrumsV14ExpertProfile:
    """Immutable V14-only Expert Drums settings, with no companion fallbacks."""

    profile_id: str
    component_id: str
    onset_threshold: float
    class_thresholds: tuple[float, ...]
    min_distance_ms: float
    segment_duration_seconds: float
    overlap: float
    class_to_midi: tuple[int, ...]
    model_parameters: Mapping[str, object]
    configuration_sha256: str


def load_drums_v14_expert_profile(
    bundle: ModelBundle, profile_id: str
) -> DrumsV14ExpertProfile:
    """Resolve a V14 profile without importing legacy batch inference code."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.capability != CAPABILITY:
        raise BundleValidationError("profile is not a drums.v14-expert/v1 profile")
    if profile.instruments != ("drums",) or profile.difficulty_policies != ("expert_only",):
        raise BundleValidationError("drums V14 profile must declare Drums Expert-only output")
    if len(profile.required_components) != 1:
        raise BundleValidationError("drums V14 profile requires exactly one V14 component")
    component_id = profile.required_components[0]
    component = bundle.component(component_id)
    if (
        component is None
        or component.checkpoint is None
        or component.config is None
        or component.config_sha256 is None
        or component.config_byte_length is None
        or component.architecture != ARCHITECTURE
        or component.preprocessing != PREPROCESSING
        or profile.configuration is None
    ):
        raise BundleValidationError("drums V14 profile has incomplete verified assets")
    try:
        raw = json.loads(profile.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("drums V14 configuration is unreadable") from error
    required = {
        "schema_version",
        "format",
        "model_architecture",
        "preprocessing",
        "segment_duration_seconds",
        "overlap",
        "onset_threshold",
        "class_thresholds",
        "min_distance_ms",
        "postprocess",
        "class_to_midi",
        "model_parameters",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != FORMAT
        or raw.get("model_architecture") != ARCHITECTURE
        or raw.get("preprocessing") != PREPROCESSING
        or raw.get("postprocess") != "none"
    ):
        raise BundleValidationError("drums V14 configuration has unsupported fields")
    numeric = ("segment_duration_seconds", "overlap", "onset_threshold", "min_distance_ms")
    if not all(
        isinstance(raw[key], (int, float)) and not isinstance(raw[key], bool) for key in numeric
    ):
        raise BundleValidationError("drums V14 configuration has invalid numeric settings")
    thresholds, midi_notes, model_parameters = (
        raw["class_thresholds"],
        raw["class_to_midi"],
        raw["model_parameters"],
    )
    if (
        raw["segment_duration_seconds"] <= 0
        or not 0 <= raw["overlap"] < 1
        or not 0 < raw["onset_threshold"] <= 1
        or raw["min_distance_ms"] <= 0
        or not isinstance(thresholds, list)
        or len(thresholds) != len(CLASS_NAMES)
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and 0 < value <= 1
            for value in thresholds
        )
        or not isinstance(midi_notes, list)
        or len(midi_notes) != len(CLASS_NAMES)
        or not all(isinstance(note, int) and 96 <= note <= 100 for note in midi_notes)
        or not isinstance(model_parameters, dict)
    ):
        raise BundleValidationError("drums V14 configuration settings are out of range")
    return DrumsV14ExpertProfile(
        profile_id=profile_id,
        component_id=component_id,
        onset_threshold=float(raw["onset_threshold"]),
        class_thresholds=tuple(float(value) for value in thresholds),
        min_distance_ms=float(raw["min_distance_ms"]),
        segment_duration_seconds=float(raw["segment_duration_seconds"]),
        overlap=float(raw["overlap"]),
        class_to_midi=tuple(midi_notes),
        model_parameters=dict(model_parameters),
        configuration_sha256=hashlib.sha256(profile.configuration.read_bytes()).hexdigest(),
    )
