"""Typed, fail-closed Expert Bass profile for catalog-trained Bass V1.

The Bass V1 worker intentionally shares a five-lane CRNN *topology* with the
Guitar research model.  This module is the semantic boundary that prevents
that implementation detail from becoming a Guitar deployment contract: both
the components and the portable configuration must be Bass-labelled before a
runtime profile can be loaded.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.inference.guitar_neural import GuitarNeuralCharter
from src.model_bundle import BundleValidationError, ModelBundle

FORMAT = "strum-bass-neural-expert-profile/v1"
CAPABILITY = "bass.neural-v1-expert/v1"
PREPROCESSING = "bass-logmel-windows/v1"
EVALUATION_FORMAT = "strum-bass-neural-evaluation/v1"
ONSET_COMPONENT = "bass.onset"
FRET_COMPONENT = "bass.fret"
ONSET_ARCHITECTURE = "FiveLaneOnsetCRNN/v1"
FRET_ARCHITECTURE = "FiveLaneFretClassifier/v1"
EXPECTED_AUDIO = {
    "sample_rate": 22050,
    "n_mels": 128,
    "n_fft": 2048,
    "hop_length": 512,
    "fmin": 30.0,
    "fmax": 8000.0,
}


@dataclass(frozen=True)
class BassNeuralExpertProfile:
    """All runtime-critical settings for one V1 Expert Bass profile."""

    profile_id: str
    onset_component: str
    fret_component: str
    audio: Mapping[str, object]
    onset_model: Mapping[str, object]
    fret_model: Mapping[str, object]
    onset_threshold: float
    peak_min_distance_frames: int
    fret_thresholds: tuple[float, ...]
    note_duration_ms: float
    evaluation_sha256: str
    configuration_sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise BundleValidationError(f"{label} must be a JSON object")
    return raw


def _resolve_bundle_file(bundle: ModelBundle, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"{label} must be a non-empty bundle-relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise BundleValidationError(f"{label} must be bundle-relative")
    resolved = (bundle.root / candidate).resolve()
    try:
        resolved.relative_to(bundle.root)
    except ValueError as error:
        raise BundleValidationError(f"{label} escapes the bundle root") from error
    if not resolved.is_file():
        raise BundleValidationError(f"{label} is unavailable")
    return resolved


def _candidate_config(bundle: ModelBundle) -> dict[str, Any]:
    onset, fret = bundle.component(ONSET_COMPONENT), bundle.component(FRET_COMPONENT)
    if (
        onset is None
        or fret is None
        or onset.config is None
        or fret.config is None
        or onset.architecture != ONSET_ARCHITECTURE
        or fret.architecture != FRET_ARCHITECTURE
        or onset.preprocessing != PREPROCESSING
        or fret.preprocessing != PREPROCESSING
    ):
        raise BundleValidationError("Bass neural profile has incompatible trained components")
    raw = _read_json(onset.config, "Bass neural component configuration")
    if fret.config.read_bytes() != onset.config.read_bytes():
        raise BundleValidationError("Bass neural components must use one identical configuration")
    required = {
        "schema_version",
        "format",
        "instrument",
        "model_implementation",
        "preprocessing",
        "audio",
        "onset_model",
        "fret_model",
        "onset_inference",
    }
    if (
        (set(raw) != required and set(raw) != required | {"training"})
        or raw.get("schema_version") != 1
        or raw.get("format") != "strum-bass-neural-model-config/v1"
        or raw.get("instrument") != "bass"
        or raw.get("model_implementation") != "five-lane-crnn/v1"
        or raw.get("preprocessing") != PREPROCESSING
        or not isinstance(raw.get("audio"), dict)
        or not isinstance(raw.get("onset_model"), dict)
        or not isinstance(raw.get("fret_model"), dict)
        or not isinstance(raw.get("onset_inference"), dict)
        or ("training" in raw and not isinstance(raw["training"], dict))
        or raw["audio"] != EXPECTED_AUDIO
        or raw["onset_model"].get("n_mels") != EXPECTED_AUDIO["n_mels"]
        or raw["fret_model"].get("n_mels") != EXPECTED_AUDIO["n_mels"]
        or raw["fret_model"].get("n_frames") != 22
        or raw["fret_model"].get("n_frets") != 5
    ):
        raise BundleValidationError("Bass neural component configuration is unsupported")
    return raw


def load_bass_neural_candidate(bundle: ModelBundle) -> dict[str, Any]:
    """Validate a Bass V1 experiment before evaluation or packaging."""
    return _candidate_config(bundle)


def load_bass_neural_expert_profile(
    bundle: ModelBundle, profile_id: str
) -> BassNeuralExpertProfile:
    """Load a deployable Bass profile without applying legacy defaults."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.capability != CAPABILITY:
        raise BundleValidationError("profile is not a bass.neural-v1-expert/v1 profile")
    if profile.instruments != ("bass",) or profile.difficulty_policies != ("expert_only",):
        raise BundleValidationError("Bass neural profile must declare Bass Expert-only output")
    if (
        profile.required_components != (ONSET_COMPONENT, FRET_COMPONENT)
        or profile.configuration is None
    ):
        raise BundleValidationError("Bass neural profile requires onset, fret, and configuration")
    candidate = _candidate_config(bundle)
    raw = _read_json(profile.configuration, "Bass neural profile configuration")
    required = {
        "schema_version",
        "format",
        "preprocessing",
        "audio",
        "onset_model",
        "fret_model",
        "onset_threshold",
        "peak_min_distance_frames",
        "fret_thresholds",
        "note_duration_ms",
        "evaluation",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != FORMAT
        or raw.get("preprocessing") != PREPROCESSING
        or raw.get("audio") != candidate["audio"]
        or raw.get("onset_model") != candidate["onset_model"]
        or raw.get("fret_model") != candidate["fret_model"]
    ):
        raise BundleValidationError("Bass neural profile configuration is incompatible")
    onset_threshold = raw.get("onset_threshold")
    min_distance = raw.get("peak_min_distance_frames")
    note_duration = raw.get("note_duration_ms")
    thresholds = raw.get("fret_thresholds")
    if (
        not isinstance(onset_threshold, (int, float))
        or isinstance(onset_threshold, bool)
        or not 0 < onset_threshold <= 1
        or not isinstance(min_distance, int)
        or isinstance(min_distance, bool)
        or min_distance < 1
        or not isinstance(note_duration, (int, float))
        or isinstance(note_duration, bool)
        or not 1 <= note_duration <= 10_000
        or not isinstance(thresholds, list)
        or len(thresholds) != 5
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1
            for value in thresholds
        )
    ):
        raise BundleValidationError("Bass neural profile configuration settings are out of range")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "artifact",
        "sha256",
        "source_bundle_manifest_sha256",
        "minimum_onset_f1",
        "minimum_fret_f1",
    }:
        raise BundleValidationError("Bass neural profile requires a verified evaluation artifact")
    evaluation_path = _resolve_bundle_file(
        bundle, evaluation["artifact"], "Bass evaluation artifact"
    )
    if (
        not isinstance(evaluation.get("sha256"), str)
        or not isinstance(evaluation.get("source_bundle_manifest_sha256"), str)
        or len(evaluation["source_bundle_manifest_sha256"]) != 64
        or _sha256(evaluation_path) != evaluation["sha256"]
    ):
        raise BundleValidationError("Bass evaluation artifact hash does not match")
    if not all(
        isinstance(evaluation[key], (int, float))
        and not isinstance(evaluation[key], bool)
        and 0 < evaluation[key] <= 1
        for key in ("minimum_onset_f1", "minimum_fret_f1")
    ):
        raise BundleValidationError("Bass evaluation thresholds are invalid")
    report = _read_json(evaluation_path, "Bass evaluation artifact")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    required_report = {
        "schema_version",
        "format",
        "model_id",
        "bundle_manifest_sha256",
        "task_view_sha256",
        "split",
        "records_evaluated",
        "alignment_tolerance_ms",
        "metrics",
    }
    if (
        set(report) != required_report
        or report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("model_id") != bundle.model_id
        or report.get("bundle_manifest_sha256") != evaluation["source_bundle_manifest_sha256"]
        or not isinstance(report.get("task_view_sha256"), str)
        or len(report["task_view_sha256"]) != 64
        or report.get("split") != "val"
        or not isinstance(report.get("records_evaluated"), int)
        or report["records_evaluated"] < 1
        or not isinstance(report.get("alignment_tolerance_ms"), (int, float))
        or isinstance(report["alignment_tolerance_ms"], bool)
        or not 1 <= report["alignment_tolerance_ms"] <= 1_000
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool)
            and 0 <= metrics[key] <= 1
            for key in ("onset_f1", "fret_f1", "event_f1")
        )
        or metrics["onset_f1"] < evaluation["minimum_onset_f1"]
        or metrics["fret_f1"] < evaluation["minimum_fret_f1"]
    ):
        raise BundleValidationError("Bass evaluation artifact does not satisfy the deployment gate")
    return BassNeuralExpertProfile(
        profile_id=profile_id,
        onset_component=ONSET_COMPONENT,
        fret_component=FRET_COMPONENT,
        audio=dict(raw["audio"]),
        onset_model=dict(raw["onset_model"]),
        fret_model=dict(raw["fret_model"]),
        onset_threshold=float(onset_threshold),
        peak_min_distance_frames=min_distance,
        fret_thresholds=tuple(float(value) for value in thresholds),
        note_duration_ms=float(note_duration),
        evaluation_sha256=evaluation["sha256"],
        configuration_sha256=_sha256(profile.configuration),
    )


class BassNeuralCharter(GuitarNeuralCharter):
    """Bass-labelled runtime for the shared five-lane CRNN topology."""

    @classmethod
    def from_bundle_profile(
        cls,
        bundle: ModelBundle,
        profile: BassNeuralExpertProfile,
        *,
        device: str | None,
    ) -> BassNeuralCharter:
        onset = bundle.component(profile.onset_component)
        fret = bundle.component(profile.fret_component)
        if onset is None or fret is None or onset.checkpoint is None or fret.checkpoint is None:
            raise BundleValidationError("Bass neural profile components are incomplete")
        return cls(
            onset.checkpoint,
            fret.checkpoint,
            device=device,
            config={
                "onset": {
                    "model": dict(profile.onset_model),
                    "inference": {
                        "peak_threshold": profile.onset_threshold,
                        "peak_min_distance_frames": profile.peak_min_distance_frames,
                    },
                },
                "fret": {"model": dict(profile.fret_model)},
            },
        )
