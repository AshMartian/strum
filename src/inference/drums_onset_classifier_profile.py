"""Typed, evaluation-only profile for catalog-trained Drums V2 classifiers.

This is deliberately separate from :mod:`drums_v14_profile`.  A V2 classifier
labels already-extracted onset windows; it neither detects onsets nor produces
velocity.  It must never be selected as a replacement for the direct V14
Expert Drums profile.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from src.model_bundle import BundleValidationError, ModelBundle

PROFILE_FORMAT = "strum-drums-onset-classifier-evaluation-profile/v1"
MODEL_CONFIG_FORMAT = "strum-drums-onset-classifier-model-config/v1"
CAPABILITY = "drums.onset-classifier-evaluation/v1"
ARCHITECTURE = "OnsetClassifier/v2"
PREPROCESSING = "drums-onset-windows/v1"
EXECUTION_SCOPE = "preprocessed_onset_windows_only"
CLASS_NAMES = ("Kick", "Snare", "HiHat", "HighTom", "Ride", "LowTom", "Crash", "FloorTom")
FINE_MEL_SHAPE = (1, 128, 87)
COARSE_MEL_SHAPE = (1, 128, 44)
CONTEXT_SHAPE = (64,)

_MODEL_DEFAULTS: dict[str, object] = {
    "num_classes": 8,
    "branch_channels": [1, 32, 64, 128, 256],
    "context_size": 4,
    "context_hidden": 64,
    "classifier_hidden": 512,
    "spectral_dim": 32,
    "dropout": 0.3,
    "use_freq_attn": True,
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


@dataclass(frozen=True)
class DrumsOnsetClassifierEvaluationProfile:
    """A verified V2 classifier that can only evaluate prepared windows."""

    profile_id: str
    component_id: str
    model_parameters: Mapping[str, object]
    configuration_sha256: str


def normalize_v2_model_parameters(raw: object) -> dict[str, object]:
    """Return the exact worker-supported V2 constructor settings.

    Worker training exposes only ``onset_classifier_v2``.  Rejecting variant
    architectures here avoids treating an arbitrary legacy classifier as a
    compatible runtime artifact merely because it has similarly named tensors.
    """
    if not isinstance(raw, dict) or set(raw) - set(_MODEL_DEFAULTS):
        raise BundleValidationError("Drums V2 model parameters have unsupported fields")
    values = {**_MODEL_DEFAULTS, **raw}
    if values["num_classes"] != 8 or values["branch_channels"] != [1, 32, 64, 128, 256]:
        raise BundleValidationError("Drums V2 model parameters are not the supported classifier")
    if values["context_size"] != 4 or values["context_hidden"] != 64:
        raise BundleValidationError("Drums V2 context parameters are incompatible")
    if values["classifier_hidden"] != 512 or values["spectral_dim"] != 32:
        raise BundleValidationError("Drums V2 classifier parameters are incompatible")
    if values["dropout"] != 0.3 or values["use_freq_attn"] is not False:
        raise BundleValidationError("Drums V2 spectral parameters are incompatible")
    if any(
        values[name] is not expected
        for name, expected in {
            "use_hpss": False,
            "enhanced_spectral": False,
            "use_contrastive": False,
            "use_aux_head": False,
            "use_dual_head": False,
            "use_lowfreq_branch": False,
            "use_lowfreq_spectral": False,
            "use_crash_flux": False,
        }.items()
    ):
        raise BundleValidationError("Drums V2 optional branches are unsupported")
    if values["projection_dim"] != 128 or values["tom_head_hidden"] != 256 or values["crash_flux_dim"] != 32:
        raise BundleValidationError("Drums V2 auxiliary parameters are incompatible")
    return values


def _load_json(path, message: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(message) from error
    if not isinstance(raw, dict):
        raise BundleValidationError(message)
    return raw


def load_drums_onset_classifier_evaluation_profile(
    bundle: ModelBundle, profile_id: str
) -> DrumsOnsetClassifierEvaluationProfile:
    """Load a bounded Stage-2 evaluator profile, never an auto-chart profile."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.capability != CAPABILITY:
        raise BundleValidationError("profile is not a Drums onset-classifier evaluation profile")
    if profile.instruments != ("drums",) or profile.difficulty_policies != ("evaluation_only",):
        raise BundleValidationError("Drums classifier profile must be evaluation-only")
    if len(profile.required_components) != 1 or profile.configuration is None:
        raise BundleValidationError("Drums classifier profile requires one configured component")
    component_id = profile.required_components[0]
    component = bundle.component(component_id)
    if (
        component is None
        or component.checkpoint is None
        or component.config is None
        or component.architecture != ARCHITECTURE
        or component.preprocessing != PREPROCESSING
    ):
        raise BundleValidationError("Drums classifier profile has incomplete verified assets")
    model_config = _load_json(component.config, "Drums classifier model configuration is unreadable")
    if set(model_config) != {"schema_version", "format", "model_architecture", "preprocessing", "model_parameters"} or (
        model_config.get("schema_version") != 1
        or model_config.get("format") != MODEL_CONFIG_FORMAT
        or model_config.get("model_architecture") != ARCHITECTURE
        or model_config.get("preprocessing") != PREPROCESSING
    ):
        raise BundleValidationError("Drums classifier model configuration has unsupported fields")
    parameters = normalize_v2_model_parameters(model_config.get("model_parameters"))
    configuration = _load_json(profile.configuration, "Drums classifier profile configuration is unreadable")
    required = {
        "schema_version", "format", "model_architecture", "preprocessing", "execution_scope",
        "class_names", "fine_mel_shape", "coarse_mel_shape", "context_shape", "output_contract",
        "auto_chart_status",
    }
    if (
        set(configuration) != required
        or configuration.get("schema_version") != 1
        or configuration.get("format") != PROFILE_FORMAT
        or configuration.get("model_architecture") != ARCHITECTURE
        or configuration.get("preprocessing") != PREPROCESSING
        or configuration.get("execution_scope") != EXECUTION_SCOPE
        or configuration.get("class_names") != list(CLASS_NAMES)
        or configuration.get("fine_mel_shape") != list(FINE_MEL_SHAPE)
        or configuration.get("coarse_mel_shape") != list(COARSE_MEL_SHAPE)
        or configuration.get("context_shape") != list(CONTEXT_SHAPE)
        or configuration.get("output_contract") != "sigmoid_8_class_probabilities/v1"
        or configuration.get("auto_chart_status") != "not_supported"
    ):
        raise BundleValidationError("Drums classifier profile configuration has unsupported fields")
    return DrumsOnsetClassifierEvaluationProfile(
        profile_id=profile_id,
        component_id=component_id,
        model_parameters=parameters,
        configuration_sha256=hashlib.sha256(profile.configuration.read_bytes()).hexdigest(),
    )
