"""Strict, non-deployable loader for catalog-trained fret-mapper components.

This is deliberately a candidate boundary, not an auto-chart handler.  It
proves a mapper component was produced with the exact declared architecture
and a ``weights_only``-loadable payload.  A future composed Guitar/Bass
profile must additionally bind an onset source, a Basic Pitch runtime version,
Viterbi policy, and complete held-out chart evaluation before it may execute.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.model_bundle import BundleValidationError, ModelBundle
from src.models.fret_mapper import FretMapperMLP

FORMAT = "strum-fret-mapper-model-config/v1"
WEIGHTS_FORMAT = "strum-fret-mapper-weights/v1"
PREPROCESSING = "basic-pitch-onset-features/v1"
MODEL_IMPLEMENTATION = "FretMapperMLP/v1"
FEATURE_DIMENSION = 95
OUTPUT_DIMENSION = 5


@dataclass(frozen=True)
class FretMapperCandidate:
    """A validated mapper component that remains unavailable to chart run."""

    instrument: str
    component_id: str
    model: FretMapperMLP
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    basic_pitch: Mapping[str, object]


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise BundleValidationError(f"{label} must be a JSON object")
    return raw


def _finite_float(value: object, label: str, *, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= float(value) <= maximum
    ):
        raise BundleValidationError(f"{label} is invalid")
    return float(value)


def _candidate_config(bundle: ModelBundle, instrument: str) -> tuple[str, dict[str, Any]]:
    if instrument not in {"guitar", "bass"}:
        raise BundleValidationError("fret-mapper instrument is unsupported")
    component_id = f"fret_mapper.{instrument}"
    component = bundle.component(component_id)
    if (
        component is None
        or component.checkpoint is None
        or component.config is None
        or component.architecture != MODEL_IMPLEMENTATION
        or component.preprocessing != PREPROCESSING
    ):
        raise BundleValidationError("fret-mapper component is incompatible")
    raw = _read_json(component.config, "fret-mapper component configuration")
    required = {
        "schema_version",
        "format",
        "instrument",
        "pipeline_id",
        "model_implementation",
        "preprocessing",
        "feature_dimension",
        "label_schema",
        "basic_pitch",
        "model",
        "training",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != FORMAT
        or raw.get("instrument") != instrument
        or raw.get("pipeline_id") != f"strum.fret-mapper/{instrument}/v1"
        or raw.get("model_implementation") != MODEL_IMPLEMENTATION
        or raw.get("preprocessing") != PREPROCESSING
        or raw.get("feature_dimension") != FEATURE_DIMENSION
        or raw.get("label_schema") != "five-lane-fret-mapper-midi/v1"
        or not isinstance(raw.get("basic_pitch"), dict)
        or not isinstance(raw.get("model"), dict)
        or not isinstance(raw.get("training"), dict)
    ):
        raise BundleValidationError("fret-mapper component configuration is unsupported")
    basic_pitch = raw["basic_pitch"]
    if (
        set(basic_pitch)
        != {"distribution", "version", "onset_threshold", "frame_threshold", "min_note_length"}
        or basic_pitch.get("distribution") != "basic-pitch"
        or not isinstance(basic_pitch.get("version"), str)
        or not basic_pitch["version"]
        or not isinstance(basic_pitch.get("min_note_length"), int)
        or isinstance(basic_pitch["min_note_length"], bool)
        or basic_pitch["min_note_length"] < 1
    ):
        raise BundleValidationError("fret-mapper Basic Pitch contract is unsupported")
    _finite_float(
        basic_pitch.get("onset_threshold"), "Basic Pitch onset threshold", minimum=0, maximum=1
    )
    _finite_float(
        basic_pitch.get("frame_threshold"), "Basic Pitch frame threshold", minimum=0, maximum=1
    )
    model = raw["model"]
    if (
        set(model) != {"format", "input_dimension", "hidden", "output_dimension", "dropout"}
        or model.get("format") != "strum-fret-mapper-mlp/v1"
        or model.get("input_dimension") != FEATURE_DIMENSION
        or not isinstance(model.get("hidden"), int)
        or isinstance(model["hidden"], bool)
        or not 1 <= model["hidden"] <= 256
        or model.get("output_dimension") != OUTPUT_DIMENSION
    ):
        raise BundleValidationError("fret-mapper architecture contract is unsupported")
    _finite_float(model.get("dropout"), "fret-mapper dropout", minimum=0, maximum=1)
    return component_id, raw


def _tensor(value: object, label: str, *, shape: tuple[int, ...] | None = None) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise BundleValidationError(f"{label} must be a CPU tensor")
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise BundleValidationError(f"{label} must contain finite floating-point values")
    if shape is not None and tuple(value.shape) != shape:
        raise BundleValidationError(f"{label} has an incompatible shape")
    return value.detach().to(dtype=torch.float32, device="cpu").contiguous()


def load_fret_mapper_candidate(bundle: ModelBundle, instrument: str) -> FretMapperCandidate:
    """Load a typed mapper component through PyTorch's tensor-only loader.

    The returned candidate must not be passed to a chart handler.  This exists
    so evaluation/package code has a fail-closed checkpoint boundary instead
    of treating a raw training artifact as executable.
    """
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise BundleValidationError("fret-mapper candidate bundle failed verification")
    component_id, config = _candidate_config(bundle, instrument)
    component = bundle.component(component_id)
    assert component is not None and component.checkpoint is not None
    try:
        raw = torch.load(component.checkpoint, map_location="cpu", weights_only=True)
    except (OSError, pickle.UnpicklingError, RuntimeError, ValueError, TypeError) as error:
        raise BundleValidationError("fret-mapper checkpoint is not tensor-only loadable") from error
    if not isinstance(raw, dict) or set(raw) != {
        "format",
        "model_state",
        "input_dimension",
        "hidden",
        "output_dimension",
        "feature_mean",
        "feature_std",
    }:
        raise BundleValidationError("fret-mapper checkpoint has an unsupported payload")
    model_config = config["model"]
    if (
        raw.get("format") != WEIGHTS_FORMAT
        or raw.get("input_dimension") != FEATURE_DIMENSION
        or raw.get("input_dimension") != model_config["input_dimension"]
        or raw.get("hidden") != model_config["hidden"]
        or raw.get("output_dimension") != OUTPUT_DIMENSION
        or raw.get("output_dimension") != model_config["output_dimension"]
        or not isinstance(raw.get("model_state"), dict)
    ):
        raise BundleValidationError("fret-mapper checkpoint architecture does not match its config")
    feature_mean = _tensor(
        raw.get("feature_mean"), "fret-mapper feature mean", shape=(FEATURE_DIMENSION,)
    )
    feature_std = _tensor(
        raw.get("feature_std"), "fret-mapper feature standard deviation", shape=(FEATURE_DIMENSION,)
    )
    if torch.any(feature_std <= 0).item():
        raise BundleValidationError("fret-mapper feature standard deviation must be positive")
    model = FretMapperMLP(
        in_dim=FEATURE_DIMENSION,
        hidden=model_config["hidden"],
        out_dim=OUTPUT_DIMENSION,
        p_drop=model_config["dropout"],
    )
    try:
        model.load_state_dict(raw["model_state"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise BundleValidationError(
            "fret-mapper checkpoint state does not match its architecture"
        ) from error
    model.eval()
    return FretMapperCandidate(
        instrument=instrument,
        component_id=component_id,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        basic_pitch=dict(config["basic_pitch"]),
    )
