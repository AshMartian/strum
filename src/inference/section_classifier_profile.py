"""Typed, tensor-only SectionClassifier candidate and evaluation profile.

Section classification is deliberately *not* an auto-chart profile.  These
helpers are the narrow bridge between a catalog-trained experiment and the
separate calibration/ablation work needed before a SectionRouter may influence
an executable Guitar or Bass chart profile.  They never consult legacy
checkpoint locations and they load PyTorch weights with ``weights_only=True``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.model_bundle import BundleValidationError, ModelBundle
from src.models.section_classifier import SectionClassifier
from src.section_frontend import ROUTER_FEATURE_EXTRACTOR

MODEL_CONFIG_FORMAT = "strum-section-classifier-model-config/v1"
PROFILE_FORMAT = "strum-section-classifier-evaluation-profile/v1"
EVALUATION_FORMAT = "strum-section-classifier-held-out-evaluation/v1"
CAPABILITY = "section.classifier-evaluation/v1"
ARCHITECTURE = "SectionClassifier/v1"
PREPROCESSING = "section-logmel-librosa-router-windows/v1"
EXECUTION_SCOPE = "router_window_classification_only"
LABELS = ("silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed")


@dataclass(frozen=True)
class SectionClassifierCandidate:
    """Verified tensor-only experimental component, not a chart profile."""

    instrument: str
    component_id: str
    checkpoint: Path
    configuration_sha256: str


@dataclass(frozen=True)
class SectionClassifierEvaluationProfile:
    """A packaged held-out evaluator that remains unavailable to chart run."""

    profile_id: str
    instrument: str
    component_id: str
    calibration_temperature: float
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
        resolved.relative_to(bundle.root.resolve())
    except ValueError as error:
        raise BundleValidationError(f"{label} escapes the bundle root") from error
    if not resolved.is_file():
        raise BundleValidationError(f"{label} is unavailable")
    return resolved


def _instrument_component(instrument: str) -> str:
    if instrument not in {"guitar", "bass"}:
        raise BundleValidationError("Section profile instrument must be guitar or bass")
    return f"section_classifier.{instrument}"


def _require_candidate_config(bundle: ModelBundle, instrument: str) -> tuple[Path, str]:
    component_id = _instrument_component(instrument)
    component = bundle.component(component_id)
    if (
        component is None
        or component.checkpoint is None
        or component.config is None
        or component.architecture != ARCHITECTURE
        or component.preprocessing != PREPROCESSING
    ):
        raise BundleValidationError("Section candidate has incomplete verified assets")
    raw = _read_json(component.config, "Section candidate configuration")
    required = {
        "schema_version",
        "format",
        "instrument",
        "pipeline_id",
        "model_implementation",
        "preprocessing",
        "labels",
        "window_seconds",
        "hop_seconds",
        "mel_shape",
        "feature_extractor",
        "runtime_profile",
        "training",
    }
    expected_runtime = {
        "format": "strum-section-router-deployment-requirements/v1",
        "status": "not_packageable",
        "reason": "section_router_execution_and_held_out_evaluation_not_proven",
        "requirements": [
            "section_router_profile_loader_tensor_only",
            "held_out_section_calibration_evaluation",
            "held_out_chart_impact_ablation",
            f"composed_{instrument}_chart_profile_contract",
        ],
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != MODEL_CONFIG_FORMAT
        or raw.get("instrument") != instrument
        or raw.get("pipeline_id") != f"strum.section-classifier/{instrument}/v1"
        or raw.get("model_implementation") != ARCHITECTURE
        or raw.get("preprocessing") != PREPROCESSING
        or raw.get("labels") != list(LABELS)
        or raw.get("window_seconds") != 2.0
        or raw.get("hop_seconds") != 1.0
        or raw.get("mel_shape") != [128, 87]
        or raw.get("feature_extractor") != ROUTER_FEATURE_EXTRACTOR
        or raw.get("runtime_profile") != expected_runtime
        or not isinstance(raw.get("training"), dict)
    ):
        raise BundleValidationError("Section candidate configuration is unsupported")
    return component.checkpoint, _sha256(component.config)


def load_section_classifier_state(checkpoint: Path) -> dict[str, object]:
    """Load and shape-check only tensor weights from a known candidate checkpoint."""
    import torch  # noqa: PLC0415

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise BundleValidationError("Section candidate checkpoint is not tensor-only") from error
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        )
    ):
        raise BundleValidationError("Section candidate checkpoint has no tensor state_dict")
    model = SectionClassifier()
    try:
        model.load_state_dict(dict(state), strict=True)
    except (RuntimeError, TypeError) as error:
        raise BundleValidationError("Section candidate tensor state is incompatible") from error
    return dict(state)


def load_section_classifier_candidate(
    bundle: ModelBundle, instrument: str
) -> SectionClassifierCandidate:
    """Validate one raw SectionClassifier component before held-out evaluation."""
    checkpoint, configuration_sha256 = _require_candidate_config(bundle, instrument)
    load_section_classifier_state(checkpoint)
    return SectionClassifierCandidate(
        instrument=instrument,
        component_id=_instrument_component(instrument),
        checkpoint=checkpoint,
        configuration_sha256=configuration_sha256,
    )


def _require_evaluation_report(
    report: dict[str, Any],
    *,
    bundle: ModelBundle,
    instrument: str,
    expected_bundle_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    calibration = report.get("calibration") if isinstance(report.get("calibration"), dict) else {}
    per_class = report.get("per_class") if isinstance(report.get("per_class"), dict) else {}
    required = {
        "schema_version",
        "format",
        "model_id",
        "bundle_manifest_sha256",
        "task_view_sha256",
        "instrument",
        "component_id",
        "split",
        "records_evaluated",
        "calibration",
        "metrics",
        "per_class",
        "confusion_matrix",
    }
    if bundle.manifest_path is None:
        raise BundleValidationError("Section candidate has no bundle manifest")
    expected_manifest_sha256 = expected_bundle_manifest_sha256 or _sha256(bundle.manifest_path)
    if (
        set(report) != required
        or report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("model_id") != bundle.model_id
        or report.get("bundle_manifest_sha256") != expected_manifest_sha256
        or not isinstance(report.get("task_view_sha256"), str)
        or len(report["task_view_sha256"]) != 64
        or report.get("instrument") != instrument
        or report.get("component_id") != _instrument_component(instrument)
        or report.get("split") != "test"
        or not isinstance(report.get("records_evaluated"), int)
        or isinstance(report["records_evaluated"], bool)
        or report["records_evaluated"] < 1
        or not isinstance(report.get("confusion_matrix"), list)
        or len(report["confusion_matrix"]) != len(LABELS)
        or any(
            not isinstance(row, list)
            or len(row) != len(LABELS)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in row
            )
            for row in report["confusion_matrix"]
        )
        or set(calibration)
        != {
            "method",
            "selection_split",
            "records_evaluated",
            "temperature",
            "selection_nll",
        }
        or calibration.get("method") != "temperature-grid-nll/v1"
        or calibration.get("selection_split") != "val"
        or not isinstance(calibration.get("records_evaluated"), int)
        or calibration["records_evaluated"] < 1
        or not isinstance(calibration.get("temperature"), (int, float))
        or isinstance(calibration["temperature"], bool)
        or not 0.05 <= calibration["temperature"] <= 10
        or not isinstance(calibration.get("selection_nll"), (int, float))
        or isinstance(calibration["selection_nll"], bool)
        or calibration["selection_nll"] < 0
        or set(metrics) != {"accuracy", "macro_f1", "nll", "brier", "expected_calibration_error"}
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool)
            and (0 <= metrics[key] <= 1 if key not in {"nll"} else metrics[key] >= 0)
            for key in metrics
        )
        or set(per_class) != set(LABELS)
        or any(
            not isinstance(item, dict)
            or set(item) != {"support", "precision", "recall", "f1"}
            or not isinstance(item["support"], int)
            or item["support"] < 0
            or not all(
                isinstance(item[key], (int, float))
                and not isinstance(item[key], bool)
                and 0 <= item[key] <= 1
                for key in ("precision", "recall", "f1")
            )
            for item in per_class.values()
        )
    ):
        raise BundleValidationError("Section held-out evaluation report is invalid")
    return {"metrics": metrics, "calibration": calibration}


def load_section_classifier_evaluation_profile(
    bundle: ModelBundle, profile_id: str
) -> SectionClassifierEvaluationProfile:
    """Load a calibrated evaluator profile; it is intentionally not chart-runnable."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.capability != CAPABILITY:
        raise BundleValidationError("profile is not a Section classifier evaluation profile")
    if (
        len(profile.instruments) != 1
        or profile.instruments[0] not in {"guitar", "bass"}
        or profile.difficulty_policies != ("evaluation_only",)
        or profile.required_components != (_instrument_component(profile.instruments[0]),)
        or profile.configuration is None
    ):
        raise BundleValidationError("Section evaluation profile scope is invalid")
    instrument = profile.instruments[0]
    candidate = load_section_classifier_candidate(bundle, instrument)
    raw = _read_json(profile.configuration, "Section evaluation profile configuration")
    required = {
        "schema_version",
        "format",
        "instrument",
        "component_id",
        "model_implementation",
        "preprocessing",
        "feature_extractor",
        "execution_scope",
        "auto_chart_status",
        "evaluation",
    }
    if (
        set(raw) != required
        or raw.get("schema_version") != 1
        or raw.get("format") != PROFILE_FORMAT
        or raw.get("instrument") != instrument
        or raw.get("component_id") != candidate.component_id
        or raw.get("model_implementation") != ARCHITECTURE
        or raw.get("preprocessing") != PREPROCESSING
        or raw.get("feature_extractor") != ROUTER_FEATURE_EXTRACTOR
        or raw.get("execution_scope") != EXECUTION_SCOPE
        or raw.get("auto_chart_status") != "not_supported"
    ):
        raise BundleValidationError("Section evaluation profile configuration is incompatible")
    evaluation = raw.get("evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "artifact",
        "sha256",
        "source_bundle_manifest_sha256",
        "minimum_accuracy",
        "maximum_expected_calibration_error",
    }:
        raise BundleValidationError("Section evaluation profile requires a held-out artifact")
    artifact = _resolve_bundle_file(bundle, evaluation["artifact"], "Section evaluation artifact")
    if (
        not isinstance(evaluation.get("sha256"), str)
        or _sha256(artifact) != evaluation["sha256"]
        or not isinstance(evaluation.get("source_bundle_manifest_sha256"), str)
        or len(evaluation["source_bundle_manifest_sha256"]) != 64
        or not isinstance(evaluation.get("minimum_accuracy"), (int, float))
        or not 0 < evaluation["minimum_accuracy"] <= 1
        or not isinstance(evaluation.get("maximum_expected_calibration_error"), (int, float))
        or not 0 <= evaluation["maximum_expected_calibration_error"] <= 1
    ):
        raise BundleValidationError("Section evaluation profile gates are invalid")
    report = _read_json(artifact, "Section evaluation artifact")
    validated = _require_evaluation_report(
        report,
        bundle=bundle,
        instrument=instrument,
        expected_bundle_manifest_sha256=evaluation["source_bundle_manifest_sha256"],
    )
    if (
        report["bundle_manifest_sha256"] != evaluation["source_bundle_manifest_sha256"]
        or validated["metrics"]["accuracy"] < evaluation["minimum_accuracy"]
        or validated["metrics"]["expected_calibration_error"]
        > evaluation["maximum_expected_calibration_error"]
    ):
        raise BundleValidationError("Section evaluation artifact does not satisfy the gate")
    return SectionClassifierEvaluationProfile(
        profile_id=profile_id,
        instrument=instrument,
        component_id=candidate.component_id,
        calibration_temperature=float(validated["calibration"]["temperature"]),
        evaluation_sha256=evaluation["sha256"],
        configuration_sha256=_sha256(profile.configuration),
    )
