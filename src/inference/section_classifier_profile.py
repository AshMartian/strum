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
import math
import pickle
import re
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
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class SectionClassifierCandidate:
    """Verified tensor-only experimental component, not a chart profile."""

    instrument: str
    component_id: str
    checkpoint: Path
    configuration_sha256: str
    task_view_sha256: str


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


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256.fullmatch(value))


def _finite_number(value: object, *, minimum: float, maximum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    numeric = float(value)
    return math.isfinite(numeric) and numeric >= minimum and (maximum is None or numeric <= maximum)


def _same_metric(observed: object, expected: float) -> bool:
    return _finite_number(observed, minimum=0) and math.isclose(
        float(observed), expected, rel_tol=1e-9, abs_tol=1e-12
    )


def _require_candidate_config(bundle: ModelBundle, instrument: str) -> tuple[Path, str, str]:
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
        "task_view_sha256",
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
        or not _is_sha256(raw.get("task_view_sha256"))
    ):
        raise BundleValidationError("Section candidate configuration is unsupported")
    return component.checkpoint, _sha256(component.config), raw["task_view_sha256"]


def load_section_classifier_state(checkpoint: Path) -> dict[str, object]:
    """Load and shape-check only tensor weights from a known candidate checkpoint."""
    import torch  # noqa: PLC0415

    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except (
        OSError,
        RuntimeError,
        ValueError,
        EOFError,
        UnicodeDecodeError,
        pickle.UnpicklingError,
    ) as error:
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
    checkpoint, configuration_sha256, task_view_sha256 = _require_candidate_config(
        bundle, instrument
    )
    load_section_classifier_state(checkpoint)
    return SectionClassifierCandidate(
        instrument=instrument,
        component_id=_instrument_component(instrument),
        checkpoint=checkpoint,
        configuration_sha256=configuration_sha256,
        task_view_sha256=task_view_sha256,
    )


def _require_evaluation_report(
    report: dict[str, Any],
    *,
    bundle: ModelBundle,
    instrument: str,
    expected_bundle_manifest_sha256: str | None = None,
    expected_task_view_sha256: str | None = None,
) -> dict[str, Any]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    calibration = report.get("calibration") if isinstance(report.get("calibration"), dict) else {}
    per_class = report.get("per_class") if isinstance(report.get("per_class"), dict) else {}
    metric_evidence = (
        report.get("metric_evidence") if isinstance(report.get("metric_evidence"), dict) else {}
    )
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
        "metric_evidence",
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
        or not _is_sha256(report.get("task_view_sha256"))
        or (
            expected_task_view_sha256 is not None
            and report["task_view_sha256"] != expected_task_view_sha256
        )
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
        or isinstance(calibration["records_evaluated"], bool)
        or calibration["records_evaluated"] < 1
        or not _finite_number(calibration.get("temperature"), minimum=0.05, maximum=10)
        or not _finite_number(calibration.get("selection_nll"), minimum=0)
        or set(metrics) != {"accuracy", "macro_f1", "nll", "brier", "expected_calibration_error"}
        or not _finite_number(metrics.get("accuracy"), minimum=0, maximum=1)
        or not _finite_number(metrics.get("macro_f1"), minimum=0, maximum=1)
        or not _finite_number(metrics.get("nll"), minimum=0)
        or not _finite_number(metrics.get("brier"), minimum=0, maximum=2)
        or not _finite_number(metrics.get("expected_calibration_error"), minimum=0, maximum=1)
        or set(per_class) != set(LABELS)
        or any(
            not isinstance(item, dict)
            or set(item) != {"support", "precision", "recall", "f1"}
            or not isinstance(item["support"], int)
            or item["support"] < 0
            or not all(
                _finite_number(item[key], minimum=0, maximum=1)
                for key in ("precision", "recall", "f1")
            )
            for item in per_class.values()
        )
        or set(metric_evidence) != {"nll_sum", "brier_sum", "confidence_bins"}
        or not _finite_number(metric_evidence.get("nll_sum"), minimum=0)
        or not _finite_number(metric_evidence.get("brier_sum"), minimum=0)
        or not isinstance(metric_evidence.get("confidence_bins"), list)
        or len(metric_evidence["confidence_bins"]) != 10
        or any(
            not isinstance(item, dict)
            or set(item) != {"count", "confidence_sum", "correct_count"}
            or not isinstance(item["count"], int)
            or isinstance(item["count"], bool)
            or item["count"] < 0
            or not _finite_number(item["confidence_sum"], minimum=0, maximum=item["count"])
            or not isinstance(item["correct_count"], int)
            or isinstance(item["correct_count"], bool)
            or not 0 <= item["correct_count"] <= item["count"]
            for item in metric_evidence["confidence_bins"]
        )
    ):
        raise BundleValidationError("Section held-out evaluation report is invalid")
    matrix = report["confusion_matrix"]
    records_evaluated = report["records_evaluated"]
    if sum(sum(row) for row in matrix) != records_evaluated:
        raise BundleValidationError("Section held-out evaluation report count is inconsistent")
    f1_values: list[float] = []
    for index, label in enumerate(LABELS):
        true_positive = matrix[index][index]
        false_positive = sum(row[index] for row in matrix) - true_positive
        false_negative = sum(matrix[index]) - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        if (
            per_class[label]["support"] != sum(matrix[index])
            or not _same_metric(per_class[label]["precision"], precision)
            or not _same_metric(per_class[label]["recall"], recall)
            or not _same_metric(per_class[label]["f1"], f1)
        ):
            raise BundleValidationError(
                "Section held-out evaluation report classes are inconsistent"
            )
        f1_values.append(f1)
    accuracy = sum(matrix[index][index] for index in range(len(LABELS))) / records_evaluated
    if not _same_metric(metrics["accuracy"], accuracy) or not _same_metric(
        metrics["macro_f1"], sum(f1_values) / len(f1_values)
    ):
        raise BundleValidationError("Section held-out evaluation report metrics are inconsistent")
    bins = metric_evidence["confidence_bins"]
    if sum(item["count"] for item in bins) != records_evaluated:
        raise BundleValidationError("Section held-out evaluation evidence count is inconsistent")
    nll = float(metric_evidence["nll_sum"]) / records_evaluated
    brier = float(metric_evidence["brier_sum"]) / records_evaluated
    ece = sum(
        (item["count"] / records_evaluated)
        * abs(item["correct_count"] / item["count"] - float(item["confidence_sum"]) / item["count"])
        for item in bins
        if item["count"]
    )
    if not (
        _same_metric(metrics["nll"], nll)
        and _same_metric(metrics["brier"], brier)
        and _same_metric(metrics["expected_calibration_error"], ece)
    ):
        raise BundleValidationError("Section held-out evaluation evidence is inconsistent")
    return {
        "metrics": metrics,
        "calibration": calibration,
        "task_view_sha256": report["task_view_sha256"],
    }


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
        expected_task_view_sha256=candidate.task_view_sha256,
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
