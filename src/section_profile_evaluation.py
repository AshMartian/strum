"""Held-out calibration and evaluation-only packaging for SectionClassifier.

This module intentionally stops before runtime routing.  It establishes a
reproducible, song-disjoint classifier measurement and packages it only as an
``evaluation_only`` profile.  A router-on/off chart-impact ablation and a
composed executable chart profile remain separate release gates.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src import PROJECT_ROOT
from src.catalog_task_manifest import MANIFEST_FORMAT
from src.inference.section_classifier_profile import (
    ARCHITECTURE,
    CAPABILITY,
    EVALUATION_FORMAT,
    EXECUTION_SCOPE,
    LABELS,
    PREPROCESSING,
    PROFILE_FORMAT,
    SectionClassifierCandidate,
    _require_evaluation_report,
    load_section_classifier_candidate,
    load_section_classifier_state,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.models.section_classifier import SectionClassifier
from src.section_frontend import ROUTER_FEATURE_EXTRACTOR
from src.section_worker_training import DEPLOYMENT_STATUS


class SectionProfileEvaluationError(ValueError):
    """Raised for invalid or incomplete Section promotion inputs."""


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
        raise SectionProfileEvaluationError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise SectionProfileEvaluationError(f"{label} must be a JSON object")
    return raw


def _require_task_view(task_view: dict[str, Any], instrument: str) -> None:
    task = task_view.get("task")
    if instrument not in {"guitar", "bass"}:
        raise SectionProfileEvaluationError("Section evaluation instrument is invalid")
    if (
        task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != f"section_{instrument}"
        or task.get("pipeline_id") != f"strum.section-classifier/{instrument}/v1"
        or task.get("instrument") != instrument
        or task.get("label_schema")
        != {
            "id": "midi-section-events/v1",
            "track_prefixes": [f"PART {instrument.upper()}"],
            "difficulty_encoding": "not-applicable",
        }
    ):
        raise SectionProfileEvaluationError(
            "Section evaluation requires its exact catalog task view"
        )


def _run_script(command: list[str]) -> None:
    try:
        subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise SectionProfileEvaluationError("Section held-out cache preparation failed") from error


def _materialize_held_out_cache(
    *, task_view_path: Path, catalog_root: Path, cache_dir: Path
) -> dict[str, int]:
    """Rebuild validation/test windows from private approved catalog assets."""
    labels_path = cache_dir.parent / "section-labels.json"
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_catalog_section_labels.py"),
            "--manifest",
            str(task_view_path),
            "--catalog-root",
            str(catalog_root),
            "--out",
            str(labels_path),
        ]
    )
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "preprocess_section_windows.py"),
            "--labels",
            str(labels_path),
            "--catalog-manifest",
            str(task_view_path),
            "--catalog-root",
            str(catalog_root),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    counts: dict[str, int] = {}
    for split in ("val", "test"):
        mel_path = cache_dir / f"{split}_section_mel.npy"
        label_path = cache_dir / f"{split}_section_label.npy"
        try:
            mel = np.load(mel_path, mmap_mode="r")
            labels = np.load(label_path, mmap_mode="r")
        except (OSError, ValueError) as error:
            raise SectionProfileEvaluationError("Section held-out cache is missing") from error
        if (
            mel.dtype != np.dtype(np.float32)
            or mel.ndim != 3
            or mel.shape[1:] != (128, 87)
            or labels.ndim != 1
            or len(mel) != len(labels)
            or not len(labels)
            or np.any(labels < 0)
            or np.any(labels >= len(LABELS))
        ):
            raise SectionProfileEvaluationError("Section held-out cache is incompatible")
        counts[split] = len(labels)
    return counts


def _logits(
    candidate: SectionClassifierCandidate, features: np.ndarray, *, device: str
) -> np.ndarray:
    import torch  # noqa: PLC0415

    state = load_section_classifier_state(candidate.checkpoint)
    model = SectionClassifier().to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for offset in range(0, len(features), 256):
            # Memmapped caches are intentionally read-only; make an owned
            # tensor input rather than relying on PyTorch's writable-array
            # assumption.
            batch = torch.from_numpy(
                np.array(features[offset : offset + 256], dtype=np.float32, copy=True)
            ).unsqueeze(1)
            values = model(batch.to(device)).cpu().numpy()
            outputs.append(np.asarray(values, dtype=np.float64))
    return np.concatenate(outputs, axis=0)


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    values = logits / temperature
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def _select_temperature(logits: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    candidates = np.arange(0.5, 3.01, 0.1)
    choices = [(float(value), _nll(_softmax(logits, float(value)), labels)) for value in candidates]
    return min(choices, key=lambda item: (item[1], item[0]))


def _metrics(
    probs: np.ndarray, labels: np.ndarray
) -> tuple[
    dict[str, float],
    dict[str, dict[str, float | int]],
    list[list[int]],
    dict[str, object],
]:
    predictions = probs.argmax(axis=1)
    matrix = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for expected, predicted in zip(labels, predictions, strict=True):
        matrix[int(expected), int(predicted)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(LABELS):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum() - tp)
        fn = int(matrix[index, :].sum() - tp)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[name] = {
            "support": int(matrix[index, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1_values.append(f1)
    confidence = probs.max(axis=1)
    correct = predictions == labels
    confidence_bins: list[dict[str, float | int]] = []
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        mask = (confidence >= lower) & (
            (confidence < upper) if upper < 1 else (confidence <= upper)
        )
        confidence_bins.append(
            {
                "count": int(mask.sum()),
                "confidence_sum": float(confidence[mask].sum()),
                "correct_count": int(correct[mask].sum()),
            }
        )
    ece = sum(
        (item["count"] / len(labels))
        * abs(item["correct_count"] / item["count"] - item["confidence_sum"] / item["count"])
        for item in confidence_bins
        if item["count"]
    )
    one_hot = np.eye(len(LABELS), dtype=np.float64)[labels]
    nll_sum = float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).sum())
    brier_sum = float(np.square(probs - one_hot).sum())
    summary = {
        "accuracy": float(correct.mean()),
        "macro_f1": float(np.mean(f1_values)),
        "nll": nll_sum / len(labels),
        "brier": brier_sum / len(labels),
        "expected_calibration_error": float(ece),
    }
    return (
        summary,
        per_class,
        matrix.tolist(),
        {
            "nll_sum": nll_sum,
            "brier_sum": brier_sum,
            "confidence_bins": confidence_bins,
        },
    )


def evaluate_section_candidate(
    *,
    bundle_root: Path,
    task_view_path: Path,
    catalog_root: Path,
    output_path: Path,
    instrument: str,
    device: str = "cpu",
) -> dict[str, object]:
    """Calibrate on validation windows and report metrics only on held-out test windows."""
    if device not in {"cpu", "cuda", "mps"}:
        raise SectionProfileEvaluationError("Section evaluation device is invalid")
    if output_path.exists():
        raise SectionProfileEvaluationError("Section evaluation output must not already exist")
    task_view = _read_json(task_view_path, "Section task view")
    _require_task_view(task_view, instrument)
    bundle = load_model_bundle(bundle_root, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise SectionProfileEvaluationError("Section candidate bundle failed verification")
    try:
        candidate = load_section_classifier_candidate(bundle, instrument)
    except BundleValidationError as error:
        raise SectionProfileEvaluationError(
            "Section candidate is not tensor-only and compatible"
        ) from error
    if bundle.manifest_path is None:
        raise SectionProfileEvaluationError("Section candidate bundle has no manifest")
    task_view_sha256 = _sha256(task_view_path)
    if task_view_sha256 != candidate.task_view_sha256:
        raise SectionProfileEvaluationError(
            "Section task view does not match the candidate training lineage"
        )
    with tempfile.TemporaryDirectory(prefix="strum-section-held-out-") as temporary:
        cache_dir = Path(temporary) / "cache"
        counts = _materialize_held_out_cache(
            task_view_path=task_view_path, catalog_root=catalog_root, cache_dir=cache_dir
        )
        val_features = np.load(cache_dir / "val_section_mel.npy", mmap_mode="r")
        val_labels = np.asarray(
            np.load(cache_dir / "val_section_label.npy", mmap_mode="r"), dtype=np.int64
        )
        test_features = np.load(cache_dir / "test_section_mel.npy", mmap_mode="r")
        test_labels = np.asarray(
            np.load(cache_dir / "test_section_label.npy", mmap_mode="r"), dtype=np.int64
        )
        temperature, selection_nll = _select_temperature(
            _logits(candidate, val_features, device=device), val_labels
        )
        metrics, per_class, matrix, metric_evidence = _metrics(
            _softmax(_logits(candidate, test_features, device=device), temperature), test_labels
        )
    report = {
        "schema_version": 1,
        "format": EVALUATION_FORMAT,
        "model_id": bundle.model_id,
        "bundle_manifest_sha256": _sha256(bundle.manifest_path),
        "task_view_sha256": task_view_sha256,
        "instrument": instrument,
        "component_id": candidate.component_id,
        "split": "test",
        "records_evaluated": counts["test"],
        "calibration": {
            "method": "temperature-grid-nll/v1",
            "selection_split": "val",
            "records_evaluated": counts["val"],
            "temperature": temperature,
            "selection_nll": selection_nll,
        },
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "metric_evidence": metric_evidence,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _require_candidate_experiment(
    experiment_dir: Path, instrument: str
) -> tuple[Path, dict[str, Any], str]:
    experiment = _read_json(experiment_dir / "experiment.json", "Section experiment")
    expected_pipeline = {"id": f"strum.section-classifier/{instrument}", "version": 1}
    if (
        experiment.get("format") != "strum-experiment/v1"
        or experiment.get("lifecycle") != "completed"
        or experiment.get("pipeline") != expected_pipeline
        or experiment.get("deployment_status") != DEPLOYMENT_STATUS
    ):
        raise SectionProfileEvaluationError(
            "Section experiment is not a packageable worker artifact"
        )
    bundle_root = experiment_dir / "bundle"
    bundle = load_model_bundle(bundle_root, check_files=True)
    if bundle.manifest_path is None or bundle.validate(check_files=True, verify_hashes=True):
        raise SectionProfileEvaluationError("Section experiment bundle failed verification")
    try:
        candidate = load_section_classifier_candidate(bundle, instrument)
    except BundleValidationError as error:
        raise SectionProfileEvaluationError(
            "Section experiment has no compatible tensor candidate"
        ) from error
    task_view = experiment.get("task_view")
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task_view.get("sha256"), str)
        or len(task_view["sha256"]) != 64
        or task_view["sha256"] != candidate.task_view_sha256
    ):
        raise SectionProfileEvaluationError("Section experiment task view lineage is invalid")
    return bundle_root, experiment, candidate.task_view_sha256


def package_section_evaluation_profile(
    *,
    experiment_dir: Path,
    evaluation_path: Path,
    output_dir: Path,
    profile_id: str,
    instrument: str,
    minimum_accuracy: float,
    maximum_expected_calibration_error: float,
) -> dict[str, object]:
    """Package a held-out evaluator, never a selectable auto-chart profile."""
    if (
        not profile_id
        or len(profile_id) > 128
        or any(not (char.isalnum() or char in "._-") for char in profile_id)
    ):
        raise SectionProfileEvaluationError("Section profile_id is invalid")
    if output_dir.exists():
        raise SectionProfileEvaluationError("Section profile output must not already exist")
    if not (isinstance(minimum_accuracy, (int, float)) and 0 < minimum_accuracy <= 1):
        raise SectionProfileEvaluationError("Section minimum_accuracy is invalid")
    if not (
        isinstance(maximum_expected_calibration_error, (int, float))
        and 0 <= maximum_expected_calibration_error <= 1
    ):
        raise SectionProfileEvaluationError("Section maximum_expected_calibration_error is invalid")
    bundle_root, _experiment, task_view_sha256 = _require_candidate_experiment(
        experiment_dir, instrument
    )
    bundle = load_model_bundle(bundle_root, check_files=True)
    if bundle.manifest_path is None:
        raise SectionProfileEvaluationError("Section experiment bundle has no manifest")
    report = _read_json(evaluation_path, "Section evaluation")
    try:
        validated = _require_evaluation_report(
            report,
            bundle=bundle,
            instrument=instrument,
            expected_task_view_sha256=task_view_sha256,
        )
    except BundleValidationError as error:
        raise SectionProfileEvaluationError(
            "Section evaluation is not a verified held-out report"
        ) from error
    metrics = validated["metrics"]
    if (
        metrics["accuracy"] < minimum_accuracy
        or metrics["expected_calibration_error"] > maximum_expected_calibration_error
    ):
        raise SectionProfileEvaluationError(
            "Section evaluation does not satisfy the requested gate"
        )
    shutil.copytree(bundle_root, output_dir)
    evaluation_destination = output_dir / "evaluations" / "held-out.json"
    evaluation_destination.parent.mkdir(parents=True)
    shutil.copy2(evaluation_path, evaluation_destination)
    profile_config = {
        "schema_version": 1,
        "format": PROFILE_FORMAT,
        "instrument": instrument,
        "component_id": f"section_classifier.{instrument}",
        "model_implementation": ARCHITECTURE,
        "preprocessing": PREPROCESSING,
        "feature_extractor": ROUTER_FEATURE_EXTRACTOR,
        "execution_scope": EXECUTION_SCOPE,
        "auto_chart_status": "not_supported",
        "evaluation": {
            "artifact": "evaluations/held-out.json",
            "sha256": _sha256(evaluation_destination),
            "source_bundle_manifest_sha256": _sha256(bundle.manifest_path),
            "minimum_accuracy": float(minimum_accuracy),
            "maximum_expected_calibration_error": float(maximum_expected_calibration_error),
        },
    }
    config_path = output_dir / "profiles" / f"{profile_id}.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(profile_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = _read_json(manifest_path, "Section bundle manifest")
    profiles = manifest.setdefault("profiles", {})
    if not isinstance(profiles, dict) or profile_id in profiles:
        raise SectionProfileEvaluationError("Section profile id already exists in bundle")
    profiles[profile_id] = {
        "capability": CAPABILITY,
        "instruments": [instrument],
        "required_components": [f"section_classifier.{instrument}"],
        "difficulty_policies": ["evaluation_only"],
        "configuration": config_path.relative_to(output_dir).as_posix(),
        "configuration_sha256": _sha256(config_path),
        "configuration_byte_length": config_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    packaged = load_model_bundle(output_dir, check_files=True)
    errors = packaged.validate(check_files=True, verify_hashes=True)
    if errors:
        raise SectionProfileEvaluationError(
            "packaged Section evaluation profile failed verification"
        )
    return {
        "model_id": packaged.model_id,
        "profile_id": profile_id,
        "capability": CAPABILITY,
        "deployment_status": "evaluation_only_not_auto_chart_runnable",
        "remaining_requirements": [
            "held_out_router_on_off_chart_impact_ablation",
            f"composed_{instrument}_chart_profile_contract",
            "registered_chart_execution_handler",
        ],
    }
