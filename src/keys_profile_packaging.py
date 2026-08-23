"""Held-out evaluation and immutable packaging for Keys V1 experiments.

The bridge accepts only the dedicated ``keys_onset_fret`` view and evaluates
``PART KEYS`` Expert labels.  It produces a Keys-only profile; it never turns
the shared five-lane implementation into a Guitar or Bass runtime profile.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.catalog_task_manifest import (
    MANIFEST_FORMAT,
    resolve_catalog_task_manifest_songs,
    task_label_schema_is_supported,
)
from src.inference.keys_neural_profile import (
    CAPABILITY,
    EVALUATION_FORMAT,
    FORMAT,
    FRET_COMPONENT,
    ONSET_COMPONENT,
    KeysNeuralCharter,
    load_keys_neural_candidate,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle

_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class KeysProfilePackagingError(ValueError):
    """Raised when a Keys experiment cannot safely be evaluated or packaged."""


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
        raise KeysProfilePackagingError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise KeysProfilePackagingError(f"{label} must be a JSON object")
    return raw


def _f1(tp: int, fp: int, fn: int) -> float:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _match_events(
    predicted: Iterable[tuple[float, set[int]]],
    expected: Iterable[tuple[float, set[int]]],
    tolerance_ms: float,
) -> tuple[list[tuple[set[int], set[int]]], int, int]:
    targets = list(expected)
    used: set[int] = set()
    matches: list[tuple[set[int], set[int]]] = []
    unmatched_predictions = 0
    for pred_time, pred_frets in predicted:
        choices = [
            (abs(pred_time - target_time), index)
            for index, (target_time, _target_frets) in enumerate(targets)
            if index not in used and abs(pred_time - target_time) <= tolerance_ms
        ]
        if not choices:
            unmatched_predictions += 1
            continue
        _, index = min(choices)
        used.add(index)
        matches.append((pred_frets, {fret for fret in targets[index][1] if fret >= 0}))
    return matches, unmatched_predictions, len(targets) - len(used)


def _require_keys_task_view(task_view: dict[str, Any]) -> None:
    task = task_view.get("task")
    if (
        task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != "keys_onset_fret"
        or task.get("pipeline_id") != "keys.onset-fret/v1"
        or task.get("instrument") != "keys"
        or not task_label_schema_is_supported("keys_onset_fret", task.get("label_schema"))
    ):
        raise KeysProfilePackagingError(
            "Keys evaluation requires a Keys onset/fret catalog task view"
        )


def _candidate_charter(bundle_root: Path, device: str) -> KeysNeuralCharter:
    bundle = load_model_bundle(bundle_root, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise BundleValidationError("; ".join(errors))
    candidate = load_keys_neural_candidate(bundle)
    onset, fret = bundle.component(ONSET_COMPONENT), bundle.component(FRET_COMPONENT)
    if onset is None or fret is None or onset.checkpoint is None or fret.checkpoint is None:
        raise KeysProfilePackagingError("Keys candidate has incomplete components")
    return KeysNeuralCharter(
        onset.checkpoint,
        fret.checkpoint,
        device=device,
        config={
            "onset": {"model": candidate["onset_model"], "inference": candidate["onset_inference"]},
            "fret": {"model": candidate["fret_model"]},
        },
    )


def evaluate_keys_candidate(
    *,
    bundle_root: Path,
    task_view_path: Path,
    catalog_root: Path,
    output_path: Path,
    device: str,
    tolerance_ms: float = 50.0,
    limit_songs: int = 0,
) -> dict[str, object]:
    """Evaluate a candidate against revalidated held-out ``PART KEYS`` labels."""
    if not 1 <= tolerance_ms <= 1_000:
        raise KeysProfilePackagingError("Keys evaluation tolerance is invalid")
    if not isinstance(limit_songs, int) or isinstance(limit_songs, bool) or limit_songs < 0:
        raise KeysProfilePackagingError("Keys evaluation limit_songs is invalid")
    if output_path.exists():
        raise KeysProfilePackagingError("Keys evaluation output must not already exist")
    task_view = _read_json(task_view_path, "Keys task view")
    _require_keys_task_view(task_view)
    songs = [
        song
        for song in resolve_catalog_task_manifest_songs(task_view, catalog_root)
        if song["split"] == "val"
    ]
    if limit_songs:
        songs = songs[:limit_songs]
    if not songs:
        raise KeysProfilePackagingError("Keys evaluation requires at least one validation song")
    charter = _candidate_charter(bundle_root, device)
    from scripts.preprocess_guitar_windows import load_audio_mono_22050, parse_onsets_from_manifest

    onset_tp = onset_fp = onset_fn = fret_tp = fret_fp = fret_fn = event_tp = event_mismatch = 0
    for song in songs:
        if song.get("label_tracks") != ["PART KEYS"]:
            raise KeysProfilePackagingError("Keys evaluation label source is not PART KEYS")
        audio = load_audio_mono_22050(Path(str(song["audio_path"])))
        expected = parse_onsets_from_manifest(Path(str(song["midi_path"])), label_track="PART KEYS")
        if audio is None or not expected:
            raise KeysProfilePackagingError("Keys evaluation catalog asset is unreadable")
        predicted = charter.transcribe(audio)
        matches, unmatched_predictions, unmatched_targets = _match_events(
            ((event.time_sec * 1000.0, set(event.frets)) for event in predicted),
            expected,
            tolerance_ms,
        )
        onset_tp += len(matches)
        onset_fp += unmatched_predictions
        onset_fn += unmatched_targets
        for predicted_frets, expected_frets in matches:
            fret_tp += len(predicted_frets & expected_frets)
            fret_fp += len(predicted_frets - expected_frets)
            fret_fn += len(expected_frets - predicted_frets)
            if predicted_frets == expected_frets:
                event_tp += 1
            else:
                event_mismatch += 1
    bundle = load_model_bundle(bundle_root, check_files=True)
    if bundle.manifest_path is None:
        raise KeysProfilePackagingError("Keys candidate has no bundle manifest")
    report = {
        "schema_version": 1,
        "format": EVALUATION_FORMAT,
        "model_id": bundle.model_id,
        "bundle_manifest_sha256": _sha256(bundle.manifest_path),
        "task_view_sha256": _sha256(task_view_path),
        "split": "val",
        "records_evaluated": len(songs),
        "alignment_tolerance_ms": tolerance_ms,
        "metrics": {
            "onset_f1": _f1(onset_tp, onset_fp, onset_fn),
            "fret_f1": _f1(fret_tp, fret_fp, fret_fn),
            "event_f1": _f1(event_tp, onset_fp + event_mismatch, onset_fn + event_mismatch),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _require_candidate_experiment(experiment_dir: Path) -> tuple[Path, dict[str, Any]]:
    experiment = _read_json(experiment_dir / "experiment.json", "Keys experiment")
    if (
        experiment.get("format") != "strum-experiment/v1"
        or experiment.get("lifecycle") != "completed"
        or experiment.get("pipeline") != {"id": "keys.onset-fret", "version": 1}
        or experiment.get("deployment_status") != "requires_keys_profile_evaluation_and_packaging"
        or not isinstance(experiment.get("model_bundle"), dict)
    ):
        raise KeysProfilePackagingError("Keys experiment is not a packageable worker artifact")
    bundle_root = experiment_dir / "bundle"
    bundle = load_model_bundle(bundle_root, check_files=True)
    if bundle.manifest_path is None:
        raise KeysProfilePackagingError("Keys experiment bundle is unavailable")
    if bundle.validate(check_files=True, verify_hashes=True):
        raise KeysProfilePackagingError("Keys experiment bundle failed verification")
    source = experiment["model_bundle"]
    if source.get("model_id") != bundle.model_id or source.get("manifest_sha256") != _sha256(
        bundle.manifest_path
    ):
        raise KeysProfilePackagingError("Keys experiment bundle provenance does not match")
    load_keys_neural_candidate(bundle)
    return bundle_root, experiment


def _require_deployment_evaluation(
    report: dict[str, Any], *, model_id: str, bundle_manifest_sha256: str
) -> dict[str, float]:
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    required = {
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
        set(report) != required
        or report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("model_id") != model_id
        or report.get("bundle_manifest_sha256") != bundle_manifest_sha256
        or not isinstance(report.get("task_view_sha256"), str)
        or len(report["task_view_sha256"]) != 64
        or not isinstance(report.get("records_evaluated"), int)
        or isinstance(report["records_evaluated"], bool)
        or report["records_evaluated"] < 1
        or report.get("split") != "val"
        or not isinstance(report.get("alignment_tolerance_ms"), (int, float))
        or isinstance(report["alignment_tolerance_ms"], bool)
        or not 1 <= report["alignment_tolerance_ms"] <= 1_000
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool)
            and 0 <= metrics[key] <= 1
            for key in ("onset_f1", "fret_f1", "event_f1")
        )
    ):
        raise KeysProfilePackagingError("Keys evaluation is not a verified held-out report")
    return {key: float(metrics[key]) for key in ("onset_f1", "fret_f1", "event_f1")}


def package_keys_profile(
    *,
    experiment_dir: Path,
    evaluation_path: Path,
    output_dir: Path,
    profile_id: str,
    minimum_onset_f1: float,
    minimum_fret_f1: float,
    onset_threshold: float | None = None,
    fret_thresholds: tuple[float, ...] | None = None,
    note_duration_ms: float = 100.0,
) -> dict[str, object]:
    """Copy an evaluated Keys experiment into an immutable Expert profile."""
    if not _PROFILE_ID.fullmatch(profile_id):
        raise KeysProfilePackagingError("Keys profile_id is invalid")
    if output_dir.exists():
        raise KeysProfilePackagingError("Keys profile output must not already exist")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1
        for value in (minimum_onset_f1, minimum_fret_f1)
    ):
        raise KeysProfilePackagingError("Keys deployment metric gates must be between zero and one")
    if not isinstance(note_duration_ms, (int, float)) or not 1 <= note_duration_ms <= 10_000:
        raise KeysProfilePackagingError("Keys note_duration_ms is invalid")
    bundle_root, experiment = _require_candidate_experiment(experiment_dir)
    bundle = load_model_bundle(bundle_root, check_files=True)
    candidate = load_keys_neural_candidate(bundle)
    if bundle.manifest_path is None:
        raise KeysProfilePackagingError("Keys candidate bundle is unavailable")
    report = _read_json(evaluation_path, "Keys evaluation")
    metrics = _require_deployment_evaluation(
        report, model_id=bundle.model_id, bundle_manifest_sha256=_sha256(bundle.manifest_path)
    )
    if metrics["onset_f1"] < minimum_onset_f1 or metrics["fret_f1"] < minimum_fret_f1:
        raise KeysProfilePackagingError(
            "Keys evaluation does not satisfy the requested deployment gate"
        )
    inference = candidate["onset_inference"]
    configured_onset_threshold = (
        onset_threshold if onset_threshold is not None else inference.get("peak_threshold")
    )
    min_distance = inference.get("peak_min_distance_frames")
    configured_fret_thresholds = fret_thresholds or (0.5,) * 5
    if (
        not isinstance(configured_onset_threshold, (int, float))
        or not 0 < configured_onset_threshold <= 1
        or not isinstance(min_distance, int)
        or isinstance(min_distance, bool)
        or min_distance < 1
        or len(configured_fret_thresholds) != 5
        or not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) and 0 < value <= 1
            for value in configured_fret_thresholds
        )
    ):
        raise KeysProfilePackagingError("Keys inference thresholds are invalid")
    shutil.copytree(bundle_root, output_dir)
    evaluation_destination = output_dir / "evaluations" / "validation.json"
    evaluation_destination.parent.mkdir(parents=True)
    shutil.copy2(evaluation_path, evaluation_destination)
    profile_config = {
        "schema_version": 1,
        "format": FORMAT,
        "preprocessing": candidate["preprocessing"],
        "audio": candidate["audio"],
        "onset_model": candidate["onset_model"],
        "fret_model": candidate["fret_model"],
        "onset_threshold": float(configured_onset_threshold),
        "peak_min_distance_frames": min_distance,
        "fret_thresholds": [float(value) for value in configured_fret_thresholds],
        "note_duration_ms": float(note_duration_ms),
        "evaluation": {
            "artifact": "evaluations/validation.json",
            "sha256": _sha256(evaluation_destination),
            "source_bundle_manifest_sha256": _sha256(bundle.manifest_path),
            "minimum_onset_f1": float(minimum_onset_f1),
            "minimum_fret_f1": float(minimum_fret_f1),
        },
    }
    config_path = output_dir / "profiles" / f"{profile_id}.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(profile_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = _read_json(manifest_path, "Keys bundle manifest")
    profiles = manifest.setdefault("profiles", {})
    if not isinstance(profiles, dict) or profile_id in profiles:
        raise KeysProfilePackagingError("Keys profile id already exists in bundle")
    profiles[profile_id] = {
        "capability": CAPABILITY,
        "instruments": ["keys"],
        "required_components": [ONSET_COMPONENT, FRET_COMPONENT],
        "difficulty_policies": ["expert_only"],
        "configuration": config_path.relative_to(output_dir).as_posix(),
        "configuration_sha256": _sha256(config_path),
        "configuration_byte_length": config_path.stat().st_size,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    provenance_path = output_dir / "provenance" / "source-experiment.json"
    provenance_path.parent.mkdir(parents=True)
    provenance_path.write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "packaged",
        "model_id": bundle.model_id,
        "profile_id": profile_id,
        "capability": CAPABILITY,
        "deployment_status": "deployable",
        "bundle_name": output_dir.name,
        "manifest_sha256": _sha256(manifest_path),
    }
