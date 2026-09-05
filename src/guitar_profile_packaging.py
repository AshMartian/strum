"""Evaluation and immutable packaging for catalog-trained Guitar V1 models.

Raw training experiments remain experiments.  A caller must run the supplied
song-disjoint evaluation and choose explicit quality gates before this module
will produce a worker-executable profile bundle.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from src.catalog_guitar_manifest import MANIFEST_FORMAT, resolve_guitar_manifest_songs
from src.inference.guitar_neural import GuitarNeuralCharter
from src.inference.guitar_neural_profile import (
    CAPABILITY,
    EVALUATION_FORMAT,
    FORMAT,
    FRET_COMPONENT,
    ONSET_COMPONENT,
    PREPROCESSING,
    load_guitar_neural_candidate,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.profile_quality_policy import (
    evaluation_matches_experiment,
    profile_quality_policy,
    profile_quality_policy_sha256,
)

_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class GuitarProfilePackagingError(ValueError):
    """Raised when an experiment cannot be safely evaluated or packaged."""


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
        raise GuitarProfilePackagingError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise GuitarProfilePackagingError(f"{label} must be a JSON object")
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
    """Greedily match each predicted onset to one ground-truth onset."""
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


def _candidate_charter(bundle_root: Path, device: str) -> GuitarNeuralCharter:
    bundle = load_model_bundle(bundle_root, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise BundleValidationError("; ".join(errors))
    candidate = load_guitar_neural_candidate(bundle)
    onset, fret = bundle.component(ONSET_COMPONENT), bundle.component(FRET_COMPONENT)
    if onset is None or fret is None or onset.checkpoint is None or fret.checkpoint is None:
        raise GuitarProfilePackagingError("Guitar candidate has incomplete components")
    return GuitarNeuralCharter(
        onset.checkpoint,
        fret.checkpoint,
        device=device,
        config={
            "onset": {"model": candidate["onset_model"], "inference": candidate["onset_inference"]},
            "fret": {"model": candidate["fret_model"]},
        },
    )


def evaluate_guitar_candidate(
    *,
    bundle_root: Path,
    task_view_path: Path,
    catalog_root: Path,
    output_path: Path,
    device: str,
    tolerance_ms: float = 50.0,
    limit_songs: int = 0,
) -> dict[str, object]:
    """Evaluate an un-packaged pair on revalidated task-view test songs.

    Private catalog locations are used only while this function runs; the
    resulting report records hashes and aggregate metrics, never those paths.
    """
    if (
        isinstance(tolerance_ms, bool)
        or tolerance_ms != profile_quality_policy()["alignment_tolerance_ms"]
    ):
        raise GuitarProfilePackagingError("Guitar evaluation tolerance is invalid")
    if not isinstance(limit_songs, int) or isinstance(limit_songs, bool) or limit_songs != 0:
        raise GuitarProfilePackagingError("Guitar evaluation limit_songs is invalid")
    if output_path.exists():
        raise GuitarProfilePackagingError("Guitar evaluation output must not already exist")
    task_view = _read_json(task_view_path, "Guitar task view")
    if task_view.get("format") != MANIFEST_FORMAT:
        raise GuitarProfilePackagingError("Guitar evaluation requires a Guitar catalog task view")
    # Promotion uses the same source-disjoint task view as training.
    from src.five_lane_runtime_admission import PROFILE_GRADE_ADMISSION

    if task_view.get("task", {}).get("profile_grade_admission") != PROFILE_GRADE_ADMISSION:
        raise GuitarProfilePackagingError("Guitar evaluation requires profile-grade preparation")
    _, experiment = _require_candidate_experiment(bundle_root.parent)
    if experiment.get("task_view", {}).get("sha256") != _sha256(task_view_path):
        raise GuitarProfilePackagingError("Guitar evaluation task view differs from training")
    songs = [
        song
        for song in resolve_guitar_manifest_songs(task_view, catalog_root)
        if song["split"] == "test"
    ]
    if len(songs) < profile_quality_policy()["minimum_test_sources"]:
        raise GuitarProfilePackagingError(
            "Guitar evaluation requires the complete profile-grade test split"
        )
    charter = _candidate_charter(bundle_root, device)
    from scripts.build_guitar_manifest import parse_part_guitar_expert  # noqa: PLC0415
    from scripts.preprocess_guitar_windows import load_audio_mono_22050  # noqa: PLC0415

    onset_tp = onset_fp = onset_fn = fret_tp = fret_fp = fret_fn = event_tp = event_mismatch = 0
    for song in songs:
        audio = load_audio_mono_22050(Path(str(song["audio_path"])))
        expected = parse_part_guitar_expert(Path(str(song["midi_path"])))
        if audio is None or not expected:
            raise GuitarProfilePackagingError("Guitar evaluation catalog asset is unreadable")
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
        raise GuitarProfilePackagingError("Guitar candidate has no bundle manifest")
    report = {
        "schema_version": 1,
        "format": EVALUATION_FORMAT,
        "model_id": bundle.model_id,
        "bundle_manifest_sha256": _sha256(bundle.manifest_path),
        "task_view_sha256": _sha256(task_view_path),
        "split": "test",
        "records_evaluated": len(songs),
        "alignment_tolerance_ms": tolerance_ms,
        "quality_policy": profile_quality_policy(),
        "quality_policy_sha256": profile_quality_policy_sha256(),
        "metrics": {
            "onset_f1": _f1(onset_tp, onset_fp, onset_fn),
            "fret_f1": _f1(fret_tp, fret_fp, fret_fn),
            "event_f1": _f1(
                event_tp,
                onset_fp + event_mismatch,
                onset_fn + event_mismatch,
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _require_candidate_experiment(experiment_dir: Path) -> tuple[Path, dict[str, Any]]:
    experiment = _read_json(experiment_dir / "experiment.json", "Guitar experiment")
    if (
        experiment.get("format") != "strum-experiment/v1"
        or experiment.get("lifecycle") != "completed"
        or experiment.get("pipeline") != {"id": "guitar.onset-fret", "version": 1}
        or experiment.get("deployment_status") != "requires_profile_packaging"
        or not isinstance(experiment.get("model_bundle"), dict)
    ):
        raise GuitarProfilePackagingError("Guitar experiment is not a packageable worker artifact")
    bundle_root = experiment_dir / "bundle"
    bundle = load_model_bundle(bundle_root, check_files=True)
    if bundle.manifest_path is None:
        raise GuitarProfilePackagingError("Guitar experiment bundle is unavailable")
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise GuitarProfilePackagingError("Guitar experiment bundle failed verification")
    source = experiment["model_bundle"]
    if source.get("model_id") != bundle.model_id or source.get("manifest_sha256") != _sha256(
        bundle.manifest_path
    ):
        raise GuitarProfilePackagingError("Guitar experiment bundle provenance does not match")
    return bundle_root, experiment


def package_guitar_profile(
    *,
    experiment_dir: Path,
    evaluation_path: Path,
    output_dir: Path,
    profile_id: str,
) -> dict[str, object]:
    """Copy an evaluated worker experiment into an immutable Expert profile.

    The source experiment is never rewritten; it remains an experiment even
    after a separate deployment bundle has been created from it.
    """
    if not _PROFILE_ID.fullmatch(profile_id):
        raise GuitarProfilePackagingError("Guitar profile_id is invalid")
    if output_dir.exists():
        raise GuitarProfilePackagingError("Guitar profile output must not already exist")
    policy = profile_quality_policy()
    note_duration_ms = policy["note_duration_ms"]
    bundle_root, experiment = _require_candidate_experiment(experiment_dir)
    bundle = load_model_bundle(bundle_root, check_files=True)
    candidate = load_guitar_neural_candidate(bundle)
    if bundle.manifest_path is None:
        raise GuitarProfilePackagingError("Guitar candidate bundle is unavailable")
    report = _read_json(evaluation_path, "Guitar evaluation")
    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if (
        report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("model_id") != bundle.model_id
        or report.get("bundle_manifest_sha256") != _sha256(bundle.manifest_path)
        or report.get("quality_policy") != policy
        or report.get("quality_policy_sha256") != profile_quality_policy_sha256()
        or not isinstance(report.get("records_evaluated"), int)
        or isinstance(report["records_evaluated"], bool)
        or report["records_evaluated"] < policy["minimum_test_sources"]
        or report.get("split") != policy["evaluation_split"]
        or report.get("alignment_tolerance_ms") != policy["alignment_tolerance_ms"]
        or not evaluation_matches_experiment(report, experiment)
        or not all(
            isinstance(metrics.get(key), (int, float))
            and not isinstance(metrics[key], bool)
            and 0 <= metrics[key] <= 1
            for key in ("onset_f1", "fret_f1", "event_f1")
        )
        or metrics["onset_f1"] < policy["minimum_onset_f1"]
        or metrics["fret_f1"] < policy["minimum_fret_f1"]
    ):
        raise GuitarProfilePackagingError(
            "Guitar evaluation does not satisfy the requested deployment gate"
        )
    inference = candidate["onset_inference"]
    configured_onset_threshold = policy["onset_threshold"]
    min_distance = inference.get("peak_min_distance_frames")
    configured_fret_thresholds = tuple(policy["fret_thresholds"])
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
        raise GuitarProfilePackagingError("Guitar inference thresholds are invalid")
    shutil.copytree(bundle_root, output_dir)
    evaluation_destination = output_dir / "evaluations" / "validation.json"
    evaluation_destination.parent.mkdir(parents=True)
    shutil.copy2(evaluation_path, evaluation_destination)
    profile_config = {
        "schema_version": 1,
        "format": FORMAT,
        "preprocessing": PREPROCESSING,
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
            "minimum_onset_f1": float(policy["minimum_onset_f1"]),
            "minimum_fret_f1": float(policy["minimum_fret_f1"]),
            "quality_policy": policy,
            "quality_policy_sha256": profile_quality_policy_sha256(),
        },
    }
    config_path = output_dir / "profiles" / f"{profile_id}.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        json.dumps(profile_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = output_dir / MANIFEST_FILENAME
    manifest = _read_json(manifest_path, "Guitar bundle manifest")
    profiles = manifest.setdefault("profiles", {})
    if not isinstance(profiles, dict) or profile_id in profiles:
        raise GuitarProfilePackagingError("Guitar profile id already exists in bundle")
    profiles[profile_id] = {
        "capability": CAPABILITY,
        "instruments": ["guitar"],
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
