"""Held-out admission and immutable packaging for five-lane transforms.

The training script intentionally writes a *candidate* bundle with no profile.
This module is the only bridge from that candidate to the executable
``difficulty.transform/v1`` capability. It verifies a candidate's immutable
calibration decoder, recomputes quality only against its source-disjoint test
split, and copies the exact weights, configuration, and report into a separate
bundle. Promotion is gated by STRUM's versioned quality policy; callers cannot
choose score thresholds.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from scripts.train_chart_transform import (
    DatasetValidationError,
    EventTransformMLP,
    TrainingConfig,
    _load_audio_assets,
    _make_tensors,
    load_dataset,
    split_by_song_three_way,
)
from src.chart_transform_calibration import (
    calibration_evidence,
    metrics_from_probabilities,
    state_dict_sha256,
    validate_calibration_evidence,
    validate_checkpoint_selection_evidence,
)
from src.chart_transform_quality_policy import (
    evaluate_quality_policy,
    quality_policy_evidence,
    validate_quality_policy_evidence,
)
from src.model_bundle import (
    MANIFEST_FILENAME,
    BundleValidationError,
    ModelBundle,
    load_model_bundle,
)

EVALUATION_FORMAT = "strum-chart-transform-held-out-evaluation/v1"
PROFILE_FORMAT = "strum-chart-transform-promoted-profile/v1"
CAPABILITY = "difficulty.transform/v1"
ARCHITECTURE = "EventTransformMLP/v1"
PREPROCESSING = "midi-five-lane-events/v1"
DEPLOYMENT_STATUS = "requires_transform_profile_evaluation_and_promotion"
CANDIDATE_LINEAGE_FORMAT = "strum-chart-transform-candidate-lineage/v1"


class ChartTransformPromotionError(ValueError):
    """Raised when a candidate, report, or package violates the promotion contract."""


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
        raise ChartTransformPromotionError(f"{label} is unreadable") from error
    if not isinstance(raw, dict):
        raise ChartTransformPromotionError(f"{label} must be a JSON object")
    return raw


def _component_id(config: dict[str, Any]) -> str:
    instrument = config.get("instrument")
    source = config.get("source_difficulty")
    target = config.get("target_difficulty")
    if not all(isinstance(value, str) and value for value in (instrument, source, target)):
        raise ChartTransformPromotionError("candidate configuration has invalid transform identity")
    return f"chart_transform.{instrument}.{source.lower()}_to_{target.lower()}"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _candidate_lineage(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the V2 three-way lineage required for promotion."""
    lineage = config.get("lineage")
    if not isinstance(lineage, dict):
        raise ChartTransformPromotionError("transform candidate lacks immutable lineage")
    required = {
        "schema_version",
        "format",
        "dataset",
        "task_view",
        "split",
        "audio_manifest_sha256",
    }
    dataset = lineage.get("dataset")
    task_view = lineage.get("task_view")
    split = lineage.get("split")
    required_split = {
        "unit",
        "algorithm",
        "seed",
        "calibration_fraction",
        "test_fraction",
        "assignments_sha256",
        "train_song_ids",
        "calibration_song_ids",
        "test_song_ids",
    }
    if (
        set(lineage) != required
        or lineage.get("schema_version") != 1
        or lineage.get("format") != CANDIDATE_LINEAGE_FORMAT
        or not isinstance(dataset, dict)
        or set(dataset) != {"dataset_id", "format", "manifest_sha256", "records_sha256"}
        or not isinstance(dataset.get("dataset_id"), str)
        or not dataset["dataset_id"]
        or dataset.get("format") != "strum-chart-pairs/v1"
        or not _is_sha256(dataset.get("manifest_sha256"))
        or not _is_sha256(dataset.get("records_sha256"))
        or not isinstance(task_view, dict)
        or set(task_view) != {"task_view_id", "task_view_sha256", "pipeline", "catalog"}
        or not _is_sha256(task_view.get("task_view_id"))
        or not _is_sha256(task_view.get("task_view_sha256"))
        or task_view.get("pipeline") != {"id": "chart_transform.five_lane", "version": 1}
        or not isinstance(task_view.get("catalog"), dict)
        or set(task_view["catalog"]) != {"catalog_id", "manifest_sha256", "records_sha256"}
        or not isinstance(task_view["catalog"].get("catalog_id"), str)
        or not task_view["catalog"]["catalog_id"]
        or not _is_sha256(task_view["catalog"].get("manifest_sha256"))
        or not _is_sha256(task_view["catalog"].get("records_sha256"))
        or not isinstance(split, dict)
        or set(split) != required_split
        or split.get("unit") != "song_id"
        or split.get("algorithm") != "sha256-source-id-rank/v2-three-way"
        or not isinstance(split.get("seed"), int)
        or isinstance(split.get("seed"), bool)
        or not isinstance(split.get("calibration_fraction"), (int, float))
        or isinstance(split.get("calibration_fraction"), bool)
        or not isinstance(split.get("test_fraction"), (int, float))
        or isinstance(split.get("test_fraction"), bool)
        or not 0 < float(split["calibration_fraction"]) < 1
        or not 0 < float(split["test_fraction"]) < 1
        or float(split["calibration_fraction"]) + float(split["test_fraction"]) >= 1
        or not _is_sha256(split.get("assignments_sha256"))
        or not isinstance(split.get("train_song_ids"), list)
        or not isinstance(split.get("calibration_song_ids"), list)
        or not isinstance(split.get("test_song_ids"), list)
        or not all(isinstance(song_id, str) and song_id for song_id in split["train_song_ids"])
        or not all(
            isinstance(song_id, str) and song_id for song_id in split["calibration_song_ids"]
        )
        or not all(isinstance(song_id, str) and song_id for song_id in split["test_song_ids"])
        or not split["train_song_ids"]
        or not split["calibration_song_ids"]
        or not split["test_song_ids"]
        or len(
            set(split["train_song_ids"])
            | set(split["calibration_song_ids"])
            | set(split["test_song_ids"])
        )
        != sum(
            len(split[key]) for key in ("train_song_ids", "calibration_song_ids", "test_song_ids")
        )
        or (
            lineage["audio_manifest_sha256"] is not None
            and not _is_sha256(lineage["audio_manifest_sha256"])
        )
    ):
        raise ChartTransformPromotionError("transform candidate lineage is invalid")
    return lineage


def _candidate(bundle_root: str | Path) -> tuple[ModelBundle, str, dict[str, Any]]:
    try:
        bundle = load_model_bundle(bundle_root, check_files=True)
    except BundleValidationError as error:
        raise ChartTransformPromotionError(
            "transform candidate bundle failed verification"
        ) from error
    if bundle.profiles:
        raise ChartTransformPromotionError(
            "transform candidate must not declare an inference profile"
        )
    if bundle.manifest_path is None or len(bundle.components) != 1:
        raise ChartTransformPromotionError("transform candidate must contain exactly one component")
    component = next(iter(bundle.components.values()))
    if (
        component.checkpoint is None
        or component.config is None
        or component.sha256 is None
        or component.config_sha256 is None
        or component.architecture != ARCHITECTURE
        or component.preprocessing != PREPROCESSING
    ):
        raise ChartTransformPromotionError("transform candidate component is incomplete")
    config = _read_json(component.config, "transform candidate configuration")
    if component.name != _component_id(config):
        raise ChartTransformPromotionError(
            "transform candidate component does not match configuration"
        )
    required = {
        "dataset_manifest",
        "output_dir",
        "model_id",
        "source_difficulty",
        "target_difficulty",
        "seed",
        "validation_fraction",
        "lane_count",
        "alignment_tolerance_ms",
        "hidden_dim",
        "learning_rate",
        "epochs",
        "device",
        "audio_feature_mode",
        "audio_manifest",
        "audio_sample_rate",
        "audio_window_ms",
        "audio_max_duration_seconds",
        "init_checkpoint",
        "checkpoint_mode",
        "parent_provenance",
        "strum_revision",
        "instrument",
        "lineage",
        "decoder_calibration",
    }
    if (
        set(config) != required
        or config.get("dataset_manifest") is not None
        or config.get("output_dir") is not None
    ):
        raise ChartTransformPromotionError("transform candidate configuration is unsupported")
    _candidate_lineage(config)
    return bundle, component.name, config


def _candidate_state(
    bundle: ModelBundle, component_id: str, config: dict[str, Any]
) -> EventTransformMLP:
    component = bundle.component(component_id)
    assert component is not None and component.checkpoint is not None
    try:
        payload = torch.load(component.checkpoint, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise ChartTransformPromotionError(
            "transform candidate checkpoint is not tensor-only"
        ) from error
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if (
        not isinstance(state, dict)
        or payload.get("model_type") != "EventTransformMLP"
        or payload.get("lane_count") != config["lane_count"]
        or payload.get("hidden_dim") != config["hidden_dim"]
        or payload.get("audio_feature_mode") != config["audio_feature_mode"]
        or not all(
            isinstance(name, str) and isinstance(value, torch.Tensor)
            for name, value in state.items()
        )
    ):
        raise ChartTransformPromotionError("transform candidate checkpoint is incompatible")
    calibration = config.get("decoder_calibration")
    selection = calibration.get("checkpoint_selection") if isinstance(calibration, dict) else None
    if (
        not isinstance(selection, dict)
        or not validate_checkpoint_selection_evidence(selection)
        or not isinstance(config.get("epochs"), int)
        or isinstance(config["epochs"], bool)
        or selection.get("epochs_evaluated") != config["epochs"]
        or payload.get("checkpoint_selection") != selection
        or state_dict_sha256(state) != selection.get("selected_state_sha256")
    ):
        raise ChartTransformPromotionError("transform candidate checkpoint selection is invalid")
    model = EventTransformMLP(
        lane_count=config["lane_count"],
        hidden_dim=config["hidden_dim"],
        audio_feature_dim=int(payload.get("audio_feature_dim", 0)),
    )
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ChartTransformPromotionError(
            "transform candidate tensor state is incompatible"
        ) from error
    return model


def _verify_declared_lineage(
    *,
    lineage: dict[str, Any],
    dataset_manifest: dict[str, Any],
    dataset_manifest_path: Path,
    pairs: list[Any],
    train_pairs: list[Any],
    calibration_pairs: list[Any],
    test_pairs: list[Any],
    audio_manifest_sha256: str | None,
) -> None:
    """Prove the supplied local task view is the one the candidate declares."""
    dataset = lineage["dataset"]
    task_lineage = lineage["task_view"]
    split_lineage = lineage["split"]
    task_view = dataset_manifest.get("task_view")
    if not isinstance(task_view, dict):
        raise ChartTransformPromotionError("held-out evaluation requires a catalog task view")
    records_value = dataset_manifest.get("records")
    if not isinstance(records_value, str):
        raise ChartTransformPromotionError("held-out transform dataset is invalid")
    records_path = (dataset_manifest_path.parent / records_value).resolve()
    try:
        records_path.relative_to(dataset_manifest_path.parent.resolve())
    except ValueError as error:
        raise ChartTransformPromotionError("held-out transform dataset is invalid") from error
    assignments = {pair.song_id: pair.split for pair in pairs}
    expected_assignments = (
        {pair.song_id: "train" for pair in train_pairs}
        | {pair.song_id: "calibration" for pair in calibration_pairs}
        | {pair.song_id: "test" for pair in test_pairs}
    )
    if (
        not records_path.is_file()
        or dataset.get("dataset_id") != dataset_manifest.get("dataset_id")
        or dataset.get("format") != dataset_manifest.get("format")
        or dataset.get("manifest_sha256") != _sha256(dataset_manifest_path)
        or dataset.get("records_sha256") != _sha256(records_path)
        or task_lineage.get("task_view_id") != task_view.get("task_view_id")
        or task_lineage.get("task_view_sha256") != _canonical_sha256(task_view)
        or task_lineage.get("pipeline") != task_view.get("pipeline")
        or task_lineage.get("catalog") != task_view.get("catalog")
        or assignments != expected_assignments
        or split_lineage.get("algorithm") != task_view.get("split", {}).get("algorithm")
        or split_lineage.get("seed") != task_view.get("split", {}).get("seed")
        or split_lineage.get("calibration_fraction")
        != task_view.get("split", {}).get("calibration_fraction")
        or split_lineage.get("test_fraction") != task_view.get("split", {}).get("test_fraction")
        or split_lineage.get("assignments_sha256") != _canonical_sha256(assignments)
        or split_lineage.get("train_song_ids") != sorted(pair.song_id for pair in train_pairs)
        or split_lineage.get("calibration_song_ids")
        != sorted(pair.song_id for pair in calibration_pairs)
        or split_lineage.get("test_song_ids") != sorted(pair.song_id for pair in test_pairs)
        or lineage.get("audio_manifest_sha256") != audio_manifest_sha256
    ):
        raise ChartTransformPromotionError(
            "held-out transform dataset does not match candidate lineage"
        )


def evaluate_chart_transform_candidate(
    *,
    bundle_root: str | Path,
    dataset_manifest: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    audio_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Recompute metrics on an immutable candidate's declared held-out split.

    The evaluator requires an explicit source-disjoint ``train``,
    ``calibration``, and ``test`` task view. It verifies that the candidate's
    fixed decoder came from calibration, then applies it only to test, so a
    local seed or threshold cannot manufacture promotion evidence.
    """
    bundle, component_id, portable = _candidate(bundle_root)
    lineage = _candidate_lineage(portable)
    values = dict(portable)
    values.pop("instrument", None)
    values.pop("lineage", None)
    calibration = values.pop("decoder_calibration", None)
    values.update(
        {
            "dataset_manifest": str(dataset_manifest),
            "output_dir": "evaluation-local",
            "device": device,
        }
    )
    if audio_manifest is not None:
        values["audio_manifest"] = str(audio_manifest)
    try:
        config = TrainingConfig.from_mapping(values)
        pairs, manifest = load_dataset(config)
        assets, audio_manifest_sha256 = _load_audio_assets(config, pairs)
        pairs = [replace(pair, audio_path=assets.get(pair.song_id)) for pair in pairs]
        train_pairs, calibration_pairs, test_pairs = split_by_song_three_way(
            pairs, config.seed, config.validation_fraction
        )
        if not calibration_pairs or not test_pairs or not train_pairs:
            raise ChartTransformPromotionError(
                "promotion evaluation requires non-empty train, calibration, and test songs"
            )
        _verify_declared_lineage(
            lineage=lineage,
            dataset_manifest=manifest,
            dataset_manifest_path=Path(dataset_manifest).expanduser().resolve(),
            pairs=pairs,
            train_pairs=train_pairs,
            calibration_pairs=calibration_pairs,
            test_pairs=test_pairs,
            audio_manifest_sha256=audio_manifest_sha256,
        )
        _, _, _ = _make_tensors(train_pairs, config)
        calibration_features, calibration_targets, _ = _make_tensors(calibration_pairs, config)
        features, targets, _ = _make_tensors(test_pairs, config)
    except DatasetValidationError as error:
        raise ChartTransformPromotionError("held-out transform dataset is invalid") from error
    model = _candidate_state(bundle, component_id, portable).to(config.device)
    if not validate_calibration_evidence(calibration):
        raise ChartTransformPromotionError("transform candidate lacks decoder calibration evidence")
    component = bundle.component(component_id)
    assert component is not None and component.sha256 is not None
    if (
        calibration["component_sha256"] != component.sha256
        or calibration["split_assignments_sha256"] != lineage["split"]["assignments_sha256"]
        or calibration["calibration_song_ids"] != lineage["split"]["calibration_song_ids"]
    ):
        raise ChartTransformPromotionError("transform decoder calibration lineage is invalid")
    with torch.inference_mode():
        calibration_probabilities = torch.sigmoid(model(calibration_features.to(config.device)))
        test_logits = model(features.to(config.device))
        test_probabilities = torch.sigmoid(test_logits)
    recomputed_calibration = calibration_evidence(
        probabilities=calibration_probabilities,
        targets=calibration_targets.to(config.device),
        component_sha256=component.sha256,
        calibration_song_ids=sorted({pair.song_id for pair in calibration_pairs}),
        split_assignments_sha256=lineage["split"]["assignments_sha256"],
        checkpoint_selection=calibration["checkpoint_selection"],
    )
    if calibration != recomputed_calibration:
        raise ChartTransformPromotionError(
            "transform decoder calibration evidence does not match candidate"
        )
    thresholds = tuple(calibration["thresholds"])
    metrics = {
        "loss": float(
            torch.nn.functional.binary_cross_entropy_with_logits(
                test_logits, targets.to(config.device)
            ).item()
        ),
        **metrics_from_probabilities(test_probabilities, targets.to(config.device), thresholds),
    }
    assert (
        component is not None
        and component.sha256 is not None
        and component.config_sha256 is not None
    )
    report = {
        "schema_version": 1,
        "format": EVALUATION_FORMAT,
        "model_id": bundle.model_id,
        "candidate_manifest_sha256": _sha256(bundle.manifest_path),
        "component_id": component_id,
        "component_sha256": component.sha256,
        "component_configuration_sha256": component.config_sha256,
        "candidate_lineage_sha256": _canonical_sha256(lineage),
        "dataset_manifest_sha256": _sha256(Path(dataset_manifest)),
        "dataset_records_sha256": lineage["dataset"]["records_sha256"],
        "dataset_id": manifest["dataset_id"],
        "task_view_id": lineage["task_view"]["task_view_id"],
        "task_view_sha256": lineage["task_view"]["task_view_sha256"],
        "split_assignments_sha256": lineage["split"]["assignments_sha256"],
        "instrument": manifest["instrument"],
        "source_difficulty": config.source_difficulty,
        "target_difficulty": config.target_difficulty,
        "split": "test",
        "split_unit": "song_id",
        "held_out_song_ids": sorted({pair.song_id for pair in test_pairs}),
        "records_evaluated": len(test_pairs),
        "metrics": metrics,
        "decoder_calibration": calibration,
        "quality_policy": quality_policy_evidence(),
        "quality_gate": evaluate_quality_policy(metrics),
        "audio_manifest_sha256": audio_manifest_sha256,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {key: value for key, value in report.items() if key not in {"held_out_song_ids"}}
    policy = report["quality_policy"]
    gate = report["quality_gate"]
    assert isinstance(policy, dict) and isinstance(gate, dict)
    result["quality_policy_id"] = policy["policy_id"]
    result["quality_policy_sha256"] = policy["policy_sha256"]
    result["quality_gate_status"] = gate["status"]
    return result


def _require_report(
    report_path: Path, bundle: ModelBundle, component_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    report = _read_json(report_path, "transform held-out evaluation report")
    lineage = _candidate_lineage(config)
    component = bundle.component(component_id)
    assert bundle.manifest_path is not None and component is not None
    required = {
        "schema_version",
        "format",
        "model_id",
        "candidate_manifest_sha256",
        "component_id",
        "component_sha256",
        "component_configuration_sha256",
        "candidate_lineage_sha256",
        "dataset_manifest_sha256",
        "dataset_records_sha256",
        "dataset_id",
        "task_view_id",
        "task_view_sha256",
        "split_assignments_sha256",
        "instrument",
        "source_difficulty",
        "target_difficulty",
        "split",
        "split_unit",
        "held_out_song_ids",
        "records_evaluated",
        "metrics",
        "decoder_calibration",
        "quality_policy",
        "quality_gate",
        "audio_manifest_sha256",
    }
    metrics = report.get("metrics")
    if (
        set(report) != required
        or report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("model_id") != bundle.model_id
        or report.get("candidate_manifest_sha256") != _sha256(bundle.manifest_path)
        or report.get("component_id") != component_id
        or report.get("component_sha256") != component.sha256
        or report.get("component_configuration_sha256") != component.config_sha256
        or report.get("candidate_lineage_sha256") != _canonical_sha256(lineage)
        or report.get("dataset_manifest_sha256") != lineage["dataset"]["manifest_sha256"]
        or report.get("dataset_records_sha256") != lineage["dataset"]["records_sha256"]
        or report.get("dataset_id") != lineage["dataset"]["dataset_id"]
        or report.get("task_view_id") != lineage["task_view"]["task_view_id"]
        or report.get("task_view_sha256") != lineage["task_view"]["task_view_sha256"]
        or report.get("split_assignments_sha256") != lineage["split"]["assignments_sha256"]
        or report.get("instrument") != config["instrument"]
        or report.get("source_difficulty") != config["source_difficulty"]
        or report.get("target_difficulty") != config["target_difficulty"]
        or report.get("split") != "test"
        or report.get("split_unit") != "song_id"
        or not isinstance(report.get("held_out_song_ids"), list)
        or report["held_out_song_ids"] != lineage["split"]["test_song_ids"]
        or not isinstance(report.get("records_evaluated"), int)
        or report["records_evaluated"] < 1
        or not isinstance(metrics, dict)
        or set(metrics) != {"loss", "lane_precision", "lane_recall", "lane_f1"}
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in metrics.values()
        )
        or not validate_quality_policy_evidence(
            report.get("quality_policy"), report.get("quality_gate"), metrics
        )
        or report.get("decoder_calibration") != config.get("decoder_calibration")
    ):
        raise ChartTransformPromotionError("transform held-out evaluation report is invalid")
    return report


def package_chart_transform_profile(
    *,
    experiment_dir: str | Path,
    evaluation_path: str | Path,
    dataset_manifest: str | Path,
    output_dir: str | Path,
    profile_id: str,
    device: str = "cpu",
    audio_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Create an executable profile after independently reproducing evaluation.

    The caller's report is review evidence, not a promotion authority.  The
    package step reruns evaluation from the candidate's declared catalog task
    view and refuses any report whose complete, canonical contents differ.
    """
    if not profile_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in profile_id
    ):
        raise ChartTransformPromotionError("profile_id must be a lowercase STRUM identifier")
    bundle, component_id, config = _candidate(experiment_dir)
    report_path = Path(evaluation_path)
    report = _require_report(report_path, bundle, component_id, config)
    if report["quality_gate"]["status"] != "passed":
        raise ChartTransformPromotionError(
            "transform held-out evaluation does not meet STRUM promotion quality policy"
        )
    component = bundle.component(component_id)
    assert (
        component is not None and component.checkpoint is not None and component.config is not None
    )
    output = Path(output_dir)
    if output.exists():
        raise ChartTransformPromotionError("promotion output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".strum-transform-promotion-", dir=output.parent))
    try:
        weights = staging / "weights" / "chart_transform.pt"
        model_config = staging / "configs" / "training-config.json"
        evaluation = staging / "evaluations" / "held-out.json"
        profile_config = staging / "profiles" / f"{profile_id}.json"
        evaluate_chart_transform_candidate(
            bundle_root=experiment_dir,
            dataset_manifest=dataset_manifest,
            output_path=evaluation,
            device=device,
            audio_manifest=audio_manifest,
        )
        recomputed_report = _require_report(evaluation, bundle, component_id, config)
        if recomputed_report["quality_gate"]["status"] != "passed":
            raise ChartTransformPromotionError(
                "transform independently recomputed evaluation does not meet STRUM promotion quality policy"
            )
        if _canonical_sha256(report) != _canonical_sha256(recomputed_report):
            raise ChartTransformPromotionError(
                "transform held-out evaluation report does not match independently recomputed evidence"
            )
        for source, destination in (
            (component.checkpoint, weights),
            (component.config, model_config),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        promoted = {
            "schema_version": 1,
            "format": PROFILE_FORMAT,
            "candidate_manifest_sha256": report["candidate_manifest_sha256"],
            "component_id": component_id,
            "component_sha256": _sha256(weights),
            "component_configuration_sha256": _sha256(model_config),
            "candidate_lineage_sha256": report["candidate_lineage_sha256"],
            "lineage": config["lineage"],
            "quality_policy": report["quality_policy"],
            "decoder_calibration": report["decoder_calibration"],
            "evaluation": {
                "path": "evaluations/held-out.json",
                "sha256": _sha256(evaluation),
                "byte_length": evaluation.stat().st_size,
            },
        }
        profile_config.parent.mkdir(parents=True, exist_ok=True)
        profile_config.write_text(
            json.dumps(promoted, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "model_id": bundle.model_id,
            "compatibility": bundle.compatibility,
            "components": {
                component_id: {
                    "checkpoint": "weights/chart_transform.pt",
                    "config": "configs/training-config.json",
                    "sha256": _sha256(weights),
                    "byte_length": weights.stat().st_size,
                    "config_sha256": _sha256(model_config),
                    "config_byte_length": model_config.stat().st_size,
                    "architecture": ARCHITECTURE,
                    "preprocessing": PREPROCESSING,
                }
            },
            "profiles": {
                profile_id: {
                    "capability": CAPABILITY,
                    "instruments": [config["instrument"]],
                    "required_components": [component_id],
                    "difficulty_policies": [f"learned:{component_id}"],
                    "configuration": f"profiles/{profile_id}.json",
                    "configuration_sha256": _sha256(profile_config),
                    "configuration_byte_length": profile_config.stat().st_size,
                }
            },
        }
        manifest_path = staging / MANIFEST_FILENAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        try:
            load_model_bundle(staging, check_files=True)
        except BundleValidationError as error:
            raise ChartTransformPromotionError(
                "promoted transform bundle failed verification"
            ) from error
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "promoted",
        "model_id": bundle.model_id,
        "profile_id": profile_id,
        "capability": CAPABILITY,
        "manifest_sha256": _sha256(output / MANIFEST_FILENAME),
        "evaluation_sha256": _sha256(output / "evaluations" / "held-out.json"),
    }


def validate_promoted_chart_transform_profile(bundle: ModelBundle, profile_id: str) -> None:
    """Verify the immutable admission proof before a transform can execute."""
    profile = bundle.profile(profile_id)
    if profile is None or profile.configuration is None or len(profile.required_components) != 1:
        raise BundleValidationError("difficulty transform profile lacks promotion configuration")
    component = bundle.component(profile.required_components[0])
    if component is None or component.sha256 is None or component.architecture != ARCHITECTURE:
        raise BundleValidationError("difficulty transform profile has invalid component")
    config = _read_json(profile.configuration, "difficulty transform promotion configuration")
    evaluation = config.get("evaluation") if isinstance(config.get("evaluation"), dict) else {}
    if component.config is None or component.config_sha256 is None:
        raise BundleValidationError(
            "difficulty transform profile has incomplete component configuration"
        )
    candidate_config = _read_json(component.config, "difficulty transform candidate configuration")
    try:
        lineage = _candidate_lineage(candidate_config)
    except ChartTransformPromotionError as error:
        raise BundleValidationError(
            "difficulty transform profile has invalid candidate lineage"
        ) from error
    try:
        _candidate_state(bundle, component.name, candidate_config)
    except ChartTransformPromotionError as error:
        raise BundleValidationError(
            "difficulty transform profile has invalid candidate checkpoint selection"
        ) from error
    if (
        set(config)
        != {
            "schema_version",
            "format",
            "candidate_manifest_sha256",
            "component_id",
            "component_sha256",
            "component_configuration_sha256",
            "candidate_lineage_sha256",
            "lineage",
            "quality_policy",
            "decoder_calibration",
            "evaluation",
        }
        or config.get("schema_version") != 1
        or config.get("format") != PROFILE_FORMAT
        or config.get("component_id") != component.name
        or config.get("component_sha256") != component.sha256
        or config.get("component_configuration_sha256") != component.config_sha256
        or config.get("candidate_lineage_sha256") != _canonical_sha256(lineage)
        or config.get("lineage") != lineage
        or config.get("quality_policy") != quality_policy_evidence()
        or not validate_calibration_evidence(config.get("decoder_calibration"))
        or config.get("decoder_calibration") != candidate_config.get("decoder_calibration")
        or config["decoder_calibration"].get("component_sha256") != component.sha256
        or config["decoder_calibration"].get("split_assignments_sha256")
        != lineage["split"]["assignments_sha256"]
        or config["decoder_calibration"].get("calibration_song_ids")
        != lineage["split"]["calibration_song_ids"]
        or set(evaluation) != {"path", "sha256", "byte_length"}
        or not isinstance(evaluation.get("path"), str)
        or Path(evaluation["path"]).is_absolute()
        or not isinstance(evaluation.get("sha256"), str)
        or len(evaluation["sha256"]) != 64
        or not isinstance(evaluation.get("byte_length"), int)
        or evaluation["byte_length"] < 0
    ):
        raise BundleValidationError("difficulty transform promotion configuration is invalid")
    report_path = (bundle.root / evaluation["path"]).resolve()
    try:
        report_path.relative_to(bundle.root)
    except ValueError as error:
        raise BundleValidationError(
            "difficulty transform evaluation escapes bundle root"
        ) from error
    if (
        not report_path.is_file()
        or _sha256(report_path) != evaluation["sha256"]
        or report_path.stat().st_size != evaluation["byte_length"]
    ):
        raise BundleValidationError("difficulty transform evaluation report does not match profile")
    report = _read_json(report_path, "difficulty transform evaluation report")
    required_report = {
        "schema_version",
        "format",
        "model_id",
        "candidate_manifest_sha256",
        "component_id",
        "component_sha256",
        "component_configuration_sha256",
        "candidate_lineage_sha256",
        "dataset_manifest_sha256",
        "dataset_records_sha256",
        "dataset_id",
        "task_view_id",
        "task_view_sha256",
        "split_assignments_sha256",
        "instrument",
        "source_difficulty",
        "target_difficulty",
        "split",
        "split_unit",
        "held_out_song_ids",
        "records_evaluated",
        "metrics",
        "decoder_calibration",
        "quality_policy",
        "quality_gate",
        "audio_manifest_sha256",
    }
    metrics = report.get("metrics")
    if (
        set(report) != required_report
        or report.get("schema_version") != 1
        or report.get("format") != EVALUATION_FORMAT
        or report.get("candidate_manifest_sha256") != config["candidate_manifest_sha256"]
        or report.get("component_id") != component.name
        or report.get("component_sha256") != component.sha256
        or report.get("component_configuration_sha256") != component.config_sha256
        or report.get("candidate_lineage_sha256") != config["candidate_lineage_sha256"]
        or report.get("dataset_manifest_sha256") != lineage["dataset"]["manifest_sha256"]
        or report.get("dataset_records_sha256") != lineage["dataset"]["records_sha256"]
        or report.get("dataset_id") != lineage["dataset"]["dataset_id"]
        or report.get("task_view_id") != lineage["task_view"]["task_view_id"]
        or report.get("task_view_sha256") != lineage["task_view"]["task_view_sha256"]
        or report.get("split_assignments_sha256") != lineage["split"]["assignments_sha256"]
        or report.get("held_out_song_ids") != lineage["split"]["test_song_ids"]
        or report.get("split") != "test"
        or report.get("split_unit") != "song_id"
        or not isinstance(report.get("records_evaluated"), int)
        or report["records_evaluated"] < 1
        or not isinstance(metrics, dict)
        or set(metrics) != {"loss", "lane_precision", "lane_recall", "lane_f1"}
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in metrics.values()
        )
        or not validate_quality_policy_evidence(
            report.get("quality_policy"), report.get("quality_gate"), metrics
        )
        or report.get("quality_policy") != config.get("quality_policy")
        or report.get("decoder_calibration") != config.get("decoder_calibration")
        or report.get("quality_gate", {}).get("status") != "passed"
    ):
        raise BundleValidationError("difficulty transform evaluation evidence is invalid")


def promoted_chart_transform_thresholds(bundle: ModelBundle, profile_id: str) -> tuple[float, ...]:
    """Return only the immutable calibrated decoder thresholds after validation."""
    validate_promoted_chart_transform_profile(bundle, profile_id)
    profile = bundle.profile(profile_id)
    assert profile is not None and profile.configuration is not None
    config = _read_json(profile.configuration, "difficulty transform promotion configuration")
    calibration = config["decoder_calibration"]
    assert isinstance(calibration, dict)
    thresholds = calibration["thresholds"]
    assert isinstance(thresholds, list)
    return tuple(thresholds)
