"""Held-out admission and immutable packaging for five-lane transforms.

The training script intentionally writes a *candidate* bundle with no profile.
This module is the only bridge from that candidate to the executable
``difficulty.transform/v1`` capability.  It recomputes metrics against the
candidate's declared song-disjoint validation split and copies the exact
weights, configuration, and report into a separate bundle.  No score threshold
is invented here: promotion remains an explicit release decision, while the
report gives OCTAVE the evidence needed to make that decision.
"""

from __future__ import annotations

import hashlib
import json
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
    _metrics,
    load_dataset,
    split_by_song,
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
    }
    if (
        set(config) != required
        or config.get("dataset_manifest") is not None
        or config.get("output_dir") is not None
    ):
        raise ChartTransformPromotionError("transform candidate configuration is unsupported")
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


def evaluate_chart_transform_candidate(
    *,
    bundle_root: str | Path,
    dataset_manifest: str | Path,
    output_path: str | Path,
    device: str = "cpu",
    audio_manifest: str | Path | None = None,
) -> dict[str, object]:
    """Recompute metrics on an immutable candidate's declared held-out split.

    The evaluator requires a dataset with explicit ``train`` and ``validation``
    song splits.  It refuses randomly re-split datasets, so the report is bound
    to a stable, song-disjoint holdout and cannot be manufactured by changing a
    local seed after training.
    """
    bundle, component_id, portable = _candidate(bundle_root)
    values = dict(portable)
    values.pop("instrument", None)
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
        if {pair.split for pair in pairs} != {"train", "validation"}:
            raise ChartTransformPromotionError(
                "held-out evaluation requires declared train and validation splits"
            )
        assets, audio_manifest_sha256 = _load_audio_assets(config, pairs)
        pairs = [replace(pair, audio_path=assets.get(pair.song_id)) for pair in pairs]
        train_pairs, validation_pairs = split_by_song(
            pairs, config.seed, config.validation_fraction
        )
        if not validation_pairs or not train_pairs:
            raise ChartTransformPromotionError(
                "held-out evaluation requires non-empty train and validation songs"
            )
        _, _, _ = _make_tensors(train_pairs, config)
        features, targets, _ = _make_tensors(validation_pairs, config)
    except DatasetValidationError as error:
        raise ChartTransformPromotionError("held-out transform dataset is invalid") from error
    model = _candidate_state(bundle, component_id, portable).to(config.device)
    metrics = _metrics(model, features.to(config.device), targets.to(config.device))
    component = bundle.component(component_id)
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
        "dataset_manifest_sha256": _sha256(Path(dataset_manifest)),
        "dataset_id": manifest["dataset_id"],
        "instrument": manifest["instrument"],
        "source_difficulty": config.source_difficulty,
        "target_difficulty": config.target_difficulty,
        "split": "validation",
        "split_unit": "song_id",
        "held_out_song_ids": sorted({pair.song_id for pair in validation_pairs}),
        "records_evaluated": len(validation_pairs),
        "metrics": metrics,
        "audio_manifest_sha256": audio_manifest_sha256,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {key: value for key, value in report.items() if key not in {"held_out_song_ids"}}


def _require_report(
    report_path: Path, bundle: ModelBundle, component_id: str, config: dict[str, Any]
) -> dict[str, Any]:
    report = _read_json(report_path, "transform held-out evaluation report")
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
        "dataset_manifest_sha256",
        "dataset_id",
        "instrument",
        "source_difficulty",
        "target_difficulty",
        "split",
        "split_unit",
        "held_out_song_ids",
        "records_evaluated",
        "metrics",
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
        or report.get("instrument") != config["instrument"]
        or report.get("source_difficulty") != config["source_difficulty"]
        or report.get("target_difficulty") != config["target_difficulty"]
        or report.get("split") != "validation"
        or report.get("split_unit") != "song_id"
        or not isinstance(report.get("held_out_song_ids"), list)
        or not report["held_out_song_ids"]
        or not isinstance(report.get("records_evaluated"), int)
        or report["records_evaluated"] < 1
        or not isinstance(metrics, dict)
        or set(metrics) != {"loss", "lane_precision", "lane_recall", "lane_f1"}
        or not all(isinstance(value, (int, float)) for value in metrics.values())
    ):
        raise ChartTransformPromotionError("transform held-out evaluation report is invalid")
    return report


def package_chart_transform_profile(
    *,
    experiment_dir: str | Path,
    evaluation_path: str | Path,
    output_dir: str | Path,
    profile_id: str,
) -> dict[str, object]:
    """Create an immutable executable profile from one evaluated raw candidate."""
    if not profile_id or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789.-" for character in profile_id
    ):
        raise ChartTransformPromotionError("profile_id must be a lowercase STRUM identifier")
    bundle, component_id, config = _candidate(experiment_dir)
    report_path = Path(evaluation_path)
    report = _require_report(report_path, bundle, component_id, config)
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
        for source, destination in (
            (component.checkpoint, weights),
            (component.config, model_config),
            (report_path, evaluation),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        promoted = {
            "schema_version": 1,
            "format": PROFILE_FORMAT,
            "candidate_manifest_sha256": report["candidate_manifest_sha256"],
            "component_id": component_id,
            "component_sha256": _sha256(weights),
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
    if (
        set(config)
        != {
            "schema_version",
            "format",
            "candidate_manifest_sha256",
            "component_id",
            "component_sha256",
            "evaluation",
        }
        or config.get("schema_version") != 1
        or config.get("format") != PROFILE_FORMAT
        or config.get("component_id") != component.name
        or config.get("component_sha256") != component.sha256
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
    if (
        report.get("format") != EVALUATION_FORMAT
        or report.get("candidate_manifest_sha256") != config["candidate_manifest_sha256"]
        or report.get("component_id") != component.name
        or report.get("component_sha256") != component.sha256
    ):
        raise BundleValidationError(
            "difficulty transform evaluation report is not bound to profile"
        )
