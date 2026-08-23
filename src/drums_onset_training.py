"""Worker adapter for catalog-backed Drums onset-classifier experiments.

The onset classifier is an existing, cache-based trainer.  This module is a
thin orchestration boundary around it: it resolves a signed catalog task view,
uses STRUM's existing onset-window preprocessor, then invokes the existing
trainer with an owned configuration.  Its checkpoint is deliberately recorded
as a *training* artifact, not as a deployable model bundle; the current
Drums V14 inference profile does not consume an onset-classifier checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT
from src.catalog_drums_manifest import MANIFEST_FORMAT, task_view_sha256
from src.song_source_catalog import CatalogValidationError

EXPERIMENT_FORMAT = "strum-drums-onset-experiment/v1"
TRAINING_PROFILE = "onset_classifier_v2"


class DrumsTrainingError(ValueError):
    """Raised when an onset-classifier experiment cannot be prepared safely."""


@dataclass(frozen=True)
class DrumsTrainingOptions:
    """Validated, worker-owned configuration for the established trainer."""

    model_id: str
    profile: str = TRAINING_PROFILE
    seed: int = 20260813
    batch_size: int = 256
    epochs: int = 100
    learning_rate: float = 0.001
    max_train_batches: int = 2000
    max_test_batches: int = 500
    num_workers: int = 0
    strum_revision: str | None = None

    @classmethod
    def from_mapping(cls, raw: object) -> DrumsTrainingOptions:
        if not isinstance(raw, dict):
            raise DrumsTrainingError("training options must be an object")
        permitted = {
            "model_id",
            "profile",
            "seed",
            "batch_size",
            "epochs",
            "learning_rate",
            "max_train_batches",
            "max_test_batches",
            "num_workers",
            "strum_revision",
        }
        if set(raw) - permitted:
            raise DrumsTrainingError("unsupported Drums training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not model_id:
            raise DrumsTrainingError("Drums training requires model_id")

        defaults = cls(model_id=model_id)
        values: dict[str, Any] = {"model_id": model_id}
        for key in (
            "profile",
            "seed",
            "batch_size",
            "epochs",
            "learning_rate",
            "max_train_batches",
            "max_test_batches",
            "num_workers",
            "strum_revision",
        ):
            values[key] = raw.get(key, getattr(defaults, key))

        if values["profile"] != TRAINING_PROFILE:
            raise DrumsTrainingError("unsupported Drums training profile")
        if isinstance(values["seed"], bool) or not isinstance(values["seed"], int):
            raise DrumsTrainingError("seed must be an integer")
        for key in ("batch_size", "epochs", "max_train_batches", "max_test_batches"):
            if isinstance(values[key], bool) or not isinstance(values[key], int) or values[key] < 1:
                raise DrumsTrainingError(f"{key} must be a positive integer")
        if (
            isinstance(values["num_workers"], bool)
            or not isinstance(values["num_workers"], int)
            or values["num_workers"] < 0
        ):
            raise DrumsTrainingError("num_workers must be a non-negative integer")
        if (
            not isinstance(values["learning_rate"], (float, int))
            or isinstance(values["learning_rate"], bool)
            or values["learning_rate"] <= 0
        ):
            raise DrumsTrainingError("learning_rate must be positive")
        if values["strum_revision"] is not None and not isinstance(values["strum_revision"], str):
            raise DrumsTrainingError("strum_revision must be a string")
        values["learning_rate"] = float(values["learning_rate"])
        return cls(**values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_task_view(task_view: Path) -> dict[str, Any]:
    try:
        raw = json.loads(task_view.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DrumsTrainingError("Drums task view is unreadable or invalid JSON") from error
    if not isinstance(raw, dict) or raw.get("format") != MANIFEST_FORMAT:
        raise DrumsTrainingError("task view is not a Drums onset catalog manifest")
    catalog = raw.get("catalog")
    task = raw.get("task")
    if (
        not isinstance(catalog, dict)
        or not isinstance(catalog.get("catalog_id"), str)
        or not isinstance(catalog.get("content_sha256"), str)
        or not isinstance(task, dict)
        or task.get("pipeline_id") != "drums.onset-classifier"
        or task.get("pipeline_version") != 1
    ):
        raise DrumsTrainingError("Drums task view is missing required lineage")
    return raw


def _load_training_config(output: Path, options: DrumsTrainingOptions) -> Any:
    """Build an owned trainer config from STRUM's maintained V2 baseline."""
    from omegaconf import OmegaConf  # Imported only for an actual job.

    baseline = PROJECT_ROOT / "configs" / "onset_classifier.yaml"
    try:
        config = OmegaConf.load(baseline)
    except OSError as error:
        raise DrumsTrainingError("Drums onset baseline configuration is unavailable") from error
    config.paths.data_dir = str(output)
    config.paths.output_dir = str(output)
    config.paths.checkpoint_dir = str(output / "checkpoints")
    config.cache_dir = str(output / "cache")
    config.training.batch_size = options.batch_size
    config.training.epochs = options.epochs
    config.training.learning_rate = options.learning_rate
    config.training.max_train_batches = options.max_train_batches
    # The established trainer calls this legacy setting `max_val_batches`,
    # although its evaluator reads test_index.json.
    config.training.max_val_batches = options.max_test_batches
    config.training.num_workers = options.num_workers
    OmegaConf.save(config, output / "training-config.yaml")
    return config


def _prepare_cache(task_view: Path, output: Path, catalog_root: Path) -> list[dict[str, Any]]:
    """Run the established catalog-aware onset-window preparation for every split."""
    from scripts.preprocess_onset_windows import extract_split  # noqa: PLC0415

    cache = output / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    indexes: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        extract_split(task_view, split, cache, catalog_root=catalog_root)
        index_path = cache / f"{split}_index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DrumsTrainingError(
                f"Drums catalog task view produced no usable {split} onset windows"
            ) from error
        if (
            not isinstance(index, dict)
            or not isinstance(index.get("total_onsets"), int)
            or index["total_onsets"] < 1
        ):
            raise DrumsTrainingError(f"Drums catalog task view has no usable {split} onset windows")
        lineage = index.get("lineage")
        if not isinstance(lineage, dict):
            raise DrumsTrainingError("Drums preprocessing did not record catalog lineage")
        indexes.append(
            {
                "split": split,
                "index_name": index_path.name,
                "index_sha256": _sha256(index_path),
                "total_onsets": index["total_onsets"],
                "lineage": lineage,
            }
        )
    return indexes


def _experiment_payload(
    *,
    task_view: dict[str, Any],
    options: DrumsTrainingOptions,
    indexes: list[dict[str, Any]],
    config_path: Path,
    checkpoint: Path,
    metrics: dict[str, object],
) -> dict[str, object]:
    """Create a portable experiment ledger; it intentionally has no source paths."""
    return {
        "format": EXPERIMENT_FORMAT,
        "pipeline": {"id": "drums.onset-classifier/v1", "version": 1},
        "model": {
            "id": options.model_id,
            "profile": options.profile,
            "architecture": "OnsetClassifier/v2",
        },
        "task_view": {
            "format": task_view["format"],
            "sha256": task_view_sha256(task_view),
            "catalog_id": task_view["catalog"]["catalog_id"],
            "catalog_content_sha256": task_view["catalog"]["content_sha256"],
        },
        "preprocessing": {
            "id": "drums-onset-windows/v1",
            "splits": indexes,
        },
        "training": {
            "config_name": config_path.name,
            "config_sha256": _sha256(config_path),
            "seed": options.seed,
            "batch_size": options.batch_size,
            "epochs": options.epochs,
            "learning_rate": options.learning_rate,
            "max_train_batches": options.max_train_batches,
            "max_test_batches": options.max_test_batches,
            "num_workers": options.num_workers,
            "strum_revision": options.strum_revision,
        },
        "checkpoint": {
            "name": checkpoint.relative_to(config_path.parent).as_posix(),
            "sha256": _sha256(checkpoint),
            "byte_length": checkpoint.stat().st_size,
            "format": "torch-training-checkpoint/v1",
            "deployment_status": "requires_profile_packaging",
        },
        "metrics": metrics,
    }


def _run_existing_trainer(config: Any, options: DrumsTrainingOptions) -> dict[str, object]:
    """Seed and invoke the maintained trainer without changing its learning loop."""
    # The trainer owns torch and device selection; seed all public RNGs before it starts.
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from scripts.train_onset_classifier import train  # noqa: PLC0415

    np.random.seed(options.seed)
    torch.manual_seed(options.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(options.seed)
    result = train(config)
    if not isinstance(result, dict):
        raise DrumsTrainingError("Drums trainer did not report an experiment result")
    return result


def run_drums_onset_training(
    task_view_path: str | Path,
    output_dir: str | Path,
    raw_options: object,
    *,
    catalog_root: str | Path,
) -> dict[str, object]:
    """Execute the established Drums classifier trainer from a catalog task view."""
    task_view_path, output = Path(task_view_path).resolve(), Path(output_dir).resolve()
    options = DrumsTrainingOptions.from_mapping(raw_options)
    if not isinstance(catalog_root, (str, Path)) or not str(catalog_root):
        raise DrumsTrainingError("Drums training requires worker-local catalog_root")
    task_view = _load_task_view(task_view_path)
    output.mkdir(parents=True, exist_ok=True)

    try:
        indexes = _prepare_cache(task_view_path, output, Path(catalog_root))
    except CatalogValidationError as error:
        raise DrumsTrainingError("catalog no longer matches the Drums task view") from error
    config = _load_training_config(output, options)
    trainer_result = _run_existing_trainer(config, options)
    checkpoint = Path(str(trainer_result.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise DrumsTrainingError("Drums trainer did not produce a best checkpoint")
    metrics = trainer_result.get("metrics")
    if not isinstance(metrics, dict):
        raise DrumsTrainingError("Drums trainer did not report final metrics")
    config_path = output / "training-config.yaml"
    experiment = _experiment_payload(
        task_view=task_view,
        options=options,
        indexes=indexes,
        config_path=config_path,
        checkpoint=checkpoint,
        metrics=metrics,
    )
    experiment_path = output / "experiment.json"
    experiment_path.write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "status": "completed",
        "pipeline_id": "drums.onset-classifier/v1",
        "model_id": options.model_id,
        "experiment_name": experiment_path.name,
        "task_view_sha256": experiment["task_view"]["sha256"],
        "checkpoint": experiment["checkpoint"],
        "metrics": metrics,
    }
