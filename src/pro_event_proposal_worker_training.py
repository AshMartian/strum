"""Catalog-backed raw Pro free-running event-proposal candidate training.

This is the first audio-only stage in the Pro path.  It samples both positive
and negative windows from approved catalog audio and does not consume authored
event times at inference.  The output is deliberately a profile-less research
candidate: it cannot emit MIDI or execute an auto-chart run.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.model_bundle import MANIFEST_FILENAME
from src.pro_event_proposal_preprocessing import (
    PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
    PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
    ProEventProposalPreprocessError,
    prepare_pro_event_proposal_windows,
)
from src.pro_event_worker_training import (
    _canonical_sha256,
    _resolve_device,
    _sha256,
    _source_inputs,
)
from src.pro_target_manifest import (
    PRO_TARGET_MANIFEST_FORMAT,
    resolve_catalog_pro_target_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError

EXPERIMENT_FORMAT = "strum-pro-event-proposal-candidate-experiment/v1"
MODEL_IMPLEMENTATION = "ProEventProposalCNN/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PIPELINE_BY_KIND = {
    "pro_guitar": "strum.instrument-chart/pro-guitar/v1",
    "pro_bass": "strum.instrument-chart/pro-bass/v1",
    "pro_keys": "strum.instrument-chart/pro-keys/v1",
}


class ProEventProposalTrainingError(ValueError):
    """Raised when an event-proposal candidate cannot prove its inputs."""


@dataclass(frozen=True)
class ProEventProposalTrainingOptions:
    model_id: str
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 0.0003
    device: str = "auto"
    limit_songs: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0
    seed: int = 20260822
    channels: int = 48
    negative_ratio: int = 4
    negative_exclusion_ms: int = 80

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ProEventProposalTrainingOptions:
        permitted = {
            "model_id",
            "epochs",
            "batch_size",
            "learning_rate",
            "device",
            "limit_songs",
            "max_train_batches",
            "max_val_batches",
            "seed",
            "channels",
            "negative_ratio",
            "negative_exclusion_ms",
        }
        if set(raw) - permitted:
            raise ProEventProposalTrainingError("unsupported Pro event proposal training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise ProEventProposalTrainingError("Pro event proposal model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default, minimum in (
            ("epochs", 25, 1),
            ("batch_size", 32, 1),
            ("limit_songs", 0, 0),
            ("max_train_batches", 0, 0),
            ("max_val_batches", 0, 0),
            ("seed", 20260822, 0),
            ("channels", 48, 1),
            ("negative_ratio", 4, 1),
            ("negative_exclusion_ms", 80, 0),
        ):
            value = raw.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ProEventProposalTrainingError(f"Pro event proposal {key} is invalid")
            if key == "negative_ratio" and value > 32:
                raise ProEventProposalTrainingError("Pro event proposal negative_ratio is invalid")
            if key == "negative_exclusion_ms" and value > 5000:
                raise ProEventProposalTrainingError(
                    "Pro event proposal negative_exclusion_ms is invalid"
                )
            values[key] = value
        learning_rate = raw.get("learning_rate", 0.0003)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise ProEventProposalTrainingError("Pro event proposal learning_rate is invalid")
        values["learning_rate"] = float(learning_rate)
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ProEventProposalTrainingError("Pro event proposal device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "limit_songs": self.limit_songs,
            "max_train_batches": self.max_train_batches,
            "max_val_batches": self.max_val_batches,
            "seed": self.seed,
            "channels": self.channels,
            "negative_ratio": self.negative_ratio,
            "negative_exclusion_ms": self.negative_exclusion_ms,
        }


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
        raise ProEventProposalTrainingError("Pro event proposal trainer failed") from error


def _read_task_view(path: Path, catalog_root: Path, pipeline_id: str) -> tuple[dict[str, Any], str]:
    try:
        task_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventProposalTrainingError("Pro target task view is unreadable") from error
    task_view = task_manifest.get("task_view") if isinstance(task_manifest, dict) else None
    task = task_view.get("task") if isinstance(task_view, dict) else None
    task_kind = task.get("kind") if isinstance(task, dict) else None
    if (
        not isinstance(task_manifest, dict)
        or task_manifest.get("format") != PRO_TARGET_MANIFEST_FORMAT
        or not isinstance(task_view, dict)
        or not isinstance(task, dict)
        or task_kind not in _PIPELINE_BY_KIND
        or _PIPELINE_BY_KIND[task_kind] != pipeline_id
        or task.get("pipeline_id") != pipeline_id
    ):
        raise ProEventProposalTrainingError(
            "Pro proposal requires the exact catalog target task view"
        )
    try:
        songs = resolve_catalog_pro_target_manifest_songs(task_manifest, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise ProEventProposalTrainingError(
            "Pro proposal task view cannot be revalidated"
        ) from error
    if not all(any(song.get("split") == split for song in songs) for split in ("train", "val")):
        raise ProEventProposalTrainingError(
            "Pro proposal requires non-empty train and val catalog splits"
        )
    return task_manifest, task_kind


def _history_metrics(path: Path) -> dict[str, object]:
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventProposalTrainingError("Pro proposal trainer did not write metrics") from error
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        raise ProEventProposalTrainingError("Pro proposal trainer metrics are invalid")
    expected = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_proposal_precision",
        "val_proposal_recall",
        "val_proposal_f1",
        "val_proposal_balanced_accuracy",
    }
    metrics = {
        key: value
        for key, value in history[-1].items()
        if key in expected and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not {"val_proposal_f1", "val_proposal_precision", "val_proposal_recall"} <= metrics.keys():
        raise ProEventProposalTrainingError(
            "Pro proposal trainer omitted held-out proposal metrics"
        )
    return metrics


def run_catalog_pro_event_proposal_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    pipeline_id: str,
    options: ProEventProposalTrainingOptions,
    strum_revision: str | None,
    strum_source_dirty: bool | None,
) -> dict[str, object]:
    """Train one raw event-proposal candidate; never an auto-chart profile."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ProEventProposalTrainingError(
            "Pro proposal output directory must be an empty directory"
        )
    task_manifest, task_kind = _read_task_view(task_view_path, catalog_root, pipeline_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "proposal-cache"
    try:
        cache_summary = prepare_pro_event_proposal_windows(
            manifest_path=task_view_path,
            catalog_root=catalog_root,
            cache_dir=cache_dir,
            splits=("train", "val"),
            limit_songs=options.limit_songs,
            negative_ratio=options.negative_ratio,
            negative_exclusion_ms=options.negative_exclusion_ms,
            negative_seed=options.seed,
        )
    except (CatalogValidationError, ProEventProposalPreprocessError) as error:
        raise ProEventProposalTrainingError("Pro proposal preprocessing failed") from error
    cache_preprocessing = cache_summary.get("preprocessing")
    if not isinstance(cache_preprocessing, dict):
        raise ProEventProposalTrainingError("Pro proposal preprocessing summary is invalid")
    negative_policy = cache_preprocessing.get("negative_policy")
    if (
        not isinstance(negative_policy, dict)
        or negative_policy.get("id") != PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID
    ):
        raise ProEventProposalTrainingError("Pro proposal negative policy is invalid")
    device = _resolve_device(options.device)
    checkpoints = output_dir / "training-checkpoints" / "pro_event_proposal"
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_pro_event_proposals.py"),
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-dir",
            str(checkpoints),
            "--task-kind",
            task_kind,
            "--epochs",
            str(options.epochs),
            "--batch-size",
            str(options.batch_size),
            "--learning-rate",
            str(options.learning_rate),
            "--device",
            device,
            "--max-train-batches",
            str(options.max_train_batches),
            "--max-val-batches",
            str(options.max_val_batches),
            "--seed",
            str(options.seed),
            "--channels",
            str(options.channels),
        ]
    )
    source_checkpoint = checkpoints / "best.pt"
    if not source_checkpoint.is_file():
        raise ProEventProposalTrainingError("Pro proposal trainer did not produce a checkpoint")
    component_id = f"pro.{task_kind.removeprefix('pro_')}.event_proposal"
    bundle_dir = output_dir / "bundle"
    checkpoint = bundle_dir / "weights" / f"{task_kind}-event-proposal.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)
    portable_config = {
        "schema_version": 1,
        "format": "strum-pro-event-proposal-candidate-config/v1",
        "task_kind": task_kind,
        "pipeline_id": pipeline_id,
        "model_implementation": MODEL_IMPLEMENTATION,
        "preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            # This is copied from the private cache summary, which includes
            # the task view's effective window geometry.  It contains only
            # identifiers, bounded options, and hashes/frames -- never paths.
            "negative_policy": negative_policy,
        },
        "input_contract": {
            "format": "strum-pro-arbitrary-audio-window/v1",
            "requires_midi_at_inference": False,
            "offline_window_scoring": True,
            "free_running_event_proposal": True,
            "sequence_decoding": False,
            "midi_emission": False,
        },
        "output_contract": {
            "format": "strum-pro-event-proposal-scores/v1",
            "event_attributes": False,
            "midi_emission": False,
        },
        "training": options.portable(),
    }
    config_path = bundle_dir / "configs" / f"{task_kind}-event-proposal.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {
        "manifest_schema": 1,
        "strum_version": f">={__version__}",
        "strum_source_dirty": strum_source_dirty,
    }
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    component = {
        "checkpoint": f"weights/{checkpoint.name}",
        "config": f"configs/{config_path.name}",
        "sha256": _sha256(checkpoint),
        "byte_length": checkpoint.stat().st_size,
        "config_sha256": _sha256(config_path),
        "config_byte_length": config_path.stat().st_size,
        "architecture": MODEL_IMPLEMENTATION,
        "preprocessing": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
    }
    (bundle_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": options.model_id,
                "compatibility": compatibility,
                "components": {component_id: component},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    task_view = task_manifest.get("task_view")
    lineage = task_view.get("lineage") if isinstance(task_view, dict) else None
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": pipeline_id, "version": 1},
        "candidate_scope": "free_running_audio_event_proposal_only",
        "task_view": {
            "format": PRO_TARGET_MANIFEST_FORMAT,
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage.get("catalog_id") if isinstance(lineage, dict) else None,
            "source_inputs": _source_inputs(task_manifest),
        },
        "preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "configuration_sha256": _canonical_sha256(cache_summary.get("preprocessing")),
            "negative_policy_id": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
            "cache_counts": cache_summary.get("splits"),
        },
        "configuration": {
            "sha256": _canonical_sha256(options.portable()),
            "values": options.portable(),
        },
        "checkpoint_mode": "fresh",
        "runtime": {
            "strum_version": __version__,
            "strum_revision": strum_revision,
            "strum_source_dirty": strum_source_dirty,
            "device": device,
        },
        "metrics": _history_metrics(checkpoints / "history.json"),
        "deployment_status": "not_deployable_requires_pro_sequence_evaluation_packaging_and_execution",
        "release_requirements": {
            "format": "strum-pro-chart-release-requirements/v1",
            "status": "blocked",
            "requirements": [
                "validated free-running proposal operating point with song-disjoint negative coverage",
                "variant-aware Pro sequence decoder with duration and chord constraints",
                "held-out chart-level evaluation against exact REAL_* MIDI",
                "typed Pro profile package and registered chart execution handler",
            ],
        },
        "bundle": {"name": bundle_dir.name, "manifest": MANIFEST_FILENAME},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": str(bundle_dir),
        "component_id": component_id,
        "metrics": experiment["metrics"],
        "deployment_status": experiment["deployment_status"],
        "runtime": experiment["runtime"],
    }
