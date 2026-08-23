"""Catalog-backed, experiment-only Vocal frame-activity training.

The worker owns catalog revalidation and all local path resolution.  Its
single component is intentionally not a chart profile: a full playable vocal
chart also needs phrases, lyrics, pitchless sections, and evaluation gates.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.catalog_task_manifest import MANIFEST_FORMAT, resolve_catalog_task_manifest_songs
from src.model_bundle import MANIFEST_FILENAME
from src.song_source_catalog import CatalogValidationError

PIPELINE_ID = "vocals.note-activity/v1"
TASK_KIND = "vocals_activity"
EXPERIMENT_FORMAT = "strum-vocals-activity-experiment/v1"
PREPROCESSING_ID = "vocals-logmel-frame-targets/v1"
COMPONENT_ID = "vocals.frame_activity_pitch"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class VocalsTrainingError(ValueError):
    """Raised when a Vocal worker request cannot safely start or package."""


@dataclass(frozen=True)
class VocalsTrainingOptions:
    """Bounded local-training settings for the Vocal activity experiment."""

    model_id: str
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 0.0003
    device: str = "auto"
    limit_songs: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0
    seed: int = 20260822

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> VocalsTrainingOptions:
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
        }
        if set(raw) - permitted:
            raise VocalsTrainingError("unsupported Vocal training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise VocalsTrainingError("Vocal model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        integer_options = (
            ("epochs", 25, 1),
            ("batch_size", 16, 1),
            ("limit_songs", 0, 0),
            ("max_train_batches", 0, 0),
            ("max_val_batches", 0, 0),
            ("seed", 20260822, 0),
        )
        for key, default, minimum in integer_options:
            value = raw.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise VocalsTrainingError(f"Vocal {key} is invalid")
            values[key] = value
        learning_rate = raw.get("learning_rate", 0.0003)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise VocalsTrainingError("Vocal learning_rate is invalid")
        values["learning_rate"] = float(learning_rate)
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise VocalsTrainingError("Vocal device is invalid")
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
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_task_view(
    path: Path, catalog_root: Path
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    try:
        task_view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalsTrainingError("Vocal task view is unreadable or invalid") from error
    task = task_view.get("task") if isinstance(task_view, dict) else None
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != TASK_KIND
        or task.get("pipeline_id") != PIPELINE_ID
        or task.get("instrument") != "vocals"
        or task.get("label_schema", {}).get("track_prefixes") != ["PART VOCALS"]
    ):
        raise VocalsTrainingError("Vocal training requires a Vocal activity catalog task view")
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise VocalsTrainingError("Vocal task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise VocalsTrainingError("Vocal task view requires non-empty train and val splits")
    return task_view, songs


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
        raise VocalsTrainingError("Vocal preprocessing or training script failed") from error


def _history_metrics(path: Path) -> dict[str, object] | None:
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        return None
    return {
        key: history[-1][key]
        for key in ("epoch", "train_loss", "val_loss", "val_activity_f1", "val_pitch_accuracy")
        if isinstance(history[-1].get(key), (int, float)) and not isinstance(history[-1][key], bool)
    }


def run_catalog_vocals_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    options: VocalsTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Run genuine local preprocessing/training and package no runtime profile."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise VocalsTrainingError("Vocal output directory must be an empty directory")
    task_view, songs = _read_task_view(task_view_path, catalog_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(options.device)
    cache_dir = output_dir / "cache"
    checkpoints = output_dir / "training-checkpoints" / "vocals_frame_activity"
    preprocess_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_vocals_frames.py"),
        "--manifest",
        str(task_view_path),
        "--catalog-root",
        str(catalog_root),
        "--cache-dir",
        str(cache_dir),
        "--splits",
        "train",
        "val",
    ]
    if options.limit_songs:
        preprocess_command.extend(["--limit-songs", str(options.limit_songs)])
    _run_script(preprocess_command)
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_vocals_activity.py"),
            "--cache-dir",
            str(cache_dir),
            "--checkpoint-dir",
            str(checkpoints),
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
        ]
    )
    source_checkpoint = checkpoints / "best.pt"
    if not source_checkpoint.is_file():
        raise VocalsTrainingError("Vocal trainer did not produce a required checkpoint")
    bundle_dir = output_dir / "bundle"
    checkpoint = bundle_dir / "weights" / "vocals-frame-activity.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)
    portable_config = {
        "schema_version": 1,
        "format": "strum-vocals-frame-activity-model-config/v1",
        "instrument": "vocals",
        "model_implementation": "VocalFrameCNN/v1",
        "preprocessing": PREPROCESSING_ID,
        "outputs": ["pitched_vocal_activity", "midi_pitch_36_84"],
        "excluded_outputs": ["lyrics", "phrases", "talkies", "harmonies", "chart"],
        "training": options.portable(),
    }
    config_path = bundle_dir / "configs" / "vocals-frame-activity.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {
        "manifest_schema": 1,
        "strum_version": f">={__version__}",
    }
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    component = {
        "checkpoint": "weights/vocals-frame-activity.pt",
        "config": "configs/vocals-frame-activity.json",
        "sha256": _sha256(checkpoint),
        "byte_length": checkpoint.stat().st_size,
        "config_sha256": _sha256(config_path),
        "config_byte_length": config_path.stat().st_size,
        "architecture": "VocalFrameCNN/v1",
        "preprocessing": PREPROCESSING_ID,
    }
    (bundle_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": options.model_id,
                "compatibility": compatibility,
                "components": {COMPONENT_ID: component},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lineage = task_view.get("lineage")
    task_songs = task_view.get("songs", [])
    source_inputs = [
        {
            "source_id": song.get("source_id"),
            "split": song.get("split"),
            "audio_sha256": song.get("audio", {}).get("sha256")
            if isinstance(song.get("audio"), dict)
            else None,
            "notes_midi_sha256": song.get("notes_midi", {}).get("sha256")
            if isinstance(song.get("notes_midi"), dict)
            else None,
        }
        for song in task_songs
        if isinstance(song, dict)
    ]
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": "vocals.note-activity", "version": 1},
        "task_view": {
            "format": task_view.get("format"),
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage.get("catalog_id") if isinstance(lineage, dict) else None,
            "source_inputs": source_inputs,
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "configuration_sha256": _canonical_sha256(portable_config),
            "record_count": len(songs),
        },
        "configuration": {
            "sha256": _canonical_sha256(options.portable()),
            "values": options.portable(),
        },
        "checkpoint_mode": "fresh",
        "runtime": {
            "strum_version": __version__,
            "strum_revision": strum_revision,
            "device": device,
        },
        "metrics": _history_metrics(checkpoints / "history.json"),
        "deployment_status": "requires_vocals_profile_evaluation_and_packaging",
        "bundle": {"name": bundle_dir.name, "manifest": MANIFEST_FILENAME},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": str(bundle_dir),
        "metrics": experiment["metrics"],
        "deployment_status": experiment["deployment_status"],
    }
