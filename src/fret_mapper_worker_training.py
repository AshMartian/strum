"""Worker-owned catalog training for learned Guitar and Bass fret mappers.

The mapper does not predict an entire chart.  It learns the final
Basic-Pitch-onset-to-five-lane mapping from an immutable catalog task view.
Its output is therefore an experiment component, not an auto-chart profile:
runtime integration requires separate, instrument-specific evaluation and
packaging before it can replace the established rule mapper.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import PROJECT_ROOT, __version__
from src.catalog_task_manifest import MANIFEST_FORMAT, resolve_catalog_task_manifest_songs
from src.model_bundle import MANIFEST_FILENAME
from src.song_source_catalog import CatalogValidationError

EXPERIMENT_FORMAT = "strum-experiment/v1"
PREPROCESSING_ID = "basic-pitch-onset-features/v1"
MODEL_IMPLEMENTATION = "FretMapperMLP/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TASKS = {
    "strum.fret-mapper/guitar/v1": ("fret_mapper_guitar", "guitar"),
    "strum.fret-mapper/bass/v1": ("fret_mapper_bass", "bass"),
}


class FretMapperTrainingError(ValueError):
    """Raised when a mapper worker request cannot safely start or package."""


@dataclass(frozen=True)
class FretMapperTrainingOptions:
    """Bounded mapper controls that do not expose worker-local paths."""

    model_id: str
    epochs: int = 30
    batch_size: int = 4096
    learning_rate: float = 0.001
    hidden: int = 256
    dropout: float = 0.2
    pos_weight_cap: float = 5.0
    device: str = "auto"
    max_songs: int = 0
    workers: int = 2
    onset_threshold: float = 0.5
    frame_threshold: float = 0.3
    min_note_length: int = 11
    seed: int = 42

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> FretMapperTrainingOptions:
        permitted = {
            "model_id",
            "epochs",
            "batch_size",
            "learning_rate",
            "hidden",
            "dropout",
            "pos_weight_cap",
            "device",
            "max_songs",
            "workers",
            "onset_threshold",
            "frame_threshold",
            "min_note_length",
            "seed",
        }
        if set(raw) - permitted:
            raise FretMapperTrainingError("unsupported fret-mapper training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise FretMapperTrainingError("fret-mapper model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        integer_bounds = {
            "epochs": (1, 30),
            "batch_size": (1, 4096),
            "hidden": (1, 256),
            "max_songs": (0, 0),
            "workers": (1, 2),
            "min_note_length": (1, 11),
            "seed": (0, 42),
        }
        for key, (minimum, default) in integer_bounds.items():
            value = raw.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise FretMapperTrainingError(f"fret-mapper {key} is invalid")
            values[key] = value
        float_bounds = {
            "learning_rate": (0.0, 0.001, False),
            "dropout": (0.0, 0.2, True),
            "pos_weight_cap": (0.0, 5.0, False),
            "onset_threshold": (0.0, 0.5, True),
            "frame_threshold": (0.0, 0.3, True),
        }
        for key, (minimum, default, allow_zero) in float_bounds.items():
            value = raw.get(key, default)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < minimum
                or (not allow_zero and value <= minimum)
                or (key in {"dropout", "onset_threshold", "frame_threshold"} and value > 1)
            ):
                raise FretMapperTrainingError(f"fret-mapper {key} is invalid")
            values[key] = float(value)
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise FretMapperTrainingError("fret-mapper device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "pos_weight_cap": self.pos_weight_cap,
            "device": self.device,
            "max_songs": self.max_songs,
            "workers": self.workers,
            "onset_threshold": self.onset_threshold,
            "frame_threshold": self.frame_threshold,
            "min_note_length": self.min_note_length,
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
    path: Path, catalog_root: Path, pipeline_id: str
) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    expected = _TASKS.get(pipeline_id)
    if expected is None:
        raise FretMapperTrainingError("unknown fret-mapper pipeline")
    task_kind, instrument = expected
    try:
        task_view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FretMapperTrainingError("fret-mapper task view is unreadable or invalid") from error
    task = task_view.get("task") if isinstance(task_view, dict) else None
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != task_kind
        or task.get("pipeline_id") != pipeline_id
        or task.get("instrument") != instrument
    ):
        raise FretMapperTrainingError("fret-mapper training requires its exact catalog task view")
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise FretMapperTrainingError("fret-mapper task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise FretMapperTrainingError(
            "fret-mapper task view requires non-empty train and val splits"
        )
    return task_view, songs, instrument


def _require_basic_pitch() -> None:
    if importlib.util.find_spec("basic_pitch") is None:
        raise FretMapperTrainingError("fret-mapper training requires the STRUM pitch extra")


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
        raise FretMapperTrainingError(
            "fret-mapper preprocessing or training script failed"
        ) from error


def _read_cache_split(path: Path) -> tuple[str, str]:
    try:
        with np.load(path, allow_pickle=False) as data:
            source_id = str(data["song_id"].item())
            split = str(data["split"].item())
            features = data["X"]
            labels = data["Y"]
    except (OSError, KeyError, ValueError) as error:
        raise FretMapperTrainingError("fret-mapper cache has invalid catalog metadata") from error
    if not source_id or split not in {"train", "val", "test"}:
        raise FretMapperTrainingError("fret-mapper cache has invalid catalog split metadata")
    if features.ndim != 2 or features.shape[1] != 95 or labels.shape != (features.shape[0], 5):
        raise FretMapperTrainingError("fret-mapper cache has incompatible feature or label shape")
    return source_id, split


def _validate_cache(cache_dir: Path, songs: list[dict[str, object]]) -> dict[str, int]:
    expected = {
        str(song["source_id"]): str(song["split"])
        for song in songs
        if isinstance(song.get("source_id"), str) and isinstance(song.get("split"), str)
    }
    counts = {"train": 0, "val": 0, "test": 0}
    seen: set[str] = set()
    for path in sorted(cache_dir.glob("*.npz")):
        source_id, split = _read_cache_split(path)
        if source_id in seen or expected.get(source_id) != split:
            raise FretMapperTrainingError("fret-mapper cache does not match the catalog task view")
        seen.add(source_id)
        counts[split] += 1
    if not counts["train"] or not counts["val"]:
        raise FretMapperTrainingError(
            "fret-mapper preprocessing did not produce train and val cache data"
        )
    return counts


def _read_metrics(path: Path) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FretMapperTrainingError("fret-mapper trainer did not write valid metrics") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("best_val_f1"), (int, float)):
        raise FretMapperTrainingError("fret-mapper trainer metrics are invalid")
    return {
        key: raw[key]
        for key in ("best_val_f1", "train_song_count", "val_song_count", "feature_dimension")
        if isinstance(raw.get(key), (int, float)) and not isinstance(raw[key], bool)
    }


def run_catalog_fret_mapper_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    pipeline_id: str,
    options: FretMapperTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Build a catalog-split mapper cache, train, and package an experiment component."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FretMapperTrainingError("fret-mapper output directory must be an empty directory")
    task_view, songs, instrument = _read_task_view(task_view_path, catalog_root, pipeline_id)
    _require_basic_pitch()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    checkpoint_dir = output_dir / "training-checkpoints"
    checkpoint_path = checkpoint_dir / "fret-mapper.pt"
    metrics_path = output_dir / "training-metrics.json"
    device = _resolve_device(options.device)
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_mapper_dataset.py"),
            "--catalog-manifest",
            str(task_view_path),
            "--catalog-root",
            str(catalog_root),
            "--cache-dir",
            str(cache_dir),
            "--workers",
            str(options.workers),
            "--onset-threshold",
            str(options.onset_threshold),
            "--frame-threshold",
            str(options.frame_threshold),
            "--min-note-length",
            str(options.min_note_length),
        ]
        + (["--max-songs", str(options.max_songs)] if options.max_songs else [])
    )
    cache_counts = _validate_cache(cache_dir, songs)
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_fret_mapper.py"),
            "--cache-dir",
            str(cache_dir),
            "--out",
            str(checkpoint_path),
            "--epochs",
            str(options.epochs),
            "--batch",
            str(options.batch_size),
            "--lr",
            str(options.learning_rate),
            "--hidden",
            str(options.hidden),
            "--dropout",
            str(options.dropout),
            "--pos-weight-cap",
            str(options.pos_weight_cap),
            "--seed",
            str(options.seed),
            "--device",
            device,
            "--use-catalog-splits",
            "--metrics-out",
            str(metrics_path),
        ]
    )
    if not checkpoint_path.is_file():
        raise FretMapperTrainingError("fret-mapper trainer did not produce a checkpoint")
    metrics = _read_metrics(metrics_path)
    component_id = f"fret_mapper.{instrument}"
    portable_config = {
        "schema_version": 1,
        "format": "strum-fret-mapper-model-config/v1",
        "instrument": instrument,
        "pipeline_id": pipeline_id,
        "model_implementation": MODEL_IMPLEMENTATION,
        "preprocessing": PREPROCESSING_ID,
        "feature_dimension": 95,
        "label_schema": "five-lane-fret-mapper-midi/v1",
        "basic_pitch": {
            "onset_threshold": options.onset_threshold,
            "frame_threshold": options.frame_threshold,
            "min_note_length": options.min_note_length,
        },
        "training": options.portable(),
    }
    bundle_dir = output_dir / "bundle"
    relative_checkpoint = Path("weights") / f"{instrument}-fret-mapper.pt"
    bundled_checkpoint = bundle_dir / relative_checkpoint
    bundled_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, bundled_checkpoint)
    relative_config = Path("configs") / f"{instrument}-fret-mapper-config.json"
    bundled_config = bundle_dir / relative_config
    bundled_config.parent.mkdir(parents=True, exist_ok=True)
    bundled_config.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {
        "manifest_schema": 1,
        "strum_version": f">={__version__}",
    }
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    manifest_path = bundle_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": options.model_id,
                "compatibility": compatibility,
                "components": {
                    component_id: {
                        "checkpoint": relative_checkpoint.as_posix(),
                        "config": relative_config.as_posix(),
                        "sha256": _sha256(bundled_checkpoint),
                        "byte_length": bundled_checkpoint.stat().st_size,
                        "config_sha256": _sha256(bundled_config),
                        "config_byte_length": bundled_config.stat().st_size,
                        "architecture": MODEL_IMPLEMENTATION,
                        "preprocessing": PREPROCESSING_ID,
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    task_songs = task_view.get("songs", [])
    lineage = [
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
    lineage_info = task_view.get("lineage")
    deployment_status = "requires_fret_mapper_profile_evaluation_and_packaging"
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": pipeline_id, "version": 1},
        "task_view": {
            "format": task_view.get("format"),
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage_info.get("catalog_id")
            if isinstance(lineage_info, dict)
            else None,
            "source_inputs": lineage,
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "configuration_sha256": _canonical_sha256(portable_config),
            "record_count": len(songs),
            "cache_counts": cache_counts,
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
        "metrics": metrics,
        "deployment_status": deployment_status,
        "model_bundle": {"model_id": options.model_id, "manifest_sha256": _sha256(manifest_path)},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": bundle_dir,
        "manifest_sha256": _sha256(manifest_path),
        "metrics": metrics,
        "deployment_status": deployment_status,
    }
