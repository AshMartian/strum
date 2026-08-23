"""Worker-owned orchestration for the catalog-backed Guitar V1 trainer.

The mature Guitar preprocessing and two-stage trainers intentionally remain the
source of truth in :mod:`scripts.preprocess_guitar_windows` and
:mod:`scripts.train_guitar_v1`.  This module supplies their missing boundary:
it accepts a revalidated OCTAVE task view, runs those real scripts with
worker-owned locations, and packages their checkpoints with portable lineage.
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

import yaml

from src import PROJECT_ROOT, __version__
from src.catalog_guitar_manifest import MANIFEST_FORMAT, resolve_guitar_manifest_songs
from src.model_bundle import MANIFEST_FILENAME
from src.song_source_catalog import CatalogValidationError

PIPELINE_ID = "guitar.onset-fret/v1"
EXPERIMENT_FORMAT = "strum-experiment/v1"
PREPROCESSING_ID = "guitar-logmel-windows/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class GuitarTrainingError(ValueError):
    """Raised when the Guitar worker request cannot safely start or package."""


@dataclass(frozen=True)
class GuitarTrainingOptions:
    """Bounded knobs supported by the existing two-stage training scripts."""

    model_id: str
    epochs: int = 25
    batch_size: int = 128
    device: str = "auto"
    limit_songs: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> GuitarTrainingOptions:
        permitted = {"model_id", "epochs", "batch_size", "device", "limit_songs"}
        if set(raw) - permitted:
            raise GuitarTrainingError("unsupported Guitar training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise GuitarTrainingError("Guitar model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default in (("epochs", 25), ("batch_size", 128), ("limit_songs", 0)):
            value = raw.get(key, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if key != "limit_songs" else 0)
            ):
                raise GuitarTrainingError(f"Guitar {key} is invalid")
            values[key] = value
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise GuitarTrainingError("Guitar device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
        """Return reproducibility settings without the worker-local catalog location."""
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device,
            "limit_songs": self.limit_songs,
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
        raise GuitarTrainingError("Guitar task view is unreadable or invalid") from error
    if not isinstance(task_view, dict) or task_view.get("format") != MANIFEST_FORMAT:
        raise GuitarTrainingError("Guitar training requires a Guitar catalog task view")
    try:
        songs = resolve_guitar_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise GuitarTrainingError("Guitar task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise GuitarTrainingError("Guitar task view requires non-empty train and val splits")
    return task_view, songs


def _resolve_device(value: str) -> str:
    if value != "auto":
        return value
    # Keep the worker's discovery path import-free; PyTorch only belongs to a
    # real training request.
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _write_training_config(
    output_dir: Path, options: GuitarTrainingOptions
) -> tuple[Path, dict[str, object]]:
    try:
        base = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "guitar_v1.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise GuitarTrainingError("bundled Guitar training configuration is unreadable") from error
    if not isinstance(base, dict) or not isinstance(base.get("paths"), dict):
        raise GuitarTrainingError("bundled Guitar training configuration is invalid")
    config = base
    config["paths"] = dict(config["paths"])
    config["paths"].update(
        {
            "cache_dir": str(output_dir / "cache"),
            "checkpoint_dir": str(output_dir / "training-checkpoints"),
            "onset_subdir": "guitar_v1_onset",
            "fret_subdir": "guitar_v1_fret",
        }
    )
    config["wandb"] = dict(config.get("wandb", {}), enabled=False)
    config_path = output_dir / "runtime-training-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    portable = {
        "schema_version": 1,
        "format": "strum-guitar-training-config/v1",
        "pipeline_id": PIPELINE_ID,
        "preprocessing": PREPROCESSING_ID,
        "onset_model": config.get("onset", {}).get("model"),
        "fret_model": config.get("fret", {}).get("model"),
        "onset_inference": config.get("onset", {}).get("inference"),
        "training": options.portable(),
    }
    return config_path, portable


def _run_script(command: list[str]) -> None:
    """Run an established trainer without letting legacy output corrupt JSON."""
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
        raise GuitarTrainingError("Guitar preprocessing or training script failed") from error


def _component(
    bundle_dir: Path, path: Path, *, config_path: Path, architecture: str
) -> dict[str, object]:
    checkpoint = bundle_dir / path
    config = bundle_dir / config_path
    return {
        "checkpoint": path.as_posix(),
        "config": config_path.as_posix(),
        "sha256": _sha256(checkpoint),
        "byte_length": checkpoint.stat().st_size,
        "config_sha256": _sha256(config),
        "config_byte_length": config.stat().st_size,
        "architecture": architecture,
        "preprocessing": PREPROCESSING_ID,
    }


def _copy_checkpoint(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise GuitarTrainingError("Guitar trainer did not produce a required checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _history_metrics(path: Path) -> dict[str, object] | None:
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list) or not history:
        return None
    last = history[-1]
    if not isinstance(last, dict):
        return None
    return {
        key: last[key]
        for key in ("epoch", "train_loss", "val_loss", "val_f1")
        if isinstance(last.get(key), (int, float)) and not isinstance(last[key], bool)
    }


def run_catalog_guitar_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    options: GuitarTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Run both genuine Guitar training stages and package a portable bundle."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise GuitarTrainingError("Guitar output directory must be an empty directory")
    task_view, songs = _read_task_view(task_view_path, catalog_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, portable_config = _write_training_config(output_dir, options)
    device = _resolve_device(options.device)

    preprocess_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_guitar_windows.py"),
        "--manifest",
        str(task_view_path),
        "--catalog-root",
        str(catalog_root),
        "--cache-dir",
        str(output_dir / "cache"),
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
            str(PROJECT_ROOT / "scripts" / "train_guitar_v1.py"),
            "both",
            "--config",
            str(config_path),
            "--device",
            device,
            "--epochs",
            str(options.epochs),
            "--batch-size",
            str(options.batch_size),
        ]
    )

    checkpoint_root = output_dir / "training-checkpoints"
    bundle_dir = output_dir / "bundle"
    onset_weight = bundle_dir / "weights" / "guitar-onset.pt"
    fret_weight = bundle_dir / "weights" / "guitar-fret.pt"
    _copy_checkpoint(checkpoint_root / "guitar_v1_onset" / "best.pt", onset_weight)
    _copy_checkpoint(checkpoint_root / "guitar_v1_fret" / "best.pt", fret_weight)
    bundle_config = bundle_dir / "configs" / "guitar-training-config.json"
    bundle_config.parent.mkdir(parents=True, exist_ok=True)
    bundle_config.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {
        "manifest_schema": 1,
        "strum_version": f">={__version__}",
    }
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    relative_config = Path("configs/guitar-training-config.json")
    components = {
        "guitar.onset": _component(
            bundle_dir,
            Path("weights/guitar-onset.pt"),
            config_path=relative_config,
            architecture="GuitarOnsetCRNN/v1",
        ),
        "guitar.fret": _component(
            bundle_dir,
            Path("weights/guitar-fret.pt"),
            config_path=relative_config,
            architecture="GuitarFretClassifier/v1",
        ),
    }
    bundle_manifest = {
        "schema_version": 1,
        "model_id": options.model_id,
        "compatibility": compatibility,
        "components": components,
    }
    manifest_path = bundle_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
    histories = {
        "onset": _history_metrics(checkpoint_root / "guitar_v1_onset" / "history.json"),
        "fret": _history_metrics(checkpoint_root / "guitar_v1_fret" / "history.json"),
    }
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": "guitar.onset-fret", "version": 1},
        "task_view": {
            "format": task_view.get("format"),
            "sha256": _sha256(task_view_path),
            "catalog_id": task_view.get("catalog", {}).get("catalog_id")
            if isinstance(task_view.get("catalog"), dict)
            else None,
            "source_inputs": lineage,
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
        "metrics": histories,
        "model_bundle": {"model_id": options.model_id, "manifest_sha256": _sha256(manifest_path)},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": bundle_dir,
        "manifest_sha256": _sha256(manifest_path),
        "metrics": histories,
    }
