"""Catalog-backed five-lane Keys V1 experiment training.

This reuses the established five-lane CRNN *implementation*, not the Guitar
or Bass task identity.  The task view is revalidated against an OCTAVE catalog
and can only select ``PART KEYS`` labels.  The resulting components are
research artifacts: there is intentionally no Keys runtime profile or chart
handler in this module.
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
from src.catalog_task_manifest import (
    MANIFEST_FORMAT,
    resolve_catalog_task_manifest_songs,
    task_label_schema_is_supported,
)
from src.model_bundle import MANIFEST_FILENAME
from src.song_source_catalog import CatalogValidationError

PIPELINE_ID = "keys.onset-fret/v1"
TASK_KIND = "keys_onset_fret"
EXPERIMENT_FORMAT = "strum-experiment/v1"
PREPROCESSING_ID = "keys-logmel-windows/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class KeysTrainingError(ValueError):
    """Raised when a Keys request cannot safely start or package."""


@dataclass(frozen=True)
class KeysTrainingOptions:
    """Bounded controls supported by the existing five-lane trainers."""

    model_id: str
    epochs: int = 25
    batch_size: int = 128
    device: str = "auto"
    limit_songs: int = 0

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> KeysTrainingOptions:
        permitted = {"model_id", "epochs", "batch_size", "device", "limit_songs"}
        if set(raw) - permitted:
            raise KeysTrainingError("unsupported Keys training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise KeysTrainingError("Keys model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default in (("epochs", 25), ("batch_size", 128), ("limit_songs", 0)):
            value = raw.get(key, default)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < (1 if key != "limit_songs" else 0)
            ):
                raise KeysTrainingError(f"Keys {key} is invalid")
            values[key] = value
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise KeysTrainingError("Keys device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
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
        raise KeysTrainingError("Keys task view is unreadable or invalid") from error
    task = task_view.get("task") if isinstance(task_view, dict) else None
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != TASK_KIND
        or task.get("pipeline_id") != PIPELINE_ID
        or task.get("instrument") != "keys"
        or not task_label_schema_is_supported(TASK_KIND, task.get("label_schema"))
    ):
        raise KeysTrainingError("Keys training requires a Keys onset/fret catalog task view")
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise KeysTrainingError("Keys task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise KeysTrainingError("Keys task view requires non-empty train and val splits")
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


def _write_training_config(
    output_dir: Path, options: KeysTrainingOptions
) -> tuple[Path, dict[str, object]]:
    try:
        base = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "guitar_v1.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError) as error:
        raise KeysTrainingError("bundled five-lane training configuration is unreadable") from error
    if not isinstance(base, dict) or not isinstance(base.get("paths"), dict):
        raise KeysTrainingError("bundled five-lane training configuration is invalid")
    config = base
    config["paths"] = dict(config["paths"])
    config["paths"].update(
        {
            "cache_dir": str(output_dir / "cache"),
            "checkpoint_dir": str(output_dir / "training-checkpoints"),
            "onset_subdir": "keys_v1_onset",
            "fret_subdir": "keys_v1_fret",
        }
    )
    config["wandb"] = dict(
        config.get("wandb", {}), enabled=False, project="strum-keys", run_name=options.model_id
    )
    config_path = output_dir / "runtime-training-config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    portable = {
        "schema_version": 1,
        "format": "strum-keys-neural-model-config/v1",
        "instrument": "keys",
        "model_implementation": "five-lane-crnn/v1",
        "preprocessing": PREPROCESSING_ID,
        "audio": config.get("audio"),
        "onset_model": config.get("onset", {}).get("model"),
        "fret_model": config.get("fret", {}).get("model"),
        "onset_inference": config.get("onset", {}).get("inference"),
        "training": options.portable(),
    }
    return config_path, portable


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
        raise KeysTrainingError("Keys preprocessing or training script failed") from error


def _copy_checkpoint(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise KeysTrainingError("Keys trainer did not produce a required checkpoint")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


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


def _history_metrics(path: Path) -> dict[str, object] | None:
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        return None
    last = history[-1]
    return {
        key: last[key]
        for key in ("epoch", "train_loss", "val_loss", "val_f1")
        if isinstance(last.get(key), (int, float)) and not isinstance(last[key], bool)
    }


def run_catalog_keys_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    options: KeysTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Run five-lane Keys training and package a non-deployable experiment."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise KeysTrainingError("Keys output directory must be an empty directory")
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
        "--instrument",
        "keys",
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
    _copy_checkpoint(
        checkpoint_root / "keys_v1_onset" / "best.pt", bundle_dir / "weights/keys-onset.pt"
    )
    _copy_checkpoint(
        checkpoint_root / "keys_v1_fret" / "best.pt", bundle_dir / "weights/keys-fret.pt"
    )
    relative_config = Path("configs/keys-training-config.json")
    bundle_config = bundle_dir / relative_config
    bundle_config.parent.mkdir(parents=True, exist_ok=True)
    bundle_config.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {"manifest_schema": 1, "strum_version": f">={__version__}"}
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    components = {
        "keys.onset": _component(
            bundle_dir,
            Path("weights/keys-onset.pt"),
            config_path=relative_config,
            architecture="FiveLaneOnsetCRNN/v1",
        ),
        "keys.fret": _component(
            bundle_dir,
            Path("weights/keys-fret.pt"),
            config_path=relative_config,
            architecture="FiveLaneFretClassifier/v1",
        ),
    }
    manifest_path = bundle_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": options.model_id,
                "compatibility": compatibility,
                "components": components,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
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
        for song in task_view.get("songs", [])
        if isinstance(song, dict)
    ]
    histories = {
        "onset": _history_metrics(checkpoint_root / "keys_v1_onset" / "history.json"),
        "fret": _history_metrics(checkpoint_root / "keys_v1_fret" / "history.json"),
    }
    lineage_info = task_view.get("lineage")
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": "keys.onset-fret", "version": 1},
        "task_view": {
            "format": task_view.get("format"),
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage_info.get("catalog_id")
            if isinstance(lineage_info, dict)
            else None,
            "source_inputs": lineage,
            "profile_grade_admission": task_view.get("task", {}).get("profile_grade_admission"),
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
        "deployment_status": "requires_keys_profile_evaluation_and_packaging",
        "model_bundle": {"model_id": options.model_id, "manifest_sha256": _sha256(manifest_path)},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": bundle_dir,
        "manifest_sha256": _sha256(manifest_path),
        "metrics": histories,
        "deployment_status": "requires_keys_profile_evaluation_and_packaging",
    }
