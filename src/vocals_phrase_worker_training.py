"""Catalog-backed, experiment-only lead-Vocal phrase-boundary training.

This component learns observed 105/106-or-105-span phrase boundaries only.  A
Vocal auto-chart profile remains unavailable until STRUM has compatible lyric,
talky, harmony, composition, and held-out chart-evaluation stages.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.catalog_task_manifest import MANIFEST_FORMAT, resolve_catalog_task_manifest_songs
from src.model_bundle import MANIFEST_FILENAME
from src.song_source_catalog import CatalogValidationError
from src.vocals_worker_training import (
    VocalsTrainingError,
    VocalsTrainingOptions,
    _canonical_sha256,
    _history_metrics,
    _resolve_device,
    _run_script,
    _sha256,
)

PIPELINE_ID = "vocals.phrase-boundaries/v1"
TASK_KIND = "vocals_phrase_boundaries"
EXPERIMENT_FORMAT = "strum-vocals-phrase-boundaries-experiment/v1"
PREPROCESSING_ID = "vocals-logmel-phrase-boundaries/v1"
COMPONENT_ID = "vocals.phrase_boundaries"
_EXPECTED_LABEL_SCHEMA = {
    "id": "vocals-pitch-phrase-lyrics-midi/v1",
    "track_names": ["PART VOCALS"],
    "difficulty_encoding": "vocal-phrase-boundary-events/v1",
}


class VocalPhraseTrainingError(VocalsTrainingError):
    """Raised when a phrase-boundary experiment cannot safely execute."""


def _read_task_view(
    path: Path, catalog_root: Path
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    try:
        task_view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalPhraseTrainingError("Vocal phrase task view is unreadable or invalid") from error
    task = task_view.get("task") if isinstance(task_view, dict) else None
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != TASK_KIND
        or task.get("pipeline_id") != PIPELINE_ID
        or task.get("instrument") != "vocals"
        or task.get("label_schema") != _EXPECTED_LABEL_SCHEMA
    ):
        raise VocalPhraseTrainingError(
            "Vocal phrase training requires a phrase-boundary catalog task view"
        )
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise VocalPhraseTrainingError("Vocal phrase task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise VocalPhraseTrainingError(
            "Vocal phrase task view requires non-empty train and val splits"
        )
    return task_view, songs


def run_catalog_vocal_phrase_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    options: VocalsTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Train and package a real component, never an inference profile."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise VocalPhraseTrainingError("Vocal phrase output directory must be an empty directory")
    task_view, songs = _read_task_view(task_view_path, catalog_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(options.device)
    cache_dir = output_dir / "cache"
    checkpoints = output_dir / "training-checkpoints" / "vocals_phrase_boundaries"
    preprocess_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_vocal_phrase_boundaries.py"),
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
            str(PROJECT_ROOT / "scripts" / "train_vocal_phrase_boundaries.py"),
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
        raise VocalPhraseTrainingError("Vocal phrase trainer did not produce a required checkpoint")
    bundle_dir = output_dir / "bundle"
    checkpoint = bundle_dir / "weights" / "vocals-phrase-boundaries.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)
    portable_config = {
        "schema_version": 1,
        "format": "strum-vocals-phrase-boundaries-model-config/v1",
        "instrument": "vocals",
        "model_implementation": "VocalPhraseBoundaryCNN/v1",
        "preprocessing": PREPROCESSING_ID,
        "outputs": ["lead_phrase_start", "lead_phrase_end"],
        "label_conventions": [
            "midi_105_start_marker",
            "midi_106_end_marker",
            "midi_105_sustained_span_end",
        ],
        "excluded_outputs": ["lyrics", "talkies", "harmonies", "chart"],
        "training": options.portable(),
    }
    config_path = bundle_dir / "configs" / "vocals-phrase-boundaries.json"
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
        "checkpoint": "weights/vocals-phrase-boundaries.pt",
        "config": "configs/vocals-phrase-boundaries.json",
        "sha256": _sha256(checkpoint),
        "byte_length": checkpoint.stat().st_size,
        "config_sha256": _sha256(config_path),
        "config_byte_length": config_path.stat().st_size,
        "architecture": "VocalPhraseBoundaryCNN/v1",
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
        for song in task_view.get("songs", [])
        if isinstance(song, dict)
    ]
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": PIPELINE_ID, "version": 1},
        "task_view": {
            "catalog_id": task_view.get("catalog", {}).get("catalog_id"),
            "manifest_sha256": task_view.get("lineage", {}).get("task_view_sha256"),
            "source_inputs": source_inputs,
            "record_count": len(songs),
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
        "deployment_status": "requires_vocal_chart_composition_evaluation_and_packaging",
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
