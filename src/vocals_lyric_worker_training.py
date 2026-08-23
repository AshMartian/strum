"""Catalog-backed, experiment-only lead-Vocal lyric/alignment training.

The component learns observed text emitted by the exact ``PART VOCALS`` MIDI
track.  It is deliberately not a chart handler: text logits alone cannot
choose talkies, harmonies, phrase assembly, or a valid playable MIDI output.
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

PIPELINE_ID = "vocals.lyric-alignment/v1"
TASK_KIND = "vocals_lyric_alignment"
EXPERIMENT_FORMAT = "strum-vocals-lyric-alignment-experiment/v1"
PREPROCESSING_ID = "vocals-logmel-observed-lyric-ctc/v1"
COMPONENT_ID = "vocals.lyric_alignment"
_EXPECTED_LABEL_SCHEMA = {
    "id": "vocals-pitch-phrase-lyrics-midi/v1",
    "track_names": ["PART VOCALS"],
    "difficulty_encoding": "vocal-lyric-meta-events/v1",
}


class VocalLyricTrainingError(VocalsTrainingError):
    """Raised when lyric/alignment training cannot safely execute."""


def _read_task_view(
    path: Path, catalog_root: Path
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    try:
        task_view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalLyricTrainingError("Vocal lyric task view is unreadable or invalid") from error
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
        raise VocalLyricTrainingError(
            "Vocal lyric training requires a lyric-alignment catalog task view"
        )
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise VocalLyricTrainingError("Vocal lyric task view is invalid") from error
    by_split = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not by_split["train"] or not by_split["val"]:
        raise VocalLyricTrainingError(
            "Vocal lyric task view requires non-empty train and val splits"
        )
    return task_view, songs


def run_catalog_vocal_lyric_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    options: VocalsTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Train/package one lyric component and never a Vocal profile."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise VocalLyricTrainingError("Vocal lyric output directory must be an empty directory")
    task_view, songs = _read_task_view(task_view_path, catalog_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(options.device)
    cache_dir = output_dir / "cache"
    checkpoints = output_dir / "training-checkpoints" / "vocals_lyric_alignment"
    preprocess_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "preprocess_vocal_lyric_alignment.py"),
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
            str(PROJECT_ROOT / "scripts" / "train_vocal_lyric_alignment.py"),
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
        raise VocalLyricTrainingError("Vocal lyric trainer did not produce a required checkpoint")
    # Keep the decoder vocabulary with the checkpoint.  It is fixed model
    # semantics—not catalog text—and prevents a future evaluator from
    # silently assigning a different token to one of the CTC output columns.
    from scripts.preprocess_vocal_lyric_alignment import (  # noqa: PLC0415
        TOKENIZER_ID,
        VOCABULARY,
    )

    bundle_dir = output_dir / "bundle"
    checkpoint = bundle_dir / "weights" / "vocals-lyric-alignment.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)
    portable_config = {
        "schema_version": 1,
        "format": "strum-vocals-lyric-alignment-model-config/v1",
        "instrument": "vocals",
        "model_implementation": "VocalLyricCTC/v1",
        "preprocessing": PREPROCESSING_ID,
        "tokenizer": {"id": TOKENIZER_ID, "vocabulary": list(VOCABULARY)},
        "outputs": ["observed_lyric_character_tokens", "observed_lyric_event_alignment"],
        "alignment_source": "part-vocals-lyrics-or-text-meta-event-timestamps/v1",
        "excluded_outputs": ["pitch", "phrases", "talkies", "harmonies", "chart"],
        "training": options.portable(),
    }
    config_path = bundle_dir / "configs" / "vocals-lyric-alignment.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    compatibility: dict[str, object] = {"manifest_schema": 1, "strum_version": f">={__version__}"}
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    component = {
        "checkpoint": "weights/vocals-lyric-alignment.pt",
        "config": "configs/vocals-lyric-alignment.json",
        "sha256": _sha256(checkpoint),
        "byte_length": checkpoint.stat().st_size,
        "config_sha256": _sha256(config_path),
        "config_byte_length": config_path.stat().st_size,
        "architecture": "VocalLyricCTC/v1",
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
