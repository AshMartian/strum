"""Catalog-backed exact-Pro known-event attribute candidate training.

The candidate is deliberately *not* a free-running sequence model.  Its
feature windows are centered on supplied reference event times, so it can only
evaluate exact Pro attributes at an existing event.  It has no profile, MIDI
writer, or chart execution path.
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
from src.model_bundle import MANIFEST_FILENAME
from src.pro_audio_preprocessing import prepare_pro_audio_windows
from src.pro_target_manifest import (
    PRO_TARGET_MANIFEST_FORMAT,
    resolve_catalog_pro_target_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError

EXPERIMENT_FORMAT = "strum-pro-event-attribute-candidate-experiment/v1"
PREPROCESSING_ID = "pro-logmel-event-windows/v1"
MODEL_IMPLEMENTATION = "ProEventAttributeCNN/v1"
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_PIPELINE_BY_KIND = {
    "pro_guitar": "strum.instrument-chart/pro-guitar/v1",
    "pro_bass": "strum.instrument-chart/pro-bass/v1",
    "pro_keys": "strum.instrument-chart/pro-keys/v1",
}
_EXPECTED_LABEL_SCHEMAS = {
    "pro_guitar": {
        "id": "pro-string-fret-midi/v1",
        "track_names": ["PART REAL_GUITAR", "PART REAL_GUITAR_22"],
        "difficulty_encoding": "pro-string-note-offsets/v1",
    },
    "pro_bass": {
        "id": "pro-string-fret-midi/v1",
        "track_names": ["PART REAL_BASS", "PART REAL_BASS_22"],
        "difficulty_encoding": "pro-string-note-offsets/v1",
    },
    "pro_keys": {
        "id": "pro-keys-pitch-midi/v1",
        "track_names": ["PART REAL_KEYS_X"],
        "difficulty_encoding": "pro-keys-chromatic-notes-and-range/v1",
    },
}


class ProEventTrainingError(ValueError):
    """Raised when exact-Pro candidate training cannot prove its inputs."""


@dataclass(frozen=True)
class ProEventTrainingOptions:
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

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> ProEventTrainingOptions:
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
        }
        if set(raw) - permitted:
            raise ProEventTrainingError("unsupported Pro event candidate training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise ProEventTrainingError("Pro event candidate model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default, minimum in (
            ("epochs", 25, 1),
            ("batch_size", 32, 1),
            ("limit_songs", 0, 0),
            ("max_train_batches", 0, 0),
            ("max_val_batches", 0, 0),
            ("seed", 20260822, 0),
            ("channels", 48, 1),
        ):
            value = raw.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ProEventTrainingError(f"Pro event candidate {key} is invalid")
            values[key] = value
        learning_rate = raw.get("learning_rate", 0.0003)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise ProEventTrainingError("Pro event candidate learning_rate is invalid")
        values["learning_rate"] = float(learning_rate)
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ProEventTrainingError("Pro event candidate device is invalid")
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
        raise ProEventTrainingError("Pro event preprocessing or training script failed") from error


def _read_task_view(
    path: Path, catalog_root: Path, pipeline_id: str
) -> tuple[dict[str, Any], list[dict[str, object]], str]:
    try:
        task_manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventTrainingError("Pro target task view is unreadable") from error
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
        or task.get("label_schema") != _EXPECTED_LABEL_SCHEMAS[task_kind]
    ):
        raise ProEventTrainingError("Pro candidate requires the exact catalog target task view")
    try:
        songs = resolve_catalog_pro_target_manifest_songs(task_manifest, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise ProEventTrainingError("Pro target task view cannot be revalidated") from error
    if not all(any(song.get("split") == split for song in songs) for split in ("train", "val")):
        raise ProEventTrainingError("Pro candidate requires non-empty train and val catalog splits")
    return task_manifest, songs, task_kind


def _history_metrics(path: Path) -> dict[str, object]:
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventTrainingError("Pro candidate trainer did not write metrics") from error
    if not isinstance(history, list) or not history or not isinstance(history[-1], dict):
        raise ProEventTrainingError("Pro candidate trainer metrics are invalid")
    expected = {
        "epoch",
        "train_loss",
        "val_loss",
        "val_known_event_token_f1",
        "val_known_event_state_accuracy",
        "val_known_event_exact_accuracy",
    }
    metrics = {
        key: value
        for key, value in history[-1].items()
        if key in expected and isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not {"val_known_event_token_f1", "val_known_event_exact_accuracy"} <= metrics.keys():
        raise ProEventTrainingError("Pro candidate trainer omitted held-out known-event metrics")
    return metrics


def _source_inputs(task_manifest: dict[str, Any]) -> list[dict[str, object]]:
    """Copy only immutable source IDs/hashes from the portable target view."""
    task_view = task_manifest.get("task_view")
    task_songs = task_view.get("songs") if isinstance(task_view, dict) else None
    if not isinstance(task_songs, list):
        raise ProEventTrainingError("Pro target task view has no source lineage")
    result: list[dict[str, object]] = []
    for song in task_songs:
        audio = song.get("audio") if isinstance(song, dict) else None
        midi = song.get("notes_midi") if isinstance(song, dict) else None
        if (
            not isinstance(song, dict)
            or not isinstance(song.get("source_id"), str)
            or song.get("split") not in {"train", "val", "test"}
            or not isinstance(audio, dict)
            or not isinstance(midi, dict)
            or not isinstance(audio.get("sha256"), str)
            or not isinstance(midi.get("sha256"), str)
        ):
            raise ProEventTrainingError("Pro target task source lineage is invalid")
        result.append(
            {
                "source_id": song["source_id"],
                "split": song["split"],
                "audio_sha256": audio["sha256"],
                "notes_midi_sha256": midi["sha256"],
            }
        )
    if not result or len({item["source_id"] for item in result}) != len(result):
        raise ProEventTrainingError("Pro target task source lineage is incomplete")
    return result


def run_catalog_pro_event_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    pipeline_id: str,
    options: ProEventTrainingOptions,
    strum_revision: str | None,
    strum_source_dirty: bool | None,
) -> dict[str, object]:
    """Train a raw exact-Pro event-attribute candidate; never a chart profile."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ProEventTrainingError("Pro candidate output directory must be an empty directory")
    task_manifest, songs, task_kind = _read_task_view(task_view_path, catalog_root, pipeline_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_summary = prepare_pro_audio_windows(
        manifest_path=task_view_path,
        catalog_root=catalog_root,
        cache_dir=cache_dir,
        splits=("train", "val"),
        limit_songs=options.limit_songs,
    )
    device = _resolve_device(options.device)
    checkpoints = output_dir / "training-checkpoints" / "pro_event_attributes"
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_pro_event_attributes.py"),
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
        raise ProEventTrainingError("Pro candidate trainer did not produce a checkpoint")
    component_id = f"pro.{task_kind.removeprefix('pro_')}.event_attributes"
    bundle_dir = output_dir / "bundle"
    checkpoint = bundle_dir / "weights" / f"{task_kind}-event-attributes.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_checkpoint, checkpoint)
    target_contract: dict[str, object] = (
        {
            "kind": "pro_string_fret_technique/v1",
            "string_count": 6,
            "fret_range": [0, 22],
            "techniques": [
                "normal",
                "arpeggio_form",
                "bent",
                "muted",
                "tapped",
                "harmonic",
                "pinch_harmonic",
            ],
            "track_variant_head": ["standard", "22_fret"],
        }
        if task_kind in {"pro_guitar", "pro_bass"}
        else {
            "kind": "pro_keys_pitch_channel_range_shift/v1",
            "pitch_range": [48, 72],
            "channel_metadata": "retained_in_labels_not_predicted/v1",
            "range_state_head": ["none", "C", "D", "E", "F", "G", "A"],
        }
    )
    portable_config = {
        "schema_version": 1,
        "format": "strum-pro-event-attribute-candidate-config/v1",
        "task_kind": task_kind,
        "pipeline_id": pipeline_id,
        "model_implementation": MODEL_IMPLEMENTATION,
        "preprocessing": PREPROCESSING_ID,
        "input_contract": {
            "format": "strum-pro-known-reference-event-window/v1",
            "event_time_source": "held_out_catalog_label_only",
            "free_running_event_proposal": False,
            "sequence_decoding": False,
            "midi_emission": False,
        },
        "target_contract": target_contract,
        "training": options.portable(),
    }
    config_path = bundle_dir / "configs" / f"{task_kind}-event-attributes.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # A commit ID by itself is not source provenance: a training worktree can
    # contain uncommitted implementation changes.  Keep the dirty state next
    # to the revision in the portable artifact, including ``None`` when this
    # runtime was unable to inspect Git.  That makes an unknown state explicit
    # rather than accidentally presenting the commit as a clean build.
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
        "preprocessing": PREPROCESSING_ID,
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
    raw_task_view = task_manifest.get("task_view")
    lineage = raw_task_view.get("lineage") if isinstance(raw_task_view, dict) else None
    source_inputs = _source_inputs(task_manifest)
    experiment = {
        "schema_version": 1,
        "format": EXPERIMENT_FORMAT,
        "lifecycle": "completed",
        "pipeline": {"id": pipeline_id, "version": 1},
        "candidate_scope": "known_reference_event_attributes_only",
        "task_view": {
            "format": PRO_TARGET_MANIFEST_FORMAT,
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage.get("catalog_id") if isinstance(lineage, dict) else None,
            "source_inputs": source_inputs,
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "configuration_sha256": _canonical_sha256(cache_summary.get("preprocessing")),
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
        "deployment_status": "not_deployable_requires_pro_event_proposal_sequence_evaluation_and_packaging",
        "release_requirements": {
            "format": "strum-pro-chart-release-requirements/v1",
            "status": "blocked",
            "requirements": [
                "free_running_Pro event proposal with negative-event coverage",
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
        "metrics": experiment["metrics"],
        "deployment_status": experiment["deployment_status"],
        "runtime": experiment["runtime"],
    }
