"""Catalog-backed Guitar/Bass section-classifier experiment training.

The existing section classifier learns a six-way *chart-pattern* label for a
two-second audio window.  It is not a song-structure model and it is not a
complete chart generator.  This module places that established label builder,
preprocessor, and CNN trainer behind STRUM's catalog worker boundary: labels
are derived only from the immutable task view's declared MIDI track and all
assets and splits are revalidated before training starts.

The resulting bundle deliberately declares no inference profile.  A future
section-routing evaluator must establish that this chart-derived taxonomy is
useful for a particular runtime before it can affect auto-chart output.
"""

from __future__ import annotations

import hashlib
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
from src.section_frontend import ROUTER_FEATURE_EXTRACTOR
from src.song_source_catalog import CatalogValidationError

EXPERIMENT_FORMAT = "strum-experiment/v1"
# This is the executable legacy router contract, not an approximation with a
# matching Mel shape.  Both inference and catalog preprocessing import it from
# ``src.section_frontend``.
PREPROCESSING_ID = "section-logmel-librosa-router-windows/v1"
MODEL_IMPLEMENTATION = "SectionClassifier/v1"
LABEL_FORMAT = "strum-section-labels/v1"
LABELS = ("silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed")
DEPLOYMENT_STATUS = "requires_section_profile_evaluation"
RUNTIME_PROFILE_REQUIREMENTS = (
    "section_router_profile_loader_tensor_only",
    "held_out_section_calibration_evaluation",
    "held_out_chart_impact_ablation",
)
# Retain this exported name for callers which inspect worker artifacts.  It is
# deliberately the same data object exported by the runtime frontend.
SECTION_FEATURE_EXTRACTOR = ROUTER_FEATURE_EXTRACTOR
_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TASKS = {
    "strum.section-classifier/guitar/v1": ("section_guitar", "guitar", "PART GUITAR"),
    "strum.section-classifier/bass/v1": ("section_bass", "bass", "PART BASS"),
}


class SectionTrainingError(ValueError):
    """Raised when a section training request cannot safely start or package."""


def _runtime_profile_requirements(instrument: str) -> tuple[str, ...]:
    if instrument not in {"guitar", "bass"}:
        raise SectionTrainingError("section runtime instrument is invalid")
    return (*RUNTIME_PROFILE_REQUIREMENTS, f"composed_{instrument}_chart_profile_contract")


@dataclass(frozen=True)
class SectionTrainingOptions:
    """Bounded options supported by the established section classifier."""

    model_id: str
    epochs: int = 15
    batch_size: int = 256
    learning_rate: float = 0.001
    num_workers: int = 0
    device: str = "auto"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> SectionTrainingOptions:
        permitted = {"model_id", "epochs", "batch_size", "learning_rate", "num_workers", "device"}
        if set(raw) - permitted:
            raise SectionTrainingError("unsupported section training option")
        model_id = raw.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise SectionTrainingError("section model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default in (("epochs", 15), ("batch_size", 256), ("num_workers", 0)):
            value = raw.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise SectionTrainingError(f"section {key} is invalid")
            if key != "num_workers" and value < 1:
                raise SectionTrainingError(f"section {key} is invalid")
            values[key] = value
        learning_rate = raw.get("learning_rate", 0.001)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise SectionTrainingError("section learning_rate is invalid")
        values["learning_rate"] = float(learning_rate)
        device = raw.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise SectionTrainingError("section device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_workers": self.num_workers,
            "device": self.device,
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
) -> tuple[dict[str, Any], list[dict[str, object]], str, str]:
    expected = _TASKS.get(pipeline_id)
    if expected is None:
        raise SectionTrainingError("unknown section-classifier pipeline")
    task_kind, instrument, label_track = expected
    try:
        task_view = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SectionTrainingError("section task view is unreadable or invalid") from error
    task = task_view.get("task") if isinstance(task_view, dict) else None
    label_schema = task.get("label_schema") if isinstance(task, dict) else None
    if (
        not isinstance(task_view, dict)
        or task_view.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != task_kind
        or task.get("pipeline_id") != pipeline_id
        or task.get("instrument") != instrument
        or not isinstance(label_schema, dict)
        or label_schema.get("id") != "midi-section-events/v1"
        or label_schema.get("track_prefixes") != [label_track]
        or label_schema.get("difficulty_encoding") != "not-applicable"
    ):
        raise SectionTrainingError("section training requires its exact catalog task view")
    try:
        songs = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise SectionTrainingError("section task view is invalid") from error
    split_counts = {
        split: sum(song.get("split") == split for song in songs) for split in ("train", "val")
    }
    if not split_counts["train"] or not split_counts["val"]:
        raise SectionTrainingError("section task view requires non-empty train and val splits")
    # The label builder deliberately refuses to merge matching tracks.  A
    # task view must therefore name one exact five-lane performance stream,
    # rather than relying on a legacy parser's fallback or on undefined union
    # behavior across alternate arrangements.
    if any(song.get("label_tracks") != [label_track] for song in songs):
        raise SectionTrainingError(
            "section task view requires exactly its declared five-lane label track"
        )
    return task_view, songs, instrument, label_track


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
        raise SectionTrainingError("section preprocessing or training script failed") from error


def _read_labels(
    labels_path: Path, task_view: dict[str, Any], songs: list[dict[str, object]]
) -> dict[str, int]:
    try:
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SectionTrainingError("section label builder did not write valid labels") from error
    lineage = labels.get("lineage") if isinstance(labels, dict) else None
    records = labels.get("records") if isinstance(labels, dict) else None
    task = task_view["task"]
    expected_splits = {
        str(song["source_id"]): str(song["split"])
        for song in songs
        if isinstance(song.get("source_id"), str) and isinstance(song.get("split"), str)
    }
    if (
        not isinstance(labels, dict)
        or labels.get("format") != LABEL_FORMAT
        or not isinstance(lineage, dict)
        or lineage.get("pipeline_id") != task.get("pipeline_id")
        or not isinstance(records, list)
    ):
        raise SectionTrainingError("section labels do not match the catalog task view")
    counts = {"train": 0, "val": 0, "test": 0}
    for record in records:
        if not isinstance(record, dict):
            raise SectionTrainingError("section labels contain an invalid record")
        source_id, split, label = record.get("source_id"), record.get("split"), record.get("label")
        if (
            not isinstance(source_id, str)
            or expected_splits.get(source_id) != split
            or label not in LABELS
            or not isinstance(record.get("t_start_s"), (int, float))
            or not isinstance(record.get("t_end_s"), (int, float))
        ):
            raise SectionTrainingError("section labels are not valid catalog-derived windows")
        counts[split] += 1
    if not counts["train"] or not counts["val"]:
        raise SectionTrainingError("section label builder produced no train or val windows")
    return counts


def _validate_cache(
    cache_dir: Path,
    songs: list[dict[str, object]],
    expected_counts: dict[str, int],
) -> dict[str, int]:
    expected_splits = {
        str(song["source_id"]): str(song["split"])
        for song in songs
        if isinstance(song.get("source_id"), str) and isinstance(song.get("split"), str)
    }
    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        if not expected_counts.get(split, 0):
            continue
        mel_path = cache_dir / f"{split}_section_mel.npy"
        label_path = cache_dir / f"{split}_section_label.npy"
        meta_path = cache_dir / f"{split}_section_meta.json"
        try:
            mel = np.load(mel_path, mmap_mode="r")
            targets = np.load(label_path, mmap_mode="r")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise SectionTrainingError(
                "section preprocessing did not produce a valid cache"
            ) from error
        if (
            mel.ndim != 3
            or mel.shape[1:] != (128, 87)
            or mel.dtype != np.dtype(np.float32)
            or targets.ndim != 1
            or len(mel) != len(targets)
            or not isinstance(metadata, list)
            or len(metadata) != len(targets)
            or not len(targets)
            or np.any(targets < 0)
            or np.any(targets >= len(LABELS))
        ):
            raise SectionTrainingError("section preprocessing cache is incompatible")
        for record in metadata:
            if (
                not isinstance(record, dict)
                or expected_splits.get(record.get("source_id")) != split
            ):
                raise SectionTrainingError("section cache does not match catalog task splits")
            if record.get("label") not in LABELS:
                raise SectionTrainingError("section cache contains unknown labels")
        counts[split] = len(targets)
        if counts[split] != expected_counts[split]:
            raise SectionTrainingError("section preprocessing omitted catalog label windows")
    return counts


def _read_metrics(path: Path) -> dict[str, object]:
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SectionTrainingError("section trainer did not write valid metrics") from error
    required = (
        "best_val_accuracy",
        "best_per_class_accuracy",
        "last_train_loss",
        "train_record_count",
        "val_record_count",
    )
    if not isinstance(metrics, dict) or any(key not in metrics for key in required):
        raise SectionTrainingError("section trainer metrics are incomplete")
    if not all(
        isinstance(metrics[key], (int, float)) and not isinstance(metrics[key], bool)
        for key in (
            "best_val_accuracy",
            "last_train_loss",
            "train_record_count",
            "val_record_count",
        )
    ) or not isinstance(metrics["best_per_class_accuracy"], dict):
        raise SectionTrainingError("section trainer metrics are invalid")
    held_out = metrics.get("held_out_evaluation")
    if held_out is not None:
        if not isinstance(held_out, dict) or held_out.get("status") not in {
            "completed",
            "not_run",
        }:
            raise SectionTrainingError("section trainer held-out evaluation is invalid")
        if held_out["status"] == "completed" and (
            held_out.get("split") != "test"
            or not isinstance(held_out.get("record_count"), int)
            or held_out["record_count"] < 1
            or not isinstance(held_out.get("accuracy"), (int, float))
            or isinstance(held_out.get("accuracy"), bool)
            or not isinstance(held_out.get("per_class_accuracy"), dict)
        ):
            raise SectionTrainingError("section trainer held-out evaluation is invalid")
    return {
        key: metrics[key] for key in (*required, "device", "held_out_evaluation") if key in metrics
    }


def _component(bundle_dir: Path, checkpoint_path: Path, config_path: Path) -> dict[str, object]:
    return {
        "checkpoint": checkpoint_path.as_posix(),
        "config": config_path.as_posix(),
        "sha256": _sha256(bundle_dir / checkpoint_path),
        "byte_length": (bundle_dir / checkpoint_path).stat().st_size,
        "config_sha256": _sha256(bundle_dir / config_path),
        "config_byte_length": (bundle_dir / config_path).stat().st_size,
        "architecture": MODEL_IMPLEMENTATION,
        "preprocessing": PREPROCESSING_ID,
    }


def run_catalog_section_training(
    *,
    task_view_path: Path,
    output_dir: Path,
    catalog_root: Path,
    pipeline_id: str,
    options: SectionTrainingOptions,
    strum_revision: str | None,
) -> dict[str, object]:
    """Build section labels/cache, train a classifier, and package an experiment only."""
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise SectionTrainingError("section output directory must be an empty directory")
    task_view, songs, instrument, _label_track = _read_task_view(
        task_view_path, catalog_root, pipeline_id
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "section-labels.json"
    cache_dir = output_dir / "cache"
    checkpoint_dir = output_dir / "training-checkpoints"
    metrics_path = output_dir / "training-metrics.json"
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_catalog_section_labels.py"),
            "--manifest",
            str(task_view_path),
            "--catalog-root",
            str(catalog_root),
            "--out",
            str(labels_path),
        ]
    )
    label_counts = _read_labels(labels_path, task_view, songs)
    device = _resolve_device(options.device)
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "preprocess_section_windows.py"),
            "--labels",
            str(labels_path),
            "--catalog-manifest",
            str(task_view_path),
            "--catalog-root",
            str(catalog_root),
            "--cache-dir",
            str(cache_dir),
        ]
    )
    cache_counts = _validate_cache(cache_dir, songs, label_counts)
    _run_script(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_section_classifier.py"),
            "--cache-dir",
            str(cache_dir),
            "--ckpt-dir",
            str(checkpoint_dir),
            "--epochs",
            str(options.epochs),
            "--batch-size",
            str(options.batch_size),
            "--lr",
            str(options.learning_rate),
            "--num-workers",
            str(options.num_workers),
            "--device",
            device,
            "--metrics-out",
            str(metrics_path),
            "--evaluate-split",
            "test" if cache_counts.get("test", 0) else "none",
        ]
    )
    checkpoint = checkpoint_dir / "best.pt"
    if not checkpoint.is_file():
        raise SectionTrainingError("section trainer did not produce a checkpoint")
    metrics = _read_metrics(metrics_path)
    bundle_dir = output_dir / "bundle"
    checkpoint_rel = Path("weights") / f"{instrument}-section-classifier.pt"
    bundled_checkpoint = bundle_dir / checkpoint_rel
    bundled_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, bundled_checkpoint)
    config_rel = Path("configs") / f"{instrument}-section-classifier.json"
    bundled_config = bundle_dir / config_rel
    bundled_config.parent.mkdir(parents=True, exist_ok=True)
    portable_config = {
        "schema_version": 1,
        "format": "strum-section-classifier-model-config/v1",
        "instrument": instrument,
        "pipeline_id": pipeline_id,
        "model_implementation": MODEL_IMPLEMENTATION,
        "preprocessing": PREPROCESSING_ID,
        "labels": list(LABELS),
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "mel_shape": [128, 87],
        # This is the actual shared runtime frontend.  The remaining
        # non-deployable gap is profile loading and measured chart utility,
        # not Mel-feature compatibility.
        "feature_extractor": dict(SECTION_FEATURE_EXTRACTOR),
        # Bind the raw candidate to one immutable catalog task view.  The
        # held-out evaluator and package gate must use this exact digest.
        "task_view_sha256": _sha256(task_view_path),
        "runtime_profile": {
            "format": "strum-section-router-deployment-requirements/v1",
            "status": "not_packageable",
            "reason": "section_router_execution_and_held_out_evaluation_not_proven",
            "requirements": list(_runtime_profile_requirements(instrument)),
        },
        "training": options.portable(),
    }
    bundled_config.write_text(
        json.dumps(portable_config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    component_id = f"section_classifier.{instrument}"
    compatibility: dict[str, object] = {"manifest_schema": 1, "strum_version": f">={__version__}"}
    if strum_revision:
        compatibility["strum_revision"] = strum_revision
    (bundle_dir / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": options.model_id,
                "compatibility": compatibility,
                "components": {component_id: _component(bundle_dir, checkpoint_rel, config_rel)},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    lineage = task_view.get("lineage")
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
        "pipeline": {"id": pipeline_id.removesuffix("/v1"), "version": 1},
        "task_view": {
            "format": task_view.get("format"),
            "sha256": _sha256(task_view_path),
            "catalog_id": lineage.get("catalog_id") if isinstance(lineage, dict) else None,
            "source_inputs": source_inputs,
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "configuration_sha256": _canonical_sha256(portable_config),
            "label_window_counts": label_counts,
            "cache_window_counts": cache_counts,
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
        "held_out_evaluation": metrics.get(
            "held_out_evaluation",
            {"status": "not_run", "reason": "trainer_did_not_report_held_out_evaluation"},
        ),
        "deployment_status": DEPLOYMENT_STATUS,
        "deployment_requirements": list(_runtime_profile_requirements(instrument)),
        "bundle": {"name": bundle_dir.name, "manifest": MANIFEST_FILENAME},
    }
    (output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "bundle_dir": str(bundle_dir),
        "metrics": metrics,
        "deployment_status": experiment["deployment_status"],
    }
