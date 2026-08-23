"""Versioned, machine-readable STRUM worker contract.

This entry point intentionally exposes discovery and safe preflight first. It
does not import the mutable auto-chart pipeline while OCTAVE is deciding which
runtime or model bundle to use. Training and chart execution are added as
pipeline handlers behind the same protocol rather than requiring callers to
know STRUM's source-tree scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.catalog_chart_pairs import CatalogChartPairOptions, prepare_catalog_chart_pairs
from src.catalog_drums_manifest import build_drums_manifest, write_drums_manifest
from src.catalog_guitar_manifest import build_guitar_manifest, write_guitar_manifest
from src.catalog_task_manifest import (
    PIPELINE_IDS as CATALOG_TASK_PIPELINES,
)
from src.catalog_task_manifest import (
    build_catalog_task_manifest,
    write_catalog_task_manifest,
)
from src.model_bundle import BundleValidationError, InferenceProfile, ModelBundle, load_model_bundle
from src.song_source_catalog import CatalogValidationError, load_catalog

PROTOCOL_VERSION = 1
MODEL_BUNDLE_SCHEMA_VERSIONS = (1,)


@dataclass(frozen=True)
class PipelineDescriptor:
    """A STRUM-owned pipeline capability visible to host applications."""

    id: str
    display_name: str
    kind: str
    version: int
    catalog_requirements: dict[str, object]
    prepare_schema: dict[str, object]
    train_schema: dict[str, object] | None
    checkpoint_outputs: tuple[str, ...]
    inference_capability: str | None
    status: str
    preparation_status: str
    training_status: str

    def as_json(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_outputs"] = list(self.checkpoint_outputs)
        return data


def _object_schema(
    properties: dict[str, object], *, required: tuple[str, ...] = ()
) -> dict[str, object]:
    """Return the small JSON-Schema subset exposed to the OCTAVE renderer."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


CATALOG_AUDIO_OPTIONS = {
    "audio_role": {"type": "string"},
    "fallback_audio_role": {"type": ["string", "null"]},
    "required_difficulty": {"type": "string", "default": "expert"},
}
CHART_TRANSFORM_PREPARE_SCHEMA = _object_schema(
    {
        "instrument": {"type": "string", "enum": ["guitar", "bass", "keys", "drums"]},
        "target_difficulty": {"type": "string", "enum": ["Hard", "Medium", "Easy"]},
        "split_seed": {"type": "integer", "default": 20260814},
        "validation_fraction": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
        "dataset_id": {"type": "string"},
        "overwrite": {"type": "boolean", "default": False},
    },
    required=("instrument", "target_difficulty"),
)
CHART_TRANSFORM_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "seed": {"type": "integer", "default": 20260813},
        "validation_fraction": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
        "lane_count": {"type": "integer", "minimum": 1, "default": 5},
        "alignment_tolerance_ms": {"type": "number", "minimum": 0, "default": 50},
        "hidden_dim": {"type": "integer", "minimum": 1, "default": 32},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "epochs": {"type": "integer", "minimum": 1, "default": 20},
        "device": {"type": "string", "default": "auto"},
        "audio_feature_mode": {"type": "string", "enum": ["none"]},
        "audio_sample_rate": {"type": "integer", "minimum": 1, "default": 16000},
        "audio_window_ms": {"type": "number", "exclusiveMinimum": 0, "default": 50},
        "audio_max_duration_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 900},
        "strum_revision": {"type": "string"},
    },
    required=("model_id",),
)
GUITAR_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 128},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
    },
    required=("model_id",),
)
DRUMS_ONSET_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "profile": {
            "type": "string",
            "enum": ["onset_classifier_v2"],
            "default": "onset_classifier_v2",
        },
        "seed": {"type": "integer", "default": 20260813},
        "batch_size": {"type": "integer", "minimum": 1, "default": 256},
        "epochs": {"type": "integer", "minimum": 1, "default": 100},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "max_train_batches": {"type": "integer", "minimum": 1, "default": 2000},
        "max_test_batches": {"type": "integer", "minimum": 1, "default": 500},
        "num_workers": {"type": "integer", "minimum": 0, "default": 0},
        "strum_revision": {"type": "string"},
    },
    required=("model_id",),
)
)

PIPELINES = (
    PipelineDescriptor(
        id="guitar.onset-fret/v1",
        display_name="Guitar onset + fret",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "guitar",
            "difficulties": ["expert"],
            "audio_roles": ["guitar", "mix"],
            "audio_policy": "prefer:guitar,fallback:mix",
        },
        prepare_schema=_object_schema(CATALOG_AUDIO_OPTIONS),
        train_schema=GUITAR_TRAIN_SCHEMA,
        checkpoint_outputs=("guitar.onset", "guitar.fret"),
        inference_capability="guitar.audio_to_chart/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
    ),
    PipelineDescriptor(
        id="chart_transform.five_lane/v1",
        display_name="Learned five-lane difficulty transform",
        kind="chart_to_chart",
        version=1,
        catalog_requirements={
            "instruments": ["guitar", "bass", "keys", "drums"],
            "source_difficulty": "expert",
            "target_difficulties": ["hard", "medium", "easy"],
        },
        prepare_schema=CHART_TRANSFORM_PREPARE_SCHEMA,
        train_schema=CHART_TRANSFORM_TRAIN_SCHEMA,
        checkpoint_outputs=("chart_transform",),
        inference_capability="difficulty.transform/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
    ),
    PipelineDescriptor(
        id="drums.onset-classifier/v1",
        display_name="Drums onset + velocity",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "drums",
            "difficulties": ["expert"],
            "audio_roles": ["drums", "mix"],
            "audio_policy": "prefer:drums,fallback:mix",
        },
        prepare_schema=_object_schema(CATALOG_AUDIO_OPTIONS),
        train_schema=DRUMS_ONSET_TRAIN_SCHEMA,
        checkpoint_outputs=("drums_onset_classifier",),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
    ),
    *(
        PipelineDescriptor(
            id=pipeline_id,
            display_name=task_kind.replace("_", " ").title(),
            kind="derived_labels"
            if task_kind.startswith(("fret_mapper", "section_"))
            else "chart_to_chart",
            version=1,
            catalog_requirements={
                "instrument": task_kind.replace("fret_mapper_", "").replace("section_", ""),
                "difficulties": ["expert"],
                "audio_policy": "task-specific managed role with mix fallback",
            },
            prepare_schema=_object_schema(
                {
                    **CATALOG_AUDIO_OPTIONS,
                    "disable_fallback": {"type": "boolean", "default": False},
                    "split_ratios": {"type": "array", "items": {"type": "integer"}},
                    "split_seed": {"type": "string", "default": "catalog-source-id/v1"},
                    "preprocessing": {"type": "object", "default": {}},
                }
            ),
            train_schema=None,
            checkpoint_outputs=(task_kind,),
            inference_capability=None,
            status="catalog_ready",
            preparation_status="available",
            training_status="planned",
        )
        for task_kind, pipeline_id in sorted(CATALOG_TASK_PIPELINES.items())
    ),
)


class WorkerRequestError(ValueError):
    """Raised for an invalid host-to-worker request before any task is written."""


def _revision() -> tuple[str | None, bool | None]:
    """Return a best-effort revision without making Git a runtime dependency."""
    configured = os.environ.get("STRUM_SOURCE_REVISION", "").strip()
    if configured:
        return configured, os.environ.get("STRUM_SOURCE_DIRTY", "0") == "1"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _runtime_payload() -> dict[str, object]:
    revision, dirty = _revision()
    capabilities = [
        "pipeline_discovery",
        "catalog_inspect",
        "dataset_prepare",
        "chart_preflight",
        "chart_run",
        "model_bundle_preflight",
        "checkpoint_inspect",
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runtime": {
            "id": f"strum-{__version__}" + (f"+git.{revision[:12]}" if revision else ""),
            "version": __version__,
            "source_revision": revision,
            "source_dirty": dirty,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "python_requires": ">=3.11",
        "device_support": ["cuda", "mps", "cpu"],
        "capabilities": capabilities,
        "model_bundle_schema_versions": list(MODEL_BUNDLE_SCHEMA_VERSIONS),
        "optional_dependencies": {
            "basic_pitch": {
                "available": importlib.util.find_spec("basic_pitch") is not None,
                "required_by": ["guitar.hybrid-v2-rule/v1"],
            }
        },
        "pipelines": [
            pipeline.id for pipeline in PIPELINES if pipeline.preparation_status == "available"
        ],
    }


def _manifest_sha256(bundle: ModelBundle) -> str | None:
    if bundle.manifest_path is None:
        return None
    return hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_bundle(
    path: str | Path,
    *,
    required_components: Sequence[str] = (),
) -> dict[str, object]:
    """Validate one portable bundle without deserializing any model weights.

    Explicitly selected bundles require an on-disk checkpoint, SHA-256, and
    byte length for every required component. This is deliberately stricter
    than the legacy fallback layout, which is not deployable through this API.
    """
    bundle = load_model_bundle(path, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    selected = tuple(required_components) or tuple(
        name for name, component in bundle.components.items() if component.required
    )
    components: list[dict[str, object]] = []
    for name in selected:
        component = bundle.component(name)
        if component is None:
            errors.append(f"required component is not declared: {name}")
            continue
        if component.checkpoint is None:
            errors.append(f"{name}: required component has no checkpoint")
            continue
        if component.sha256 is None:
            errors.append(f"{name}: deployable component requires sha256")
        if component.byte_length is None:
            errors.append(f"{name}: deployable component requires byte_length")
        if component.config is not None and component.config_sha256 is None:
            errors.append(f"{name}: deployable component config requires sha256")
        if component.config is not None and component.config_byte_length is None:
            errors.append(f"{name}: deployable component config requires byte_length")
        components.append(
            {
                "id": name,
                "sha256": component.sha256,
                "byte_length": component.byte_length,
                "architecture": component.architecture,
                "preprocessing": component.preprocessing,
                "config_sha256": component.config_sha256,
                "config_byte_length": component.config_byte_length,
            }
        )
    if errors:
        raise BundleValidationError("; ".join(errors))
    return {
        "status": "ready",
        "model_id": bundle.model_id,
        "manifest_sha256": _manifest_sha256(bundle),
        "components": components,
        "compatibility": bundle.compatibility,
    }


def validate_inference_profile(
    path: str | Path,
    *,
    profile_id: str,
    difficulty_policy: str,
) -> dict[str, object]:
    """Resolve a declared inference profile and verify all required model files."""
    bundle = load_model_bundle(path, check_files=True)
    profile: InferenceProfile | None = bundle.profile(profile_id)
    if profile is None:
        raise BundleValidationError(f"inference profile is not declared: {profile_id}")
    if difficulty_policy not in profile.difficulty_policies:
        raise BundleValidationError(
            f"profile {profile_id} does not support difficulty policy {difficulty_policy}"
        )
    plan = preflight_bundle(path, required_components=profile.required_components)
    return {
        **plan,
        "profile_id": profile.profile_id,
        "capability": profile.capability,
        "instruments": list(profile.instruments),
        "difficulty_policy": difficulty_policy,
        "profile_configuration_sha256": profile.configuration_sha256,
        "profile_configuration_byte_length": profile.configuration_byte_length,
    }


def preflight_chart_request(request_path: Path) -> dict[str, object]:
    """Resolve a profile for a future chart job without loading model weights.

    This is intentionally a preflight-only boundary until each production
    auto-chart backend consumes declared bundle components rather than legacy
    working-directory defaults. A caller must not treat preflight success as an
    authorization to execute the legacy pipeline with arbitrary checkpoints.
    """
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart request is unreadable or not valid JSON") from error
    expected = {"model_root", "profile_id", "difficulty_policy", "instruments", "device"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WorkerRequestError("chart preflight request has unsupported fields")
    if not all(
        isinstance(raw[key], str) and raw[key]
        for key in ("model_root", "profile_id", "difficulty_policy", "device")
    ):
        raise WorkerRequestError("chart preflight identity fields must be non-empty strings")
    instruments = raw["instruments"]
    if (
        not isinstance(instruments, list)
        or not instruments
        or not all(isinstance(instrument, str) and instrument for instrument in instruments)
        or len(set(instruments)) != len(instruments)
    ):
        raise WorkerRequestError(
            "chart preflight instruments must be a unique non-empty string list"
        )
    plan = validate_inference_profile(
        raw["model_root"],
        profile_id=raw["profile_id"],
        difficulty_policy=raw["difficulty_policy"],
    )
    if not set(instruments) <= set(plan["instruments"]):
        raise WorkerRequestError("profile does not cover requested instruments")
    profile_configuration_sha256 = None
    if plan["capability"] == "guitar.hybrid-v2-rule/v1":
        from src.inference.guitar_hybrid_profile import (
            load_guitar_hybrid_rule_profile,  # noqa: PLC0415
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_guitar_hybrid_rule_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "drums.v14-expert/v1":
        from src.inference.drums_v14_profile import (  # noqa: PLC0415
            load_drums_v14_expert_profile,
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_drums_v14_expert_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "difficulty.transform/v1":
        if (
            len(plan["components"]) != 1
            or plan["components"][0]["architecture"] != "EventTransformMLP/v1"
        ):
            raise WorkerRequestError(
                "difficulty transform requires exactly one EventTransformMLP/v1"
            )
        component_id = plan["components"][0]["id"]
        if plan["difficulty_policy"] != f"learned:{component_id}":
            raise WorkerRequestError("difficulty transform policy must name its declared component")
    return {
        "status": "ready",
        "execution": "available"
        if plan["capability"]
        in {"guitar.hybrid-v2-rule/v1", "drums.v14-expert/v1", "difficulty.transform/v1"}
        else "not_available",
        "model_id": plan["model_id"],
        "profile_id": plan["profile_id"],
        "capability": plan["capability"],
        "difficulty_policy": plan["difficulty_policy"],
        "instruments": instruments,
        "device": raw["device"],
        "manifest_sha256": plan["manifest_sha256"],
        "components": plan["components"],
        "profile_configuration_sha256": profile_configuration_sha256,
        "profile_configuration_byte_length": plan["profile_configuration_byte_length"],
    }


def _read_chart_run_request(request_path: Path, capability: str) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart run request is unreadable or not valid JSON") from error
    expected = (
        {"preflight_request", "audio_path", "output_dir"}
        if capability in {"guitar.hybrid-v2-rule/v1", "drums.v14-expert/v1"}
        else {"preflight_request", "source_midi_path", "song_path", "output_dir", "threshold"}
    )
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WorkerRequestError("chart run request has unsupported fields")
    required_locations = expected - {"song_path", "threshold"}
    if not all(isinstance(raw[key], str) and raw[key] for key in required_locations):
        raise WorkerRequestError("chart run request locations must be non-empty strings")
    if capability == "difficulty.transform/v1":
        if raw["song_path"] is not None and (
            not isinstance(raw["song_path"], str) or not raw["song_path"]
        ):
            raise WorkerRequestError("difficulty transform song_path must be a location or null")
        if (
            not isinstance(raw["threshold"], (int, float))
            or isinstance(raw["threshold"], bool)
            or not 0 < raw["threshold"] < 1
        ):
            raise WorkerRequestError("difficulty transform threshold must be between 0 and 1")
    return raw


def _write_expert_guitar_midi(chart: Any, output_path: Path) -> None:
    """Write only Expert Guitar—lower difficulties need an explicit STRUM policy."""
    import mido  # noqa: PLC0415

    ticks_per_beat = 480
    tempo = mido.bpm2tempo(float(chart.tempo_bpm))
    messages: list[tuple[int, bool, int]] = []
    for note in chart.notes:
        start = round(float(note.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        end = round(
            (float(note.time_ms) + float(note.duration_ms))
            / 1000
            * ticks_per_beat
            * 1_000_000
            / tempo
        )
        messages.extend(
            ((start, True, 96 + int(note.fret)), (max(end, start + 1), False, 96 + int(note.fret)))
        )
    for chord in chart.chords:
        start = round(float(chord.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        end = round(
            (float(chord.time_ms) + float(chord.duration_ms))
            / 1000
            * ticks_per_beat
            * 1_000_000
            / tempo
        )
        for fret in chord.frets:
            messages.extend(
                ((start, True, 96 + int(fret)), (max(end, start + 1), False, 96 + int(fret)))
            )
    messages.sort(key=lambda event: (event[0], event[1]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous = 0
    for tick, is_on, midi_note in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=midi_note,
                velocity=100 if is_on else 0,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _write_five_lane_midi(
    events: Sequence[dict[str, object]], *, instrument: str, difficulty: str, output_path: Path
) -> None:
    """Write one learned five-lane difficulty track with no implicit companion data."""
    import mido  # noqa: PLC0415

    from scripts.prepare_guitar_chart_pairs import (  # noqa: PLC0415
        DIFFICULTY_BASE_NOTES,
        FIVE_LANE_INSTRUMENT_TRACKS,
    )

    if instrument not in FIVE_LANE_INSTRUMENT_TRACKS or difficulty not in DIFFICULTY_BASE_NOTES:
        raise WorkerRequestError(
            "difficulty transform has unsupported instrument or target difficulty"
        )
    tempo, ticks_per_beat = 500_000, 480
    messages: list[tuple[int, bool, int]] = []
    for event in events:
        time_ms, lanes = event["time_ms"], event["lanes"]
        if not isinstance(time_ms, (int, float)) or not isinstance(lanes, list):
            raise WorkerRequestError("difficulty transform produced invalid event data")
        tick = round(float(time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        for lane in lanes:
            if not isinstance(lane, int) or not 0 <= lane < 5:
                raise WorkerRequestError("difficulty transform produced invalid lane data")
            note = DIFFICULTY_BASE_NOTES[difficulty] + lane
            messages.extend(((tick, True, note), (tick + 120, False, note)))
    messages.sort(key=lambda event: (event[0], event[1]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("track_name", name=FIVE_LANE_INSTRUMENT_TRACKS[instrument], time=0)
    )
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous = 0
    for tick, is_on, note in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=100 if is_on else 0,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _write_expert_drums_midi(events: Sequence[Any], output_path: Path) -> None:
    """Write only direct Expert Drums events from the typed V14 profile."""
    import mido  # noqa: PLC0415

    tempo, ticks_per_beat = 500_000, 480
    messages: list[tuple[int, bool, int, int]] = []
    for event in events:
        tick = round(float(event.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        messages.extend(
            ((tick, True, event.midi_note, event.velocity), (tick + 120, False, event.midi_note, 0))
        )
    messages.sort(key=lambda item: (item[0], item[1], item[2]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="PART DRUMS", time=0),
            mido.MetaMessage("set_tempo", tempo=tempo, time=0),
        ]
    )
    midi.tracks.append(track)
    previous = 0
    for tick, is_on, note, velocity in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=velocity,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _run_without_legacy_output(callback: Any) -> Any:
    """Run a legacy inference callable without leaking private paths to stdout."""
    # Some optional inference dependencies keep logging handlers bound to the
    # process file descriptors, bypassing ``redirect_stdout``. Redirect both
    # descriptors for the short, synchronous call so the worker always emits
    # exactly one JSON response after it has completed.
    output_fds = {1, 2}
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, io.UnsupportedOperation):
            output_fds.add(stream.fileno())
    saved_fds = {fd: os.dup(fd) for fd in output_fds}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd in output_fds:
                os.dup2(sink.fileno(), fd)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return callback()
    finally:
        for fd, saved_fd in saved_fds.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)


def run_chart_request(request_path: Path) -> dict[str, object]:
    """Execute a declared, bundle-backed chart profile with no legacy fallbacks."""
    try:
        raw_request = json.loads(request_path.read_text(encoding="utf-8"))
        preflight_request = (
            raw_request.get("preflight_request") if isinstance(raw_request, dict) else None
        )
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart run request is unreadable or not valid JSON") from error
    if not isinstance(preflight_request, str) or not preflight_request:
        raise WorkerRequestError("chart run request requires a preflight request")
    plan = preflight_chart_request(Path(preflight_request))
    request = _read_chart_run_request(request_path, plan["capability"])
    if plan["execution"] != "available":
        raise WorkerRequestError("profile has no worker chart execution handler")
    try:
        preflight_raw = json.loads(Path(request["preflight_request"]).read_text(encoding="utf-8"))
        bundle = load_model_bundle(preflight_raw["model_root"], check_files=True)
        errors = bundle.validate(check_files=True, verify_hashes=True)
        if errors:
            raise BundleValidationError("; ".join(errors))
        output_dir = Path(request["output_dir"])
        if plan["capability"] == "guitar.hybrid-v2-rule/v1":
            audio = Path(request["audio_path"])
            if not audio.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from src.inference.guitar_hybrid_profile import (  # noqa: PLC0415
                load_guitar_hybrid_rule_profile,
            )
            from src.inference.guitar_hybrid_v2 import transcribe_guitar_hybrid  # noqa: PLC0415

            profile = load_guitar_hybrid_rule_profile(bundle, preflight_raw["profile_id"])
            chart = _run_without_legacy_output(
                lambda: transcribe_guitar_hybrid(
                    audio,
                    device=plan["device"],
                    execution_profile=profile,
                    model_bundle=bundle,
                )
            )
            midi_path = output_dir / "notes.mid"
            _write_expert_guitar_midi(chart, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {
                "guitar": {
                    "status": "succeeded",
                    "expert_event_count": len(chart.notes) + len(chart.chords),
                }
            }
            response = {
                "output_name": midi_path.name,
                "expert_event_count": len(chart.notes) + len(chart.chords),
            }
        elif plan["capability"] == "drums.v14-expert/v1":
            audio = Path(request["audio_path"])
            if not audio.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from src.inference.drums_v14_profile import (
                load_drums_v14_expert_profile,  # noqa: PLC0415
            )
            from src.inference.drums_v14_runtime import DrumsV14Runtime  # noqa: PLC0415

            profile = load_drums_v14_expert_profile(bundle, preflight_raw["profile_id"])
            component = bundle.component(profile.component_id)
            if component is None or component.checkpoint is None:
                raise WorkerRequestError("drums V14 component is incomplete")
            events = _run_without_legacy_output(
                lambda: DrumsV14Runtime.from_profile(
                    profile,
                    checkpoint_path=component.checkpoint,
                    model_parameters=profile.model_parameters,
                    device=plan["device"],
                ).transcribe_audio_file(audio)
            )
            midi_path = output_dir / "notes.mid"
            _write_expert_drums_midi(events, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {"drums": {"status": "succeeded", "expert_event_count": len(events)}}
            response = {"output_name": midi_path.name, "expert_event_count": len(events)}
        elif plan["capability"] == "difficulty.transform/v1":
            import torch  # noqa: PLC0415

            from scripts.infer_chart_transform import parse_source_events, predict  # noqa: PLC0415
            from scripts.prepare_guitar_chart_pairs import (  # noqa: PLC0415
                parse_instrument_difficulties,
            )

            component_id = plan["components"][0]["id"]
            component = bundle.component(component_id)
            if component is None or component.checkpoint is None or component.config is None:
                raise WorkerRequestError("difficulty transform component is incomplete")
            config = json.loads(component.config.read_text(encoding="utf-8"))
            instrument = config.get("instrument") if isinstance(config, dict) else None
            target_difficulty = (
                config.get("target_difficulty") if isinstance(config, dict) else None
            )
            if not isinstance(instrument, str) or instrument not in plan["instruments"]:
                raise WorkerRequestError("difficulty transform component instrument is invalid")
            checkpoint = torch.load(component.checkpoint, map_location="cpu", weights_only=True)
            lane_count = checkpoint.get("lane_count") if isinstance(checkpoint, dict) else None
            if not isinstance(lane_count, int) or lane_count != 5:
                raise WorkerRequestError("difficulty transform checkpoint has invalid lane count")
            source_midi = Path(request["source_midi_path"])
            if not source_midi.is_file():
                raise WorkerRequestError("difficulty transform source MIDI is unavailable")
            source_events = parse_source_events(
                parse_instrument_difficulties(source_midi, instrument)["Expert"], lane_count
            )
            song_path = Path(request["song_path"]) if request["song_path"] else None
            if song_path is not None and not song_path.is_file():
                raise WorkerRequestError("difficulty transform song input is unavailable")
            events = _run_without_legacy_output(
                lambda: predict(
                    component.checkpoint,
                    source_events,
                    song_path=song_path,
                    device_name=plan["device"],
                    threshold=float(request["threshold"]),
                )
            )
            events_path = output_dir / "events.json"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.write_text(
                json.dumps(
                    {"instrument": instrument, "difficulty": target_difficulty, "events": events},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            midi_path = output_dir / "notes.mid"
            _write_five_lane_midi(
                events, instrument=instrument, difficulty=target_difficulty, output_path=midi_path
            )
            artifacts = {
                "events": {"name": events_path.name, "sha256": _sha256(events_path)},
                "notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)},
            }
            stages = {"difficulty_transform": {"status": "succeeded", "event_count": len(events)}}
            response = {
                "output_name": midi_path.name,
                "event_count": len(events),
                "difficulty": target_difficulty,
            }
        else:
            raise WorkerRequestError("profile has no worker chart execution handler")
        run_manifest = {
            "schema_version": 1,
            "format": "strum-chart-run/v1",
            "status": "completed",
            "model_id": plan["model_id"],
            "profile_id": plan["profile_id"],
            "capability": plan["capability"],
            "difficulty_policy": plan["difficulty_policy"],
            "components": plan["components"],
            "profile_configuration_sha256": plan["profile_configuration_sha256"],
            "profile_configuration_byte_length": plan["profile_configuration_byte_length"],
            "artifacts": artifacts,
            "stages": stages,
        }
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except WorkerRequestError:
        raise
    except (BundleValidationError, OSError, RuntimeError, ValueError) as error:
        raise WorkerRequestError("profile chart execution failed") from error
    return {
        "status": "completed",
        "profile_id": plan["profile_id"],
        "run_manifest_name": "run.json",
        **response,
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _run_event_stream(request_path: Path, stage: str, callback: Any) -> int:
    """Emit safe NDJSON lifecycle events for an OCTAVE-supervised sync job."""
    try:
        job_id = f"strum-{hashlib.sha256(request_path.read_bytes()).hexdigest()[:16]}"
    except OSError:
        job_id = "strum-unreadable-request"
    _print_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "sequence": 1,
            "stage": stage,
            "progress": 0.0,
            "state": "running",
            "code": "started",
        }
    )
    try:
        result = callback()
    except BundleValidationError:
        code, message = "model_bundle_invalid", "model bundle failed validation"
    except CatalogValidationError:
        code, message = "catalog_invalid", "catalog failed validation"
    except WorkerRequestError:
        code, message = "request_invalid", "worker request is invalid"
    else:
        _print_json(
            {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job_id,
                "sequence": 2,
                "stage": stage,
                "progress": 1.0,
                "state": "succeeded",
                "code": "completed",
                "result": result,
            }
        )
        return 0
    _print_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "sequence": 2,
            "stage": stage,
            "progress": 1.0,
            "state": "failed",
            "code": code,
            "message": message,
        }
    )
    return 2


def _pipeline_by_id(pipeline_id: str) -> PipelineDescriptor:
    for pipeline in PIPELINES:
        if pipeline.id == pipeline_id:
            return pipeline
    raise WorkerRequestError("unknown pipeline_id")


def inspect_catalog(catalog_root: str | Path, pipeline_id: str | None = None) -> dict[str, object]:
    """Validate a catalog and return a path-free capability summary for OCTAVE."""
    if pipeline_id is not None:
        _pipeline_by_id(pipeline_id)
    catalog = load_catalog(catalog_root)
    allowed = sum(record.training_use == "allowed" for record in catalog.records)
    return {
        "status": "ready",
        "catalog_id": catalog.catalog_id,
        "record_count": len(catalog.records),
        "allowed_record_count": allowed,
        "pipeline_id": pipeline_id,
    }


def _read_prepare_request(request_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("request is unreadable or not valid JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "catalog_root",
        "pipeline_id",
        "output",
        "options",
    }:
        raise WorkerRequestError(
            "request must contain catalog_root, pipeline_id, output, and options"
        )
    if not all(
        isinstance(raw[key], str) and raw[key] for key in ("catalog_root", "pipeline_id", "output")
    ):
        raise WorkerRequestError("request locations and pipeline_id must be non-empty strings")
    if not isinstance(raw["options"], dict):
        raise WorkerRequestError("request options must be an object")
    return raw


def _task_view_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_dataset_request(request_path: Path) -> dict[str, object]:
    """Materialize one catalog-only task view from a strict host request.

    Paths are accepted only as worker-local configuration. The response exposes
    the known output basename and stable identity, never original package paths
    or a catalog location.
    """
    request = _read_prepare_request(request_path)
    pipeline_id = request["pipeline_id"]
    descriptor = _pipeline_by_id(pipeline_id)
    if descriptor.preparation_status != "available":
        raise WorkerRequestError("pipeline does not support dataset preparation")
    catalog_root, output, options = (
        request["catalog_root"],
        Path(request["output"]),
        request["options"],
    )
    if pipeline_id == "guitar.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Guitar preparation option")
        manifest = build_guitar_manifest(catalog_root, **options)
        written = write_guitar_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id == "drums.onset-classifier/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Drums preparation option")
        manifest = build_drums_manifest(catalog_root, **options)
        written = write_drums_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = manifest.get("task_view_sha256") or _task_view_digest(manifest)
    elif pipeline_id == "chart_transform.five_lane/v1":
        permitted = {
            "instrument",
            "target_difficulty",
            "split_seed",
            "validation_fraction",
            "dataset_id",
            "overwrite",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported chart-transform preparation option")
        try:
            result = prepare_catalog_chart_pairs(
                catalog_root,
                output,
                CatalogChartPairOptions(
                    **{key: value for key, value in options.items() if key != "overwrite"}
                ),
                overwrite=bool(options.get("overwrite", False)),
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError("invalid chart-transform preparation options") from error
        written = result["manifest_path"]
        record_count = result["record_count"]
        task_view_id = result["task_view_id"]
    else:
        task_kind = next(
            (kind for kind, value in CATALOG_TASK_PIPELINES.items() if value == pipeline_id), None
        )
        if task_kind is None:
            raise WorkerRequestError("pipeline has no catalog task adapter")
        permitted = {
            "audio_role",
            "fallback_audio_role",
            "disable_fallback",
            "required_difficulty",
            "split_ratios",
            "split_seed",
            "preprocessing",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported catalog task preparation option")
        manifest = build_catalog_task_manifest(catalog_root, task_kind, **options)
        written = write_catalog_task_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    return {
        "status": "prepared",
        "pipeline_id": pipeline_id,
        "task_view_id": task_view_id,
        "record_count": record_count,
        "output_name": written.name,
    }


def _read_train_request(request_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("request is unreadable or not valid JSON") from error
    expected = {"pipeline_id", "task_view", "output", "options"}
    permitted = expected | {"catalog_root"}
    if not isinstance(raw, dict) or set(raw) - permitted or not expected <= set(raw):
        raise WorkerRequestError("training request has unsupported fields")
    if not all(
        isinstance(raw[key], str) and raw[key] for key in ("pipeline_id", "task_view", "output")
    ):
        raise WorkerRequestError(
            "training request locations and pipeline_id must be non-empty strings"
        )
    if not isinstance(raw["options"], dict):
        raise WorkerRequestError("training options must be an object")
    if "catalog_root" in raw and (not isinstance(raw["catalog_root"], str) or not raw["catalog_root"]):
        raise WorkerRequestError("training catalog_root must be a non-empty string")
    return raw


def run_training_request(request_path: Path) -> dict[str, object]:
    """Run one explicit, synchronous training job for a worker-supported pipeline.

    OCTAVE owns job scheduling. It can run this command in a background child
    process and persist opaque locations itself; STRUM returns only portable
    model identity, preflight data, and metrics.
    """
    request = _read_train_request(request_path)
    pipeline_id = request["pipeline_id"]
    descriptor = _pipeline_by_id(pipeline_id)
    if descriptor.training_status != "available":
        raise WorkerRequestError("pipeline does not support worker training")
    if pipeline_id == "guitar.onset-fret/v1":
        from src.guitar_worker_training import (  # noqa: PLC0415
            GuitarTrainingError,
            GuitarTrainingOptions,
            run_catalog_guitar_training,
        )

        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Guitar training requires worker-local catalog_root")
        try:
            options = GuitarTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_guitar_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (GuitarTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError("Guitar training request failed validation or execution") from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
        }
    if pipeline_id == "drums.onset-classifier/v1":
        from src.drums_onset_training import (  # noqa: PLC0415
            DrumsTrainingError,
            run_drums_onset_training,
        )

        try:
            return _run_without_legacy_output(
                lambda: run_drums_onset_training(
                    request["task_view"],
                    request["output"],
                    request["options"],
                    catalog_root=request["catalog_root"],
                )
            )
        except DrumsTrainingError as error:
            raise WorkerRequestError("Drums onset training request failed validation") from error
    if pipeline_id != "chart_transform.five_lane/v1":
        raise WorkerRequestError("pipeline has no worker training handler")
    if "catalog_root" in request:
        raise WorkerRequestError("chart-transform training does not accept catalog_root")

    # Importing PyTorch belongs to an actual job, not `strum-worker probe`.
    from scripts.train_chart_transform import (  # noqa: PLC0415
        DatasetValidationError,
        TrainingConfig,
        train,
    )

    options = request["options"]
    permitted = {
        "model_id",
        "seed",
        "validation_fraction",
        "lane_count",
        "alignment_tolerance_ms",
        "hidden_dim",
        "learning_rate",
        "epochs",
        "device",
        "audio_feature_mode",
        "audio_sample_rate",
        "audio_window_ms",
        "audio_max_duration_seconds",
        "strum_revision",
    }
    if set(options) - permitted or not isinstance(options.get("model_id"), str):
        raise WorkerRequestError("invalid chart-transform training options")
    try:
        dataset = json.loads(Path(request["task_view"]).read_text(encoding="utf-8"))
        config = TrainingConfig.from_mapping(
            {
                "dataset_manifest": request["task_view"],
                "output_dir": request["output"],
                "model_id": options["model_id"],
                "source_difficulty": dataset["source_difficulty"],
                "target_difficulty": dataset["target_difficulty"],
                **{key: value for key, value in options.items() if key != "model_id"},
            }
        )
        result = train(config)
        preflight = preflight_bundle(result["bundle_dir"])
    except (KeyError, OSError, TypeError, ValueError, DatasetValidationError) as error:
        raise WorkerRequestError("chart-transform training request failed validation") from error
    return {
        "status": "completed",
        "pipeline_id": pipeline_id,
        "model_id": preflight["model_id"],
        "bundle_name": Path(result["bundle_dir"]).name,
        "manifest_sha256": preflight["manifest_sha256"],
        "components": preflight["components"],
        "validation": result["metrics"]["validation"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Versioned STRUM worker contract")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="describe runtime capabilities").add_argument(
        "--json", action="store_true"
    )
    pipeline = commands.add_parser("pipeline", help="inspect STRUM pipelines")
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    pipeline_commands.add_parser("list", help="list pipeline descriptors").add_argument(
        "--json", action="store_true"
    )
    catalog = commands.add_parser("catalog", help="validate OCTAVE song-source catalogs")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_inspect = catalog_commands.add_parser("inspect", help="inspect one catalog")
    catalog_inspect.add_argument("--catalog-root", type=Path, required=True)
    catalog_inspect.add_argument("--pipeline")
    catalog_inspect.add_argument("--json", action="store_true")
    dataset = commands.add_parser("dataset", help="prepare STRUM task views")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_commands.add_parser("prepare", help="materialize one catalog task view")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--json", action="store_true")
    prepare.add_argument("--json-events", action="store_true")
    training = commands.add_parser("train", help="run worker-managed training")
    training_commands = training.add_subparsers(dest="training_command", required=True)
    training_run = training_commands.add_parser("run", help="run one synchronous training job")
    training_run.add_argument("--request", type=Path, required=True)
    training_run.add_argument("--json", action="store_true")
    training_run.add_argument("--json-events", action="store_true")
    training_start = training_commands.add_parser(
        "start", help="start one OCTAVE-supervised training job"
    )
    training_start.add_argument("--request", type=Path, required=True)
    training_start.add_argument("--json-events", action="store_true")
    model = commands.add_parser("model", help="inspect model bundles")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    preflight = model_commands.add_parser("preflight", help="validate a deployable model bundle")
    preflight.add_argument("--model-root", type=Path, required=True)
    preflight.add_argument("--require-component", action="append", default=[])
    preflight.add_argument("--json", action="store_true")
    checkpoint = commands.add_parser("checkpoint", help="inspect checkpoint bundle metadata")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    inspect = checkpoint_commands.add_parser("inspect", help="inspect a checkpoint bundle")
    inspect.add_argument("--model-root", type=Path, required=True)
    inspect.add_argument("--json", action="store_true")
    inference = commands.add_parser("inference", help="validate deployable inference profiles")
    inference_commands = inference.add_subparsers(dest="inference_command", required=True)
    profile = inference_commands.add_parser("profile", help="inspect one inference profile")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    validate_profile = profile_commands.add_parser(
        "validate", help="validate one inference profile"
    )
    validate_profile.add_argument("--model-root", type=Path, required=True)
    validate_profile.add_argument("--profile", required=True)
    validate_profile.add_argument("--difficulty-policy", required=True)
    validate_profile.add_argument("--json", action="store_true")
    chart = commands.add_parser("chart", help="preflight typed auto-chart requests")
    chart_commands = chart.add_subparsers(dest="chart_command", required=True)
    chart_preflight = chart_commands.add_parser(
        "preflight", help="validate a chart profile request"
    )
    chart_preflight.add_argument("--request", type=Path, required=True)
    chart_preflight.add_argument("--json", action="store_true")
    chart_run = chart_commands.add_parser("run", help="execute an explicit chart profile")
    chart_run.add_argument("--request", type=Path, required=True)
    chart_run.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "probe":
            _print_json(_runtime_payload())
            return 0
        if args.command == "pipeline" and args.pipeline_command == "list":
            _print_json(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "pipelines": [pipeline.as_json() for pipeline in PIPELINES],
                }
            )
            return 0
        if args.command == "catalog" and args.catalog_command == "inspect":
            _print_json(inspect_catalog(args.catalog_root, args.pipeline))
            return 0
        if args.command == "dataset" and args.dataset_command == "prepare":
            if args.json_events:
                return _run_event_stream(
                    args.request, "dataset_prepare", lambda: prepare_dataset_request(args.request)
                )
            _print_json(prepare_dataset_request(args.request))
            return 0
        if args.command == "train" and args.training_command in {"run", "start"}:
            if args.json_events:
                return _run_event_stream(
                    args.request, "training", lambda: run_training_request(args.request)
                )
            _print_json(run_training_request(args.request))
            return 0
        if args.command == "model" and args.model_command == "preflight":
            _print_json(
                preflight_bundle(args.model_root, required_components=args.require_component)
            )
            return 0
        if args.command == "checkpoint" and args.checkpoint_command == "inspect":
            bundle = load_model_bundle(args.model_root, check_files=False)
            _print_json(
                {
                    "model_id": bundle.model_id,
                    "manifest_sha256": _manifest_sha256(bundle),
                    "components": sorted(bundle.components),
                    "compatibility": bundle.compatibility,
                }
            )
            return 0
        if (
            args.command == "inference"
            and args.inference_command == "profile"
            and args.profile_command == "validate"
        ):
            _print_json(
                validate_inference_profile(
                    args.model_root,
                    profile_id=args.profile,
                    difficulty_policy=args.difficulty_policy,
                )
            )
            return 0
        if args.command == "chart" and args.chart_command == "preflight":
            _print_json(preflight_chart_request(args.request))
            return 0
        if args.command == "chart" and args.chart_command == "run":
            _print_json(run_chart_request(args.request))
            return 0
    except BundleValidationError:
        _print_json(
            {
                "status": "invalid",
                "code": "model_bundle_invalid",
                "message": "model bundle failed validation",
            }
        )
        return 2
    except CatalogValidationError:
        _print_json(
            {"status": "invalid", "code": "catalog_invalid", "message": "catalog failed validation"}
        )
        return 2
    except WorkerRequestError:
        _print_json(
            {"status": "invalid", "code": "request_invalid", "message": "worker request is invalid"}
        )
        return 2
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
