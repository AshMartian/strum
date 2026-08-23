"""Decode approved Pro MIDI tracks into immutable STRUM training targets.

The generic catalog task manifest intentionally knows only which managed MIDI
asset is an allowed label source.  That is sufficient for a number of task
families, but not for Pro instruments: an audio model must never infer that a
five-lane note is a string/fret target.  This module is the narrow bridge from
the exact ``PART REAL_*`` tracks selected by the catalog contract to portable,
auditable target events.

It is deliberately a dataset-preparation boundary, not an auto-chart runtime
or a training architecture.  It decodes only the expert target encodings that
are established by the Rock Band/Clone Hero Pro MIDI convention, preserves the
standard versus ``_22`` source-track distinction, and re-decodes every target
when a future trainer resolves the view.  No package paths or catalog roots
are written to the result.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mido

from src.catalog_task_manifest import (
    MANIFEST_FORMAT,
    build_catalog_task_manifest,
    resolve_catalog_task_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError

PRO_TARGET_MANIFEST_FORMAT = "strum-pro-target-task-manifest/v1"
PRO_TARGET_MANIFEST_VERSION = 1
PRO_TARGET_DECODER_ID = "strum-pro-midi-target-decoder/v1"
PRO_TASK_KINDS = frozenset({"pro_guitar", "pro_bass", "pro_keys"})

_PRO_GUITAR_TECHNIQUES = {
    0: "normal",
    1: "arpeggio_form",
    2: "bent",
    3: "muted",
    4: "tapped",
    5: "harmonic",
    6: "pinch_harmonic",
}
_PRO_KEYS_RANGE_ANCHORS = {
    0: "C",
    2: "D",
    4: "E",
    5: "F",
    7: "G",
    9: "A",
}
_PRO_STRING_EXPERT_BASE = 96
_PRO_STRING_COUNT = 6
_PRO_KEYS_MIN_PITCH = 48
_PRO_KEYS_MAX_PITCH = 72


class ProTargetDecodeError(ValueError):
    """Raised when an approved Pro source cannot supply unambiguous targets."""


def _is_note_on(message: mido.Message) -> bool:
    return message.type == "note_on" and message.velocity > 0


def _is_note_off(message: mido.Message) -> bool:
    return message.type == "note_off" or (message.type == "note_on" and message.velocity == 0)


def _named_track(midi: mido.MidiFile, track_name: str) -> mido.MidiTrack:
    matches = [track for track in midi.tracks if track.name.upper() == track_name]
    if len(matches) != 1:
        raise ProTargetDecodeError("selected Pro MIDI track is missing or duplicated")
    return matches[0]


def _load_midi(path: str | Path) -> mido.MidiFile:
    try:
        return mido.MidiFile(path)
    except (EOFError, OSError, ValueError) as error:
        raise ProTargetDecodeError("Pro MIDI asset is unreadable") from error


def _decode_pro_string_track(track: mido.MidiTrack, track_name: str) -> dict[str, object]:
    """Decode Expert Pro Guitar/Bass gems without collapsing track variants."""
    variant = "22_fret" if track_name.endswith("_22") else "standard"
    max_fret = 22 if variant == "22_fret" else 17
    pending: dict[tuple[int, int], tuple[int, int, str]] = {}
    events: list[dict[str, object]] = []
    tick = 0
    for message in track:
        tick += message.time
        if message.type not in {"note_on", "note_off"}:
            continue
        if (
            not _PRO_STRING_EXPERT_BASE
            <= message.note
            < _PRO_STRING_EXPERT_BASE + _PRO_STRING_COUNT
        ):
            continue
        key = (message.note, message.channel)
        if _is_note_on(message):
            technique = _PRO_GUITAR_TECHNIQUES.get(message.channel)
            fret = message.velocity - 100
            if technique is None or not 0 <= fret <= max_fret:
                raise ProTargetDecodeError(
                    "Pro string target uses an unsupported technique or fret"
                )
            if key in pending:
                raise ProTargetDecodeError("Pro string target has overlapping note starts")
            pending[key] = (tick, fret, technique)
        elif _is_note_off(message):
            start = pending.pop(key, None)
            if start is None:
                raise ProTargetDecodeError("Pro string target has an unmatched note end")
            onset, fret, technique = start
            events.append(
                {
                    "tick": onset,
                    "duration_ticks": tick - onset,
                    "string": message.note - _PRO_STRING_EXPERT_BASE,
                    "fret": fret,
                    "technique": technique,
                }
            )
    if pending:
        raise ProTargetDecodeError("Pro string target has an unmatched note start")
    if not events:
        raise ProTargetDecodeError("Pro string target has no Expert playable events")
    events.sort(key=lambda event: (event["tick"], event["string"], event["fret"]))
    return {
        "track_name": track_name,
        "track_variant": variant,
        "event_schema": "pro-string-fret-events/v1",
        "events": events,
    }


def _decode_pro_keys_track(track: mido.MidiTrack, track_name: str) -> dict[str, object]:
    """Decode Expert Pro Keys pitches and lane-range state without five-lane mapping."""
    pending: dict[tuple[int, int], int] = {}
    events: list[dict[str, object]] = []
    range_shifts: list[dict[str, object]] = []
    tick = 0
    for message in track:
        tick += message.time
        if message.type not in {"note_on", "note_off"}:
            continue
        if message.note in _PRO_KEYS_RANGE_ANCHORS and _is_note_on(message):
            range_shifts.append({"tick": tick, "anchor": _PRO_KEYS_RANGE_ANCHORS[message.note]})
            continue
        if not _PRO_KEYS_MIN_PITCH <= message.note <= _PRO_KEYS_MAX_PITCH:
            continue
        key = (message.note, message.channel)
        if _is_note_on(message):
            if key in pending:
                raise ProTargetDecodeError("Pro Keys target has overlapping note starts")
            pending[key] = tick
        elif _is_note_off(message):
            onset = pending.pop(key, None)
            if onset is None:
                raise ProTargetDecodeError("Pro Keys target has an unmatched note end")
            events.append(
                {
                    "tick": onset,
                    "duration_ticks": tick - onset,
                    "pitch": message.note,
                    # Channel is retained as authored metadata; it is never
                    # inferred as a left/right-hand label by STRUM.
                    "channel": message.channel,
                }
            )
    if pending:
        raise ProTargetDecodeError("Pro Keys target has an unmatched note start")
    if not events:
        raise ProTargetDecodeError("Pro Keys target has no Expert playable events")
    events.sort(key=lambda event: (event["tick"], event["pitch"], event["channel"]))
    range_shifts.sort(key=lambda event: (event["tick"], event["anchor"]))
    return {
        "track_name": track_name,
        "event_schema": "pro-keys-pitch-events/v1",
        "events": events,
        "range_shifts": range_shifts,
    }


def decode_pro_midi_targets(
    midi_path: str | Path, task_kind: str, label_tracks: list[str]
) -> list[dict[str, object]]:
    """Decode exactly the catalog-selected Pro label tracks from one MIDI asset."""
    if task_kind not in PRO_TASK_KINDS:
        raise ProTargetDecodeError("unsupported Pro target task")
    if not label_tracks or not all(isinstance(name, str) for name in label_tracks):
        raise ProTargetDecodeError("Pro target label tracks are invalid")
    midi = _load_midi(midi_path)
    decoded: list[dict[str, object]] = []
    for track_name in label_tracks:
        track = _named_track(midi, track_name)
        decoded.append(
            _decode_pro_keys_track(track, track_name)
            if task_kind == "pro_keys"
            else _decode_pro_string_track(track, track_name)
        )
    return decoded


def _target_encoding(task_kind: str) -> dict[str, object]:
    if task_kind == "pro_keys":
        return {
            "id": PRO_TARGET_DECODER_ID,
            "event_schema": "pro-keys-pitch-events/v1",
            "target_fields": ["tick", "duration_ticks", "pitch", "channel"],
            "range_shift_schema": "pro-keys-range-shifts/v1",
            "source_tracks": ["PART REAL_KEYS_X"],
        }
    return {
        "id": PRO_TARGET_DECODER_ID,
        "event_schema": "pro-string-fret-events/v1",
        "target_fields": ["tick", "duration_ticks", "string", "fret", "technique"],
        "track_variant_field": "track_variant",
        "source_tracks": (
            ["PART REAL_GUITAR", "PART REAL_GUITAR_22"]
            if task_kind == "pro_guitar"
            else ["PART REAL_BASS", "PART REAL_BASS_22"]
        ),
    }


def _source_target_entry(
    source: Mapping[str, object], decoded_tracks: list[dict[str, object]]
) -> dict[str, object]:
    return {
        "source_id": source["source_id"],
        "split": source["split"],
        "label_tracks": copy.deepcopy(source["label_tracks"]),
        "targets": decoded_tracks,
    }


def build_catalog_pro_target_manifest(
    catalog_root: str | Path,
    task_kind: str,
    **options: object,
) -> dict[str, object]:
    """Build a path-free catalog view with decoded Pro event targets.

    Catalog coverage is intentionally only a lightweight declaration.  A
    source enters this stronger view only after its declared exact REAL_* track
    has been decoded into its required Expert event language.
    """
    if task_kind not in PRO_TASK_KINDS:
        raise CatalogValidationError("unsupported Pro target task")
    task_view = build_catalog_task_manifest(catalog_root, task_kind, **options)
    try:
        resolved = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise CatalogValidationError("Pro catalog task view cannot be resolved") from error
    decoded_songs: list[dict[str, object]] = []
    exclusions: Counter[str] = Counter()
    for source, resolved_source in zip(task_view["songs"], resolved, strict=True):
        assert isinstance(source, dict) and isinstance(resolved_source, dict)
        try:
            targets = decode_pro_midi_targets(
                str(resolved_source["midi_path"]), task_kind, list(source["label_tracks"])
            )
        except ProTargetDecodeError as error:
            exclusions[str(error)] += 1
            continue
        decoded_songs.append(_source_target_entry(source, targets))
    counts = Counter(song["split"] for song in decoded_songs)
    return {
        "schema_version": PRO_TARGET_MANIFEST_VERSION,
        "format": PRO_TARGET_MANIFEST_FORMAT,
        "task_view": task_view,
        "target_encoding": _target_encoding(task_kind),
        "songs": decoded_songs,
        "summary": {
            "record_count": len(decoded_songs),
            "by_split": dict(sorted(counts.items())),
            "coverage_record_count": len(task_view["songs"]),
            "exclusion_reason_counts": dict(sorted(exclusions.items())),
        },
    }


def write_catalog_pro_target_manifest(output: str | Path, manifest: Mapping[str, object]) -> Path:
    """Write a portable Pro task view without exposing the catalog location."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def resolve_catalog_pro_target_manifest_songs(
    manifest: Mapping[str, Any], catalog_root: str | Path
) -> list[dict[str, object]]:
    """Revalidate catalog lineage and exactly re-derive Pro targets for a trainer."""
    if (
        manifest.get("schema_version") != PRO_TARGET_MANIFEST_VERSION
        or manifest.get("format") != PRO_TARGET_MANIFEST_FORMAT
    ):
        raise CatalogValidationError(f"manifest must use {PRO_TARGET_MANIFEST_FORMAT}")
    task_view = manifest.get("task_view")
    target_encoding = manifest.get("target_encoding")
    stored_songs = manifest.get("songs")
    if not isinstance(task_view, dict) or task_view.get("format") != MANIFEST_FORMAT:
        raise CatalogValidationError("Pro target manifest task view is invalid")
    task = task_view.get("task")
    task_kind = task.get("kind") if isinstance(task, dict) else None
    if task_kind not in PRO_TASK_KINDS or target_encoding != _target_encoding(task_kind):
        raise CatalogValidationError("Pro target manifest encoding is invalid")
    if not isinstance(stored_songs, list):
        raise CatalogValidationError("Pro target manifest songs are invalid")
    resolved = resolve_catalog_task_manifest_songs(task_view, catalog_root)
    source_by_id = {
        str(source["source_id"]): source
        for source in task_view.get("songs", [])
        if isinstance(source, dict) and isinstance(source.get("source_id"), str)
    }
    resolved_by_id = {
        str(source["source_id"]): source
        for source in resolved
        if isinstance(source.get("source_id"), str)
    }
    expected: list[dict[str, object]] = []
    for source_id in sorted(source_by_id):
        source = source_by_id[source_id]
        resolved_source = resolved_by_id.get(source_id)
        if resolved_source is None:
            raise CatalogValidationError("Pro target manifest source resolution is invalid")
        try:
            targets = decode_pro_midi_targets(
                str(resolved_source["midi_path"]), task_kind, list(source["label_tracks"])
            )
        except ProTargetDecodeError:
            # A source excluded at preparation may remain excluded; a source
            # that had targets must never silently become malformed at train.
            continue
        expected.append(_source_target_entry(source, targets))
    if stored_songs != expected:
        raise CatalogValidationError(
            "Pro target manifest targets do not match the approved catalog"
        )
    summary = manifest.get("summary")
    counts = Counter(song["split"] for song in expected)
    if (
        not isinstance(summary, dict)
        or summary.get("record_count") != len(expected)
        or summary.get("by_split") != dict(sorted(counts.items()))
    ):
        raise CatalogValidationError("Pro target manifest summary is invalid")
    output: list[dict[str, object]] = []
    for source in expected:
        resolved_source = resolved_by_id[str(source["source_id"])]
        output.append({**resolved_source, "targets": source["targets"]})
    return output
