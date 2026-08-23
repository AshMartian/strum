"""STRUM-owned, fail-closed admission evidence for lead-Vocal task views.

The four lead-Vocal workers intentionally train separate components.  A host
must not join their caller-reported song counts into evidence for a future
chart profile.  This module opens the private task views and catalog only
inside STRUM, revalidates each view against the current catalog, and counts
labels from the one exact ``PART VOCALS`` MIDI track.

It is an *admission/preparation* boundary only.  It loads no checkpoints,
does not evaluate a model, and cannot create a Vocal profile or chart.  Its
returned report is path-free: it contains opaque task-view/content identities
and aggregate coverage, never catalog asset locations or source IDs.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mido

from src.catalog_task_manifest import MANIFEST_FORMAT, resolve_catalog_task_manifest_songs
from src.song_source_catalog import CatalogValidationError
from src.vocal_lead_profile_contract import (
    evaluate_vocal_lead_data_coverage,
    vocal_lead_candidate_data_gate_identity,
)

ADMISSION_FORMAT = "strum-vocal-lead-catalog-task-admission/v1"
RESOLVER_ID = "strum-owned-lead-catalog-task-admission-resolver/v1"
SOURCE_PARTITION = "source-id-disjoint-train-val-test/v1"
SPLITS = ("train", "val", "test")
LABELS = ("pitched_note_events", "phrase_boundaries", "lyric_events", "talky_spans")
TASKS: dict[str, str] = {
    "vocals_activity": "vocals.note-activity/v1",
    "vocals_phrase_boundaries": "vocals.phrase-boundaries/v1",
    "vocals_lyric_alignment": "vocals.lyric-alignment/v1",
    "vocals_talky_activity": "vocals.talky-activity/v1",
}
_VOCAL_MIN_MIDI = 36
_VOCAL_MAX_MIDI = 84


class VocalLeadCatalogAdmissionError(ValueError):
    """Raised when STRUM cannot independently establish lead-task evidence."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _source_set_sha256(source_ids: set[str]) -> str:
    return _sha256_bytes(
        json.dumps(sorted(source_ids), separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    )


def _read_task_view(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        content = Path(path).read_bytes()
        raw = json.loads(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VocalLeadCatalogAdmissionError(
            "lead Vocal task view is unreadable or invalid"
        ) from error
    if not isinstance(raw, dict):
        raise VocalLeadCatalogAdmissionError("lead Vocal task view is invalid")
    return raw, _sha256_bytes(content)


def _require_task_identity(raw: Mapping[str, object], task_kind: str) -> None:
    task = raw.get("task")
    if (
        raw.get("format") != MANIFEST_FORMAT
        or not isinstance(task, Mapping)
        or task.get("kind") != task_kind
        or task.get("pipeline_id") != TASKS[task_kind]
        or task.get("instrument") != "vocals"
    ):
        raise VocalLeadCatalogAdmissionError(
            "lead Vocal task view has an unsupported task identity"
        )


def _resolved_source_map(
    raw: Mapping[str, object], catalog_root: str | Path, task_kind: str
) -> dict[str, dict[str, object]]:
    _require_task_identity(raw, task_kind)
    try:
        resolved = resolve_catalog_task_manifest_songs(raw, catalog_root)
    except CatalogValidationError as error:
        raise VocalLeadCatalogAdmissionError(
            "lead Vocal task view cannot be revalidated against the catalog"
        ) from error
    by_source: dict[str, dict[str, object]] = {}
    for song in resolved:
        source_id, split, label_tracks = (
            song.get("source_id"),
            song.get("split"),
            song.get("label_tracks"),
        )
        if (
            not isinstance(source_id, str)
            or split not in SPLITS
            or label_tracks != ["PART VOCALS"]
            or source_id in by_source
        ):
            raise VocalLeadCatalogAdmissionError(
                "lead Vocal task view does not resolve one exact PART VOCALS source per song"
            )
        by_source[source_id] = song
    return by_source


def _require_same_source_partition(
    views: Mapping[str, Mapping[str, dict[str, object]]],
) -> dict[str, dict[str, object]]:
    activity = views["vocals_activity"]
    expected = {source_id: song.get("split") for source_id, song in activity.items()}
    for task_kind, sources in views.items():
        observed = {source_id: song.get("split") for source_id, song in sources.items()}
        if observed != expected:
            raise VocalLeadCatalogAdmissionError(
                f"lead Vocal source partition differs across {task_kind} task evidence"
            )
    return dict(activity)


def _require_shared_catalog_lineage(
    task_views: Mapping[str, Mapping[str, object]],
) -> tuple[str, str]:
    catalog_id: str | None = None
    control: str | None = None
    for task_kind, raw in task_views.items():
        lineage = raw.get("lineage")
        if not isinstance(lineage, Mapping):
            raise VocalLeadCatalogAdmissionError("lead Vocal task view lineage is invalid")
        current_id, current_control = (
            lineage.get("catalog_id"),
            lineage.get("catalog_control_sha256"),
        )
        if not isinstance(current_id, str) or not isinstance(current_control, str):
            raise VocalLeadCatalogAdmissionError("lead Vocal task view lineage is invalid")
        if catalog_id is None:
            catalog_id, control = current_id, current_control
        elif (current_id, current_control) != (catalog_id, control):
            raise VocalLeadCatalogAdmissionError(
                f"lead Vocal task view catalog lineage differs for {task_kind}"
            )
    assert catalog_id is not None and control is not None  # TASKS is static and non-empty.
    return catalog_id, control


def _count_exact_part_vocals_labels(midi_path: Path) -> dict[str, int]:
    """Count completed target events from exactly one exact lead-Vocal track."""
    try:
        midi = mido.MidiFile(midi_path)
    except (OSError, ValueError, EOFError) as error:
        raise VocalLeadCatalogAdmissionError("lead Vocal MIDI target is unreadable") from error
    tracks = [track for track in midi.tracks if track.name == "PART VOCALS"]
    if len(tracks) != 1:
        raise VocalLeadCatalogAdmissionError(
            "lead Vocal MIDI must contain exactly one PART VOCALS track"
        )

    pitched_active: dict[int, list[int]] = {}
    talky_active: list[int] = []
    phrase_active: list[int] = []
    counts = Counter[str]()
    tick = 0
    for message in tracks[0]:
        tick += message.time
        if message.type in {"lyrics", "text"} and getattr(message, "text", "").strip():
            counts["lyric_events"] += 1
            continue
        if message.type == "note_on" and message.velocity > 0:
            if _VOCAL_MIN_MIDI <= message.note <= _VOCAL_MAX_MIDI:
                pitched_active.setdefault(message.note, []).append(tick)
            elif message.note == 96:
                talky_active.append(tick)
            elif message.note == 105:
                phrase_active.append(tick)
                counts["_phrase_starts"] += 1
            elif message.note == 106:
                counts["_phrase_ends"] += 1
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        if message.type == "note_on" and message.velocity != 0:
            continue
        if _VOCAL_MIN_MIDI <= message.note <= _VOCAL_MAX_MIDI:
            starts = pitched_active.get(message.note)
            if starts and tick > starts[0]:
                starts.pop(0)
                counts["pitched_note_events"] += 1
        elif message.note == 96:
            if talky_active and tick > talky_active[0]:
                talky_active.pop(0)
                counts["talky_spans"] += 1
        elif message.note == 105 and phrase_active:
            start = phrase_active.pop(0)
            if tick > start:
                counts["_phrase_ends"] += 1

    # Phrase supervision requires both sides.  Counting only starts (or only
    # ends) would let a malformed source satisfy a superficially plural
    # ``phrase_boundaries`` count.
    counts["phrase_boundaries"] = min(counts["_phrase_starts"], counts["_phrase_ends"]) * 2
    return {label: counts[label] for label in LABELS}


def _coverage_for_sources(
    sources: Mapping[str, dict[str, object]],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    source_counts = dict.fromkeys(SPLITS, 0)
    label_counts = {split: dict.fromkeys(LABELS, 0) for split in SPLITS}
    for song in sources.values():
        split, midi_path = song.get("split"), song.get("midi_path")
        if split not in SPLITS or not isinstance(midi_path, str):  # resolver invariant guard
            raise VocalLeadCatalogAdmissionError("lead Vocal resolved source is invalid")
        source_counts[split] += 1
        counted = _count_exact_part_vocals_labels(Path(midi_path))
        for label in LABELS:
            label_counts[split][label] += counted[label]
    return source_counts, label_counts


def resolve_vocal_lead_catalog_admission(
    *, catalog_root: str | Path, task_view_paths: Mapping[str, str | Path]
) -> dict[str, object]:
    """Recompute catalog-owned lead-Vocal split and label admission evidence.

    ``catalog_root`` and task-view paths are deliberately private inputs.  A
    valid return value is still not a model evaluation or profile permission;
    callers must preserve the resulting non-executable boundary.
    """
    if set(task_view_paths) != set(TASKS) or not all(
        isinstance(value, (str, Path)) for value in task_view_paths.values()
    ):
        raise VocalLeadCatalogAdmissionError(
            "lead Vocal admission requires all four private task views"
        )

    raw_views: dict[str, dict[str, Any]] = {}
    view_hashes: dict[str, str] = {}
    resolved_views: dict[str, dict[str, dict[str, object]]] = {}
    for task_kind in TASKS:
        raw, digest = _read_task_view(task_view_paths[task_kind])
        raw_views[task_kind], view_hashes[task_kind] = raw, digest
        resolved_views[task_kind] = _resolved_source_map(raw, catalog_root, task_kind)

    catalog_id, catalog_control = _require_shared_catalog_lineage(raw_views)
    sources = _require_same_source_partition(resolved_views)
    source_counts, label_counts = _coverage_for_sources(sources)
    split_sources = {
        split: {source_id for source_id, song in sources.items() if song.get("split") == split}
        for split in SPLITS
    }
    # This is redundant with a one-song/one-split map, but it documents and
    # asserts the invariant before publishing any aggregate identity.
    if any(
        split_sources[first] & split_sources[second]
        for first in SPLITS
        for second in SPLITS
        if first < second
    ):
        raise VocalLeadCatalogAdmissionError("lead Vocal source partitions overlap")

    coverage = {"source_counts": source_counts, "label_counts": label_counts}
    outcomes = evaluate_vocal_lead_data_coverage(coverage)
    missing_labels = {
        split: [label for label in LABELS if label_counts[split][label] == 0] for split in SPLITS
    }
    all_labels_present = not any(missing_labels.values())
    admitted = bool(outcomes["passed"]) and all_labels_present
    return {
        "schema_version": 1,
        "format": ADMISSION_FORMAT,
        "resolver": RESOLVER_ID,
        "status": "admitted_for_future_lead_evaluation_only" if admitted else "not_admitted",
        "catalog": {"catalog_id": catalog_id, "catalog_control_sha256": catalog_control},
        "task_views": {"format": MANIFEST_FORMAT, "sha256": view_hashes},
        "source_partition": {
            "format": SOURCE_PARTITION,
            "split_source_ids_sha256": {
                split: _source_set_sha256(split_sources[split]) for split in SPLITS
            },
        },
        "data_gate": vocal_lead_candidate_data_gate_identity(),
        "data_coverage": coverage,
        "data_gate_outcomes": outcomes,
        "label_availability": {
            "required": list(LABELS),
            "missing": missing_labels,
            "passed": all_labels_present,
        },
        "admission": {
            "passed": admitted,
            "scope": "lead-catalog-data-only/v1",
            "next_stage": "strum-recomputed-lead-held-out-evaluator/v1",
            "profile_packaging": "forbidden-until-full-vocal-profile-contract/v1",
            "chart_execution": "not_available",
        },
    }
