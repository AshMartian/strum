"""Build STRUM task views from OCTAVE-managed song-source catalogs.

This is deliberately the single catalog adapter for all chart/audio training
families.  It never parses imported packages and manifests contain only
catalog-relative content-addressed references.  Runtime resolution reloads and
validates the catalog before returning ephemeral local paths to a trainer.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.preprocessing.parsers.guitar_parser import GuitarParser
from src.song_source_catalog import (
    AUDIO_ROLES,
    CATALOG_FILENAME,
    CATALOG_FORMAT,
    CatalogAsset,
    CatalogValidationError,
    SongSourceCatalog,
    load_catalog,
    select_training_sources,
)

MANIFEST_FORMAT = "strum-catalog-task-manifest/v1"
MANIFEST_VERSION = 1
SPLIT_ALGORITHM = "sha256-source-id-mod-100/v1"
DEFAULT_SPLIT_RATIOS = (80, 10, 10)

# The catalog's instrument coverage is the ground-truth label source.  These
# descriptors only define task eligibility; task-specific preprocessing derives
# windows, MIDI targets, and feature labels in STRUM.
PIPELINE_IDS = {
    "bass": "strum.instrument-chart/bass/v1",
    # This is deliberately separate from the generic Bass chart task.  The
    # latter remains useful for future training families, whereas this task
    # view is the exact, five-lane Expert label contract consumed by the
    # catalog-backed onset/fret worker.
    "bass_onset_fret": "bass.onset-fret/v1",
    # Like Bass, this is a deliberately narrow task view for the established
    # five-lane onset/fret research implementation.  It is not interchangeable
    # with the generic Keys chart task, which remains the label contract for
    # future keys-specific training architectures.
    "keys_onset_fret": "keys.onset-fret/v1",
    # This narrow vocal-note view feeds only the bounded activity/pitch
    # experiment; it is not a complete phrase/lyric chart profile.
    "vocals_activity": "vocals.note-activity/v1",
    "keys": "strum.instrument-chart/keys/v1",
    "vocals": "strum.instrument-chart/vocals/v1",
    "pro_guitar": "strum.instrument-chart/pro-guitar/v1",
    "pro_bass": "strum.instrument-chart/pro-bass/v1",
    "pro_keys": "strum.instrument-chart/pro-keys/v1",
    "fret_mapper_guitar": "strum.fret-mapper/guitar/v1",
    "fret_mapper_bass": "strum.fret-mapper/bass/v1",
    "section_guitar": "strum.section-classifier/guitar/v1",
    "section_bass": "strum.section-classifier/bass/v1",
}

DEFAULT_AUDIO_ROLES = {
    "bass": ("bass", "mix"),
    "bass_onset_fret": ("bass", "mix"),
    "keys_onset_fret": ("keys", "mix"),
    "vocals_activity": ("vocals", "mix"),
    "keys": ("keys", "mix"),
    "vocals": ("vocals", "mix"),
    "pro_guitar": ("guitar", "mix"),
    "pro_bass": ("bass", "mix"),
    "pro_keys": ("keys", "mix"),
    "fret_mapper_guitar": ("guitar", "mix"),
    "fret_mapper_bass": ("bass", "mix"),
    "section_guitar": ("guitar", "mix"),
    "section_bass": ("bass", "mix"),
}

TASK_INSTRUMENTS = {
    "bass": "bass",
    "bass_onset_fret": "bass",
    "keys_onset_fret": "keys",
    "vocals_activity": "vocals",
    "keys": "keys",
    "vocals": "vocals",
    "pro_guitar": "pro_guitar",
    "pro_bass": "pro_bass",
    "pro_keys": "pro_keys",
    "fret_mapper_guitar": "guitar",
    "fret_mapper_bass": "bass",
    "section_guitar": "guitar",
    "section_bass": "bass",
}

# A catalog only establishes that an approved MIDI asset contains an
# instrument.  A future trainer must also know which event language it is
# allowed to derive from that asset.  Keep that declaration in the immutable
# task view, rather than teaching every future trainer implicit track-name
# conventions.  These schemas describe label *sources*, not model outputs;
# they deliberately do not claim that a trainer/profile exists yet.
TASK_LABEL_SCHEMAS: dict[str, dict[str, object]] = {
    "bass": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART BASS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "bass_onset_fret": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART BASS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "keys_onset_fret": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART KEYS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "vocals_activity": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        "track_prefixes": ["PART VOCALS"],
        "difficulty_encoding": "vocal-pitch-phrase-events/v1",
    },
    "keys": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART KEYS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "vocals": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        "track_prefixes": ["PART VOCALS"],
        "difficulty_encoding": "vocal-pitch-phrase-events/v1",
    },
    "pro_guitar": {
        "id": "pro-string-fret-midi/v1",
        # Do not use a prefix match for Pro tracks.  A catalog may describe
        # additional authoring/preview tracks with a similar name, but only
        # these two track identities carry the supported Pro Guitar event
        # language.  ``PART REAL_GUITAR_22`` is deliberately preserved as a
        # distinct source variant for a future target encoder; it must not be
        # silently collapsed into the standard track.
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
        # Catalog task views currently require Expert labels.  Pro Keys has a
        # separate track per difficulty, so accepting the lower-difficulty
        # tracks here would quietly turn the label source into a mixed-task
        # dataset.  Lower difficulties belong to the learned STRUM transform
        # after an Expert Pro Keys path exists.
        "track_names": ["PART REAL_KEYS_X"],
        "difficulty_encoding": "pro-keys-track-suffix/v1",
    },
    "fret_mapper_guitar": {
        "id": "five-lane-fret-mapper-midi/v1",
        "track_prefixes": ["PART GUITAR"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "fret_mapper_bass": {
        "id": "five-lane-fret-mapper-midi/v1",
        "track_prefixes": ["PART BASS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "section_guitar": {
        "id": "midi-section-events/v1",
        "track_prefixes": ["PART GUITAR"],
        "difficulty_encoding": "not-applicable",
    },
    "section_bass": {
        "id": "midi-section-events/v1",
        "track_prefixes": ["PART BASS"],
        "difficulty_encoding": "not-applicable",
    },
}


def deterministic_split(source_id: str, ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS) -> str:
    """Assign a stable split based solely on a catalog source ID."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) != 100:
        raise CatalogValidationError("split ratios must be three non-negative values totaling 100")
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def available_task_kinds() -> tuple[str, ...]:
    return tuple(sorted(PIPELINE_IDS))


def _has_section_label_source(record: object, instrument: str) -> bool:
    """Return whether an approved MIDI asset is usable by the Section labeler.

    Catalog instrument coverage is intentionally lightweight: it establishes
    that a source declared an instrument, not that every third-party MIDI byte
    sequence can be decoded by STRUM's derived-label parser.  Section task
    views must make that stronger promise because their trainer derives labels
    directly from five-lane chart events.
    """
    notes_midi = getattr(record, "notes_midi", None)
    path = getattr(notes_midi, "path", None)
    if not isinstance(path, Path):
        return False
    try:
        chart = GuitarParser().parse(path, instrument=instrument)
    except (EOFError, OSError, ValueError):
        return False
    return bool(chart.notes)


def _canonical_json_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _catalog_fingerprint(catalog: SongSourceCatalog) -> str:
    """Fingerprint public catalog control files, excluding catalog-root location."""
    manifest_path = catalog.root / CATALOG_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records_value = manifest["records"]
        records_text = (catalog.root / records_value).read_text(encoding="utf-8")
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CatalogValidationError("catalog control files cannot be fingerprinted") from error
    return _canonical_json_hash({"catalog": manifest, "records_jsonl": records_text})


def _require_safe_preprocessing(preprocessing: Mapping[str, object] | None) -> dict[str, object]:
    """Accept portable settings only; paths belong to runtime, never task lineage."""
    value = dict(preprocessing or {})
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise CatalogValidationError("preprocessing settings must be JSON data") from error
    # A JSON path-like string would make task manifests non-portable and leak a
    # local source/runtime location.  Keep this intentionally conservative.
    if "/" in encoded or "\\\\" in encoded:
        raise CatalogValidationError("preprocessing settings must not contain paths")
    return value


def _relative_asset_reference(catalog: SongSourceCatalog, asset: CatalogAsset) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "sha256": asset.sha256,
        "relative_path": asset.path.relative_to(catalog.root).as_posix(),
        "byte_length": asset.byte_length,
        "media_type": asset.media_type,
    }


def _asset_matches(raw: object, asset: CatalogAsset, catalog: SongSourceCatalog) -> bool:
    return isinstance(raw, dict) and raw == _relative_asset_reference(catalog, asset)


def _label_tracks(task_kind: str, track_names: tuple[str, ...]) -> list[str]:
    """Select the declared safe MIDI tracks for one immutable task view."""
    schema = TASK_LABEL_SCHEMAS[task_kind]
    exact_names = schema.get("track_names")
    if exact_names is not None:
        assert isinstance(exact_names, list)  # Static module contract.
        selected = [track_name for track_name in track_names if track_name.upper() in exact_names]
    else:
        prefixes = schema["track_prefixes"]
        assert isinstance(prefixes, list)  # Static module contract.
        selected = [
            track_name
            for track_name in track_names
            if any(track_name.upper().startswith(prefix) for prefix in prefixes)
        ]
    if not selected:
        raise CatalogValidationError("catalog coverage has no track for the declared label schema")
    return selected


def build_catalog_task_manifest(
    catalog_root: str | Path,
    task_kind: str,
    *,
    audio_role: str | None = None,
    fallback_audio_role: str | None = None,
    disable_fallback: bool = False,
    required_difficulty: str = "expert",
    split_ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS,
    split_seed: str = "catalog-source-id/v1",
    preprocessing: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Create an immutable, path-free STRUM task view for one pipeline family."""
    if task_kind not in PIPELINE_IDS:
        raise CatalogValidationError(f"unsupported STRUM catalog task: {task_kind}")
    if not isinstance(split_seed, str) or not split_seed:
        raise CatalogValidationError("split seed must be a non-empty identifier")
    instrument = TASK_INSTRUMENTS[task_kind]
    default_preferred, default_fallback = DEFAULT_AUDIO_ROLES[task_kind]
    preferred = audio_role or default_preferred
    fallback = (
        None
        if disable_fallback
        else (default_fallback if fallback_audio_role is None else fallback_audio_role)
    )
    if preferred not in AUDIO_ROLES or (fallback is not None and fallback not in AUDIO_ROLES):
        raise CatalogValidationError("task audio role is unsupported by the catalog contract")
    settings = _require_safe_preprocessing(preprocessing)
    catalog = load_catalog(catalog_root)

    selected_roles: dict[str, str] = {}
    for role in (preferred, fallback):
        if role is None:
            continue
        for source in select_training_sources(
            catalog,
            instrument,
            required_difficulties=(required_difficulty,),
            audio_role=role,
        ):
            selected_roles.setdefault(source.source_id, role)

    records = {record.source_id: record for record in catalog.records}
    songs: list[dict[str, object]] = []
    for source_id in sorted(selected_roles):
        record = records[source_id]
        role = selected_roles[source_id]
        coverage = record.instruments[instrument]
        if task_kind in {"section_guitar", "section_bass"} and not _has_section_label_source(
            record, instrument
        ):
            continue
        songs.append(
            {
                "source_id": source_id,
                "instrument": instrument,
                "required_difficulty": required_difficulty,
                "split": deterministic_split(source_id, split_ratios),
                "audio_role": role,
                "label_tracks": _label_tracks(task_kind, coverage.track_names),
                "audio": _relative_asset_reference(catalog, record.audio[role]),
                "notes_midi": _relative_asset_reference(catalog, record.notes_midi),
            }
        )
    counts = Counter(song["split"] for song in songs)
    catalog_fingerprint = _catalog_fingerprint(catalog)
    task = {
        "kind": task_kind,
        "pipeline_id": PIPELINE_IDS[task_kind],
        "instrument": instrument,
        "required_difficulty": required_difficulty,
        "audio_role": preferred,
        "fallback_audio_role": fallback,
        "split_algorithm": SPLIT_ALGORITHM,
        "split_seed": split_seed,
        "split_ratios": list(split_ratios),
        "preprocessing": settings,
        "preprocessing_sha256": _canonical_json_hash(settings),
        # Task views are caller-owned mutable dictionaries; never expose the
        # module-level canonical declaration by reference.
        "label_schema": copy.deepcopy(TASK_LABEL_SCHEMAS[task_kind]),
    }
    return {
        "schema_version": MANIFEST_VERSION,
        "format": MANIFEST_FORMAT,
        "lineage": {
            "catalog_id": catalog.catalog_id,
            "catalog_format": CATALOG_FORMAT,
            "catalog_control_sha256": catalog_fingerprint,
            "pipeline_id": PIPELINE_IDS[task_kind],
            "pipeline_version": 1,
            "split_algorithm": SPLIT_ALGORITHM,
        },
        "task": task,
        "songs": songs,
        "summary": {"record_count": len(songs), "by_split": dict(sorted(counts.items()))},
    }


def write_catalog_task_manifest(output: str | Path, manifest: Mapping[str, object]) -> Path:
    """Write a portable task view; catalog roots are intentionally absent."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def resolve_catalog_task_manifest_songs(
    manifest: Mapping[str, Any], catalog_root: str | Path
) -> list[dict[str, object]]:
    """Revalidate task lineage then return ephemeral managed paths for training."""
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("format") != MANIFEST_FORMAT
    ):
        raise CatalogValidationError(f"manifest must use {MANIFEST_FORMAT}")
    task = manifest.get("task")
    lineage = manifest.get("lineage")
    if not isinstance(task, dict) or not isinstance(lineage, dict):
        raise CatalogValidationError("manifest task lineage is invalid")
    task_kind = task.get("kind")
    if task_kind not in PIPELINE_IDS or task.get("pipeline_id") != PIPELINE_IDS[task_kind]:
        raise CatalogValidationError("manifest pipeline is invalid")
    if task.get("instrument") != TASK_INSTRUMENTS[task_kind]:
        raise CatalogValidationError("manifest task instrument is invalid")
    if lineage.get("catalog_format") != CATALOG_FORMAT or lineage.get("pipeline_id") != task.get(
        "pipeline_id"
    ):
        raise CatalogValidationError("manifest lineage is invalid")
    raw_ratios = task.get("split_ratios")
    required_difficulty = task.get("required_difficulty")
    preferred = task.get("audio_role")
    fallback = task.get("fallback_audio_role")
    split_seed = task.get("split_seed")
    if (
        not isinstance(raw_ratios, list)
        or len(raw_ratios) != 3
        or not all(isinstance(value, int) for value in raw_ratios)
        or not isinstance(required_difficulty, str)
        or preferred not in AUDIO_ROLES
        or (fallback is not None and fallback not in AUDIO_ROLES)
        or not isinstance(split_seed, str)
        or task.get("split_algorithm") != SPLIT_ALGORITHM
    ):
        raise CatalogValidationError("manifest task settings are invalid")
    settings = _require_safe_preprocessing(
        task.get("preprocessing") if isinstance(task.get("preprocessing"), dict) else None
    )
    if task.get("preprocessing_sha256") != _canonical_json_hash(settings):
        raise CatalogValidationError("manifest preprocessing lineage is invalid")
    if task.get("label_schema") != TASK_LABEL_SCHEMAS[task_kind]:
        raise CatalogValidationError("manifest label schema is invalid")
    ratios = tuple(raw_ratios)
    deterministic_split("octave-src-00000000", ratios)
    catalog = load_catalog(catalog_root)
    if lineage.get("catalog_id") != catalog.catalog_id or lineage.get(
        "catalog_control_sha256"
    ) != _catalog_fingerprint(catalog):
        raise CatalogValidationError("manifest catalog lineage does not match the selected catalog")
    raw_songs = manifest.get("songs")
    if not isinstance(raw_songs, list):
        raise CatalogValidationError("manifest songs must be a list")
    records = {record.source_id: record for record in catalog.records}
    resolved: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    for raw_song in raw_songs:
        if not isinstance(raw_song, dict):
            raise CatalogValidationError("manifest song must be an object")
        source_id = raw_song.get("source_id")
        role = raw_song.get("audio_role")
        label_tracks = raw_song.get("label_tracks")
        if (
            not isinstance(source_id, str)
            or role not in AUDIO_ROLES
            or source_id in seen_source_ids
        ):
            raise CatalogValidationError("manifest song identity is invalid")
        record = records.get(source_id)
        coverage = record.instruments.get(TASK_INSTRUMENTS[task_kind]) if record else None
        if (
            record is None
            or record.training_use != "allowed"
            or coverage is None
            or coverage.status != "present"
            or required_difficulty not in coverage.difficulties
            or role not in record.audio
            or raw_song.get("instrument") != TASK_INSTRUMENTS[task_kind]
            or raw_song.get("required_difficulty") != required_difficulty
            or raw_song.get("split") != deterministic_split(source_id, ratios)
            or label_tracks != _label_tracks(task_kind, coverage.track_names)
            or not _asset_matches(raw_song.get("audio"), record.audio[role], catalog)
            or not _asset_matches(raw_song.get("notes_midi"), record.notes_midi, catalog)
        ):
            raise CatalogValidationError("manifest song is not a valid approved catalog task input")
        seen_source_ids.add(source_id)
        resolved.append(
            {
                "source_id": source_id,
                "split": raw_song["split"],
                "audio_path": str(record.audio[role].path),
                "midi_path": str(record.notes_midi.path),
                "audio_kind": role,
                "pipeline_id": task["pipeline_id"],
                "label_schema": task["label_schema"],
                "label_tracks": label_tracks,
            }
        )
    return resolved
