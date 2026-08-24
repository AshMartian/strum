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

import mido

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
from src.vocal_audio_compatibility import (
    VOCAL_AUDIO_COMPATIBILITY,
    has_compatible_vocal_audio,
)

MANIFEST_FORMAT = "strum-catalog-task-manifest/v1"
MANIFEST_VERSION = 1
# v1 recorded a split seed but accidentally omitted it from the digest.  Keep
# resolving v1 views exactly as written, but publish v2 for every new task
# view so the declared seed is an actual immutable part of split assignment.
LEGACY_SPLIT_ALGORITHM = "sha256-source-id-mod-100/v1"
SPLIT_ALGORITHM = "sha256-source-id-seed-mod-100/v2"
DEFAULT_SPLIT_RATIOS = (80, 10, 10)

# OCTAVE catalog coverage is intentionally a lightweight import-time summary.
# Lead-Vocal training derives four distinct targets from one exact MIDI track,
# so it needs a stricter, STRUM-owned admission predicate. Keeping this set
# together guarantees that the activity, phrase, lyric, and talky task views
# begin from the same subset when they use the same catalog/audio/split options.
VOCAL_TARGET_TASK_KINDS = frozenset(
    {
        "vocals",
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    }
)
VOCAL_TARGET_COMPATIBILITY = "mido-standard-midi-exact-part-vocals/v1"

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
    # Phrase boundaries have an independently observable event language in
    # ``PART VOCALS`` (marker 105/106 or a standard 105 marker span).  Keep
    # that experiment distinct from the activity/pitch task so neither task
    # can accidentally claim the other's component or future profile role.
    "vocals_phrase_boundaries": "vocals.phrase-boundaries/v1",
    # Lyric text is a distinct event stream in PART VOCALS.  It must not be
    # folded into pitched-activity or phrase-marker targets simply because all
    # three live in the same MIDI track.
    "vocals_lyric_alignment": "vocals.lyric-alignment/v1",
    # Pitchless/talky spans are an independent source event language: note 96
    # on the exact lead vocal track.  Do not make them a pseudo-pitch class in
    # the sung-note activity task.
    "vocals_talky_activity": "vocals.talky-activity/v1",
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
    "vocals_phrase_boundaries": ("vocals", "mix"),
    "vocals_lyric_alignment": ("vocals", "mix"),
    "vocals_talky_activity": ("vocals", "mix"),
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
    "vocals_phrase_boundaries": "vocals",
    "vocals_lyric_alignment": "vocals",
    "vocals_talky_activity": "vocals",
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
        # V2 makes the source identity match the generic Bass descriptor and
        # the concrete preprocessor.  ``PART BASS ALT`` is an alternate
        # arrangement, not an additional Bass label stream to merge.
        "id": "five-lane-midi/v2",
        "track_names": ["PART BASS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "bass_onset_fret": {
        "id": "five-lane-midi/v2",
        "track_names": ["PART BASS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "keys_onset_fret": {
        "id": "five-lane-midi/v2",
        "track_names": ["PART KEYS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "vocals_activity": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        # Keep this existing V1 activity task schema stable.  Its preprocessor
        # still rejects any resolved selection other than exact PART VOCALS;
        # the new phrase-boundary task below can use an exact-name schema from
        # its first release onward.
        "track_prefixes": ["PART VOCALS"],
        "difficulty_encoding": "vocal-pitch-phrase-events/v1",
    },
    "vocals_phrase_boundaries": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        "track_names": ["PART VOCALS"],
        "difficulty_encoding": "vocal-phrase-boundary-events/v1",
    },
    "vocals_lyric_alignment": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        "track_names": ["PART VOCALS"],
        "difficulty_encoding": "vocal-lyric-meta-events/v1",
    },
    "vocals_talky_activity": {
        "id": "vocals-pitch-phrase-lyrics-midi/v1",
        "track_names": ["PART VOCALS"],
        "difficulty_encoding": "vocal-pitchless-talky-note-96-spans/v1",
    },
    "keys": {
        "id": "five-lane-midi/v2",
        "track_names": ["PART KEYS"],
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

# Existing task views used a prefix declaration despite the Bass and Keys
# trainers reading only one exact track.  Keep them resolvable only when their
# *actual selected label list* is exactly the current V2 selection; a legacy
# view that included an alternate track fails revalidation rather than gaining
# implicit union semantics.  New task views always carry the V2 declaration.
LEGACY_EXACT_FIVE_LANE_LABEL_SCHEMAS: dict[str, dict[str, object]] = {
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
    "keys": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART KEYS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
    "keys_onset_fret": {
        "id": "five-lane-midi/v1",
        "track_prefixes": ["PART KEYS"],
        "difficulty_encoding": "five-lane-note-ranges/v1",
    },
}


def task_label_schema_is_supported(task_kind: str, value: object) -> bool:
    """Return whether a task-view label schema has safe current semantics.

    The narrow legacy exception is a migration reader, not a generation path.
    Callers still resolve every song through :func:`_label_tracks`, whose V2
    exact identities reject legacy prefix-selected alternate tracks.
    """
    return value == TASK_LABEL_SCHEMAS.get(task_kind) or value == (
        LEGACY_EXACT_FIVE_LANE_LABEL_SCHEMAS.get(task_kind)
    )


def deterministic_split(
    source_id: str,
    ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS,
    *,
    seed: str | None = None,
) -> str:
    """Assign a stable song-disjoint split, optionally keyed by a declared seed.

    ``seed=None`` is the legacy v1 algorithm and exists only to resolve task
    views created before the seed was honored.  New task views use v2 and pass
    their immutable ``split_seed`` explicitly.
    """
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) != 100:
        raise CatalogValidationError("split ratios must be three non-negative values totaling 100")
    if seed is not None and (not isinstance(seed, str) or not seed):
        raise CatalogValidationError("split seed must be a non-empty string")
    identity = source_id if seed is None else f"{seed}\x00{source_id}"
    bucket = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:8], 16) % 100
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


def _has_exact_lead_vocal_label_source(record: object) -> bool:
    """Return whether a managed MIDI target is safe for all lead Vocal tasks.

    OCTAVE can record an imported package as having Vocal coverage without
    claiming that every byte sequence its permissive importer accepts is a
    standard MIDI file mido can decode. STRUM's four lead components must
    never quietly disagree on that boundary: they train from one exact
    ``PART VOCALS`` event stream and therefore share this one fail-closed
    compatibility predicate.
    """
    notes_midi = getattr(record, "notes_midi", None)
    path = getattr(notes_midi, "path", None)
    if not isinstance(path, Path):
        return False
    try:
        midi = mido.MidiFile(path)
    except (EOFError, OSError, ValueError):
        return False
    return sum(track.name == "PART VOCALS" for track in midi.tracks) == 1


def _select_compatible_vocal_audio_role(
    record: object, preferred: str, fallback: str | None
) -> str | None:
    """Pick the declared Vocal role the implemented preprocessors can decode.

    The order is exactly the task's preferred/fallback contract.  A bad
    preferred stem therefore does not suppress a usable declared fallback,
    while a task with fallback disabled has no implicit substitute.
    """
    audio = getattr(record, "audio", None)
    if not isinstance(audio, dict):
        return None
    seen_roles: set[str] = set()
    for role in (preferred, fallback):
        if role is None or role in seen_roles:
            continue
        seen_roles.add(role)
        asset = audio.get(role)
        path = getattr(asset, "path", None)
        if isinstance(path, Path) and has_compatible_vocal_audio(path):
            return role
    return None


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
    if task_kind in VOCAL_TARGET_TASK_KINDS:
        # ``vocals_activity`` retains a V1 prefix declaration only so old
        # manifests can be recognized. The canonical lead target is still one
        # exact PART VOCALS track; an ALT arrangement cannot become another
        # component's label source merely because it shares that prefix.
        selected = [track_name for track_name in track_names if track_name == "PART VOCALS"]
        if not selected:
            raise CatalogValidationError(
                "catalog coverage has no track for the declared label schema"
            )
        return selected
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

    available_roles: dict[str, list[str]] = {}
    for role in (preferred, fallback):
        if role is None:
            continue
        for source in select_training_sources(
            catalog,
            instrument,
            required_difficulties=(required_difficulty,),
            audio_role=role,
        ):
            available_roles.setdefault(source.source_id, []).append(role)

    records = {record.source_id: record for record in catalog.records}
    songs: list[dict[str, object]] = []
    vocal_target_exclusions = 0
    vocal_audio_exclusions = 0
    for source_id in sorted(available_roles):
        record = records[source_id]
        coverage = record.instruments[instrument]
        if task_kind in {"section_guitar", "section_bass"} and not _has_section_label_source(
            record, instrument
        ):
            continue
        if task_kind in VOCAL_TARGET_TASK_KINDS and not _has_exact_lead_vocal_label_source(record):
            vocal_target_exclusions += 1
            continue
        role = available_roles[source_id][0]
        if task_kind in VOCAL_TARGET_TASK_KINDS:
            role = _select_compatible_vocal_audio_role(record, preferred, fallback)
            if role is None:
                vocal_audio_exclusions += 1
                continue
        songs.append(
            {
                "source_id": source_id,
                "instrument": instrument,
                "required_difficulty": required_difficulty,
                "split": deterministic_split(source_id, split_ratios, seed=split_seed),
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
    summary: dict[str, object] = {
        "record_count": len(songs),
        "by_split": dict(sorted(counts.items())),
    }
    if task_kind in VOCAL_TARGET_TASK_KINDS:
        # The task view remains private and contains no source locations. An
        # aggregate lets OCTAVE explain why approved catalog records did not
        # become lead-training examples without revealing source identities.
        summary["target_compatibility"] = {
            "format": VOCAL_TARGET_COMPATIBILITY,
            "excluded_record_count": vocal_target_exclusions,
            "audio": {
                "format": VOCAL_AUDIO_COMPATIBILITY,
                "excluded_record_count": vocal_audio_exclusions,
            },
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
        "summary": summary,
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
        or task.get("split_algorithm") not in {LEGACY_SPLIT_ALGORITHM, SPLIT_ALGORITHM}
    ):
        raise CatalogValidationError("manifest task settings are invalid")
    settings = _require_safe_preprocessing(
        task.get("preprocessing") if isinstance(task.get("preprocessing"), dict) else None
    )
    if task.get("preprocessing_sha256") != _canonical_json_hash(settings):
        raise CatalogValidationError("manifest preprocessing lineage is invalid")
    if not task_label_schema_is_supported(task_kind, task.get("label_schema")):
        raise CatalogValidationError("manifest label schema is invalid")
    ratios = tuple(raw_ratios)
    use_seed = split_seed if task.get("split_algorithm") == SPLIT_ALGORITHM else None
    deterministic_split("octave-src-00000000", ratios, seed=use_seed)
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
        compatible_vocal_role = (
            _select_compatible_vocal_audio_role(record, preferred, fallback)
            if task_kind in VOCAL_TARGET_TASK_KINDS and record is not None
            else None
        )
        if (
            record is None
            or record.training_use != "allowed"
            or coverage is None
            or coverage.status != "present"
            or required_difficulty not in coverage.difficulties
            or role not in record.audio
            or raw_song.get("instrument") != TASK_INSTRUMENTS[task_kind]
            or raw_song.get("required_difficulty") != required_difficulty
            or raw_song.get("split") != deterministic_split(source_id, ratios, seed=use_seed)
            or label_tracks != _label_tracks(task_kind, coverage.track_names)
            or not _asset_matches(raw_song.get("audio"), record.audio[role], catalog)
            or not _asset_matches(raw_song.get("notes_midi"), record.notes_midi, catalog)
            or (
                task_kind in VOCAL_TARGET_TASK_KINDS
                and (
                    not _has_exact_lead_vocal_label_source(record) or role != compatible_vocal_role
                )
            )
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
