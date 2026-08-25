"""Build and resolve Drums task manifests from OCTAVE song-source catalogs.

The manifest is intentionally portable: it contains only catalog-relative,
content-addressed asset references.  MIDI labels are derived inside STRUM at
preprocessing time; OCTAVE never needs to know STRUM's drum-class encoding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.catalog_guitar_manifest import DEFAULT_SPLIT_RATIOS, deterministic_split
from src.song_source_catalog import (
    CATALOG_FORMAT,
    CatalogAsset,
    CatalogValidationError,
    SongSourceCatalog,
    load_catalog,
    select_training_sources,
)

MANIFEST_FORMAT = "strum-drums-onset-catalog-manifest/v1"
MANIFEST_VERSION = 1
PIPELINE_ID = "drums.onset-classifier"
PIPELINE_VERSION = 1
SPLIT_POLICY = "sha256-source-id/v1"


def _asset_reference(catalog: SongSourceCatalog, asset: CatalogAsset) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "sha256": asset.sha256,
        "relative_path": asset.path.relative_to(catalog.root).as_posix(),
        "byte_length": asset.byte_length,
        "media_type": asset.media_type,
    }


def _asset_matches(raw: object, asset: CatalogAsset, catalog: SongSourceCatalog) -> bool:
    return isinstance(raw, dict) and raw == _asset_reference(catalog, asset)


def _catalog_content_sha256(catalog: SongSourceCatalog) -> str:
    """Fingerprint the validated catalog contract, never locations or display text."""
    payload = {
        "catalog_id": catalog.catalog_id,
        "records": [
            {
                "source_id": record.source_id,
                "training_use": record.training_use,
                "notes_midi_sha256": record.notes_midi.sha256,
                "instruments": {
                    name: {"status": coverage.status, "difficulties": sorted(coverage.difficulties)}
                    for name, coverage in sorted(record.instruments.items())
                },
                "audio": {name: asset.sha256 for name, asset in sorted(record.audio.items())},
            }
            for record in catalog.records
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_drums_manifest(
    catalog_root: str | Path,
    *,
    audio_role: str = "drums",
    fallback_audio_role: str | None = "mix",
    required_difficulty: str = "expert",
    split_ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, object]:
    """Build a rights-approved Expert Drums onset/classification task view."""
    catalog = load_catalog(catalog_root)
    roles = tuple(role for role in (audio_role, fallback_audio_role) if role is not None)
    if not roles:
        raise CatalogValidationError("at least one audio role is required")
    selected_roles: dict[str, str] = {}
    for role in roles:
        for source in select_training_sources(
            catalog,
            "drums",
            required_difficulties=(required_difficulty,),
            audio_role=role,
        ):
            selected_roles.setdefault(source.source_id, role)

    records = {record.source_id: record for record in catalog.records}
    songs: list[dict[str, object]] = []
    for source_id in sorted(selected_roles):
        record = records[source_id]
        role = selected_roles[source_id]
        songs.append(
            {
                "source_id": source_id,
                "instrument": "drums",
                "required_difficulty": required_difficulty,
                "split": deterministic_split(source_id, split_ratios),
                "audio_role": role,
                "audio": _asset_reference(catalog, record.audio[role]),
                "notes_midi": _asset_reference(catalog, record.notes_midi),
            }
        )
    counts = Counter(song["split"] for song in songs)
    return {
        "schema_version": MANIFEST_VERSION,
        "format": MANIFEST_FORMAT,
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "format": CATALOG_FORMAT,
            "content_sha256": _catalog_content_sha256(catalog),
        },
        "task": {
            "pipeline_id": PIPELINE_ID,
            "pipeline_version": PIPELINE_VERSION,
            "instrument": "drums",
            "required_difficulty": required_difficulty,
            "audio_role": audio_role,
            "fallback_audio_role": fallback_audio_role,
            "split_policy": {"id": SPLIT_POLICY, "ratios": list(split_ratios)},
            "label_encoding": "strum-drums-8-lane/v1",
        },
        "songs": songs,
        "summary": {"record_count": len(songs), "by_split": dict(sorted(counts.items()))},
    }


def write_drums_manifest(output: str | Path, manifest: dict[str, object]) -> Path:
    """Write a portable task view without the catalog root or source locations."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def resolve_drums_manifest_songs(
    manifest: dict[str, Any], catalog_root: str | Path
) -> list[dict[str, object]]:
    """Revalidate a Drums task view and resolve paths only for this process."""
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("format") != MANIFEST_FORMAT
    ):
        raise CatalogValidationError(f"manifest must use {MANIFEST_FORMAT}")
    catalog = load_catalog(catalog_root)
    catalog_info = manifest.get("catalog")
    task = manifest.get("task")
    raw_songs = manifest.get("songs")
    if (
        not isinstance(catalog_info, dict)
        or catalog_info.get("catalog_id") != catalog.catalog_id
        or not isinstance(task, dict)
        or task.get("pipeline_id") != PIPELINE_ID
        or task.get("pipeline_version") != PIPELINE_VERSION
        or task.get("instrument") != "drums"
        or task.get("runtime_admission") is not None
        or not isinstance(raw_songs, list)
    ):
        raise CatalogValidationError("manifest task is invalid")
    difficulty = task.get("required_difficulty")
    split_policy = task.get("split_policy")
    raw_ratios = split_policy.get("ratios") if isinstance(split_policy, dict) else None
    if (
        not isinstance(difficulty, str)
        or not isinstance(split_policy, dict)
        or split_policy.get("id") != SPLIT_POLICY
        or not isinstance(raw_ratios, list)
        or len(raw_ratios) != 3
        or not all(isinstance(value, int) for value in raw_ratios)
    ):
        raise CatalogValidationError("manifest task settings are invalid")
    ratios = tuple(raw_ratios)
    deterministic_split("octave-src-00000000", ratios)
    if catalog_info.get("content_sha256") != _catalog_content_sha256(catalog):
        raise CatalogValidationError("manifest catalog content does not match the selected catalog")
    records = {record.source_id: record for record in catalog.records}
    seen: set[str] = set()
    resolved: list[dict[str, object]] = []
    for song in raw_songs:
        if not isinstance(song, dict):
            raise CatalogValidationError("manifest song must be an object")
        source_id, role, split = song.get("source_id"), song.get("audio_role"), song.get("split")
        if (
            not isinstance(source_id, str)
            or not isinstance(role, str)
            or not isinstance(split, str)
            or source_id in seen
        ):
            raise CatalogValidationError("manifest song identity is invalid")
        record = records.get(source_id)
        coverage = record.instruments.get("drums") if record else None
        if (
            record is None
            or record.training_use != "allowed"
            or role not in record.audio
            or coverage is None
            or coverage.status != "present"
            or difficulty not in coverage.difficulties
            or song.get("instrument") != "drums"
            or song.get("required_difficulty") != difficulty
            or split != deterministic_split(source_id, ratios)
            or not _asset_matches(song.get("audio"), record.audio[role], catalog)
            or not _asset_matches(song.get("notes_midi"), record.notes_midi, catalog)
        ):
            raise CatalogValidationError("manifest song is not an approved Drums catalog input")
        seen.add(source_id)
        resolved.append(
            {
                "source_id": source_id,
                "split": split,
                "audio_path": str(record.audio[role].path),
                "midi_path": str(record.notes_midi.path),
                "audio_kind": role,
                "input_hashes": {
                    "audio_sha256": record.audio[role].sha256,
                    "notes_midi_sha256": record.notes_midi.sha256,
                },
            }
        )
    return resolved


def task_view_sha256(manifest: dict[str, Any]) -> str:
    """Stable hash used to link preprocessed caches and model checkpoints to a view."""
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Drums task manifest from an OCTAVE catalog."
    )
    parser.add_argument("catalog", type=Path, help="OCTAVE catalog root")
    parser.add_argument("--output", type=Path, required=True, help="path for a new task manifest")
    parser.add_argument("--audio-role", default="drums", help="preferred catalog audio role")
    parser.add_argument(
        "--fallback-audio-role", default="mix", help="fallback role; use none to disable"
    )
    parser.add_argument("--required-difficulty", default="expert")
    args = parser.parse_args()
    fallback = None if args.fallback_audio_role.lower() == "none" else args.fallback_audio_role
    manifest = build_drums_manifest(
        args.catalog,
        audio_role=args.audio_role,
        fallback_audio_role=fallback,
        required_difficulty=args.required_difficulty,
    )
    output = write_drums_manifest(args.output, manifest)
    print(json.dumps({"output": output.name, "record_count": manifest["summary"]["record_count"]}))


if __name__ == "__main__":
    main()
