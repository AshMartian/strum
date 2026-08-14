"""Build and resolve Guitar task manifests from OCTAVE song-source catalogs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.song_source_catalog import (
    CATALOG_FORMAT,
    CatalogAsset,
    CatalogValidationError,
    SongSourceCatalog,
    load_catalog,
    select_training_sources,
)

MANIFEST_FORMAT = "strum-guitar-catalog-manifest/v1"
MANIFEST_VERSION = 1
DEFAULT_SPLIT_RATIOS = (80, 10, 10)


def deterministic_split(source_id: str, ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS) -> str:
    """Assign a stable split without depending on catalog order or display metadata."""
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or sum(ratios) != 100:
        raise CatalogValidationError("split ratios must be three non-negative values totaling 100")
    bucket = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < ratios[0]:
        return "train"
    if bucket < ratios[0] + ratios[1]:
        return "val"
    return "test"


def _relative_asset_reference(catalog: SongSourceCatalog, asset: CatalogAsset) -> dict[str, object]:
    return {
        "asset_id": asset.asset_id,
        "sha256": asset.sha256,
        "relative_path": asset.path.relative_to(catalog.root).as_posix(),
        "byte_length": asset.byte_length,
        "media_type": asset.media_type,
    }


def build_guitar_manifest(
    catalog_root: str | Path,
    *,
    audio_role: str = "guitar",
    fallback_audio_role: str | None = "mix",
    required_difficulty: str = "expert",
    split_ratios: tuple[int, int, int] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, object]:
    """Create a path-free Guitar manifest from rights-approved catalog records."""
    catalog = load_catalog(catalog_root)
    roles = tuple(role for role in (audio_role, fallback_audio_role) if role is not None)
    if not roles:
        raise CatalogValidationError("at least one audio role is required")
    selected_roles: dict[str, str] = {}
    for role in roles:
        for source in select_training_sources(
            catalog,
            "guitar",
            required_difficulties=(required_difficulty,),
            audio_role=role,
        ):
            selected_roles.setdefault(source.source_id, role)

    records = {record.source_id: record for record in catalog.records}
    songs: list[dict[str, object]] = []
    for source_id in sorted(selected_roles):
        record = records[source_id]
        role = selected_roles[source_id]
        audio = record.audio[role]
        songs.append(
            {
                "source_id": source_id,
                "instrument": "guitar",
                "required_difficulty": required_difficulty,
                "split": deterministic_split(source_id, split_ratios),
                "audio_role": role,
                "audio": _relative_asset_reference(catalog, audio),
                "notes_midi": _relative_asset_reference(catalog, record.notes_midi),
            }
        )
    counts = Counter(song["split"] for song in songs)
    return {
        "schema_version": MANIFEST_VERSION,
        "format": MANIFEST_FORMAT,
        "catalog": {"catalog_id": catalog.catalog_id, "format": CATALOG_FORMAT},
        "task": {
            "instrument": "guitar",
            "required_difficulty": required_difficulty,
            "audio_role": audio_role,
            "fallback_audio_role": fallback_audio_role,
            "split_ratios": list(split_ratios),
        },
        "songs": songs,
        "summary": {"record_count": len(songs), "by_split": dict(sorted(counts.items()))},
    }


def write_guitar_manifest(output: str | Path, manifest: dict[str, object]) -> Path:
    """Write a task manifest without embedding a catalog root or source location."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def _asset_matches(raw: object, asset: CatalogAsset, catalog: SongSourceCatalog) -> bool:
    if not isinstance(raw, dict):
        return False
    return raw == _relative_asset_reference(catalog, asset)


def resolve_guitar_manifest_songs(
    manifest: dict[str, Any], catalog_root: str | Path
) -> list[dict[str, object]]:
    """Re-validate a catalog manifest and return runtime-only managed asset paths."""
    if (
        manifest.get("schema_version") != MANIFEST_VERSION
        or manifest.get("format") != MANIFEST_FORMAT
    ):
        raise CatalogValidationError(f"manifest must use {MANIFEST_FORMAT}")
    catalog = load_catalog(catalog_root)
    catalog_info = manifest.get("catalog")
    if not isinstance(catalog_info, dict) or catalog_info.get("catalog_id") != catalog.catalog_id:
        raise CatalogValidationError("manifest catalog does not match the selected catalog")
    raw_songs = manifest.get("songs")
    if not isinstance(raw_songs, list):
        raise CatalogValidationError("manifest songs must be a list")
    task = manifest.get("task")
    if not isinstance(task, dict) or task.get("instrument") != "guitar":
        raise CatalogValidationError("manifest task is invalid")
    required_difficulty = task.get("required_difficulty")
    raw_ratios = task.get("split_ratios")
    if (
        not isinstance(required_difficulty, str)
        or not isinstance(raw_ratios, list)
        or len(raw_ratios) != 3
        or not all(isinstance(ratio, int) for ratio in raw_ratios)
    ):
        raise CatalogValidationError("manifest task settings are invalid")
    split_ratios = tuple(raw_ratios)
    deterministic_split("octave-src-00000000", split_ratios)
    records = {record.source_id: record for record in catalog.records}
    resolved: list[dict[str, object]] = []
    seen_source_ids: set[str] = set()
    for raw_song in raw_songs:
        if not isinstance(raw_song, dict):
            raise CatalogValidationError("manifest song must be an object")
        source_id = raw_song.get("source_id")
        role = raw_song.get("audio_role")
        split = raw_song.get("split")
        if (
            not isinstance(source_id, str)
            or not isinstance(role, str)
            or not isinstance(split, str)
            or source_id in seen_source_ids
        ):
            raise CatalogValidationError("manifest song identity is invalid")
        record = records.get(source_id)
        if record is None or record.training_use != "allowed" or role not in record.audio:
            raise CatalogValidationError("manifest song is not an approved catalog input")
        coverage = record.instruments.get("guitar")
        if (
            coverage is None
            or coverage.status != "present"
            or raw_song.get("required_difficulty") != required_difficulty
            or required_difficulty not in coverage.difficulties
            or raw_song.get("instrument") != "guitar"
            or split != deterministic_split(source_id, split_ratios)
        ):
            raise CatalogValidationError("manifest song guitar coverage is invalid")
        if not _asset_matches(
            raw_song.get("audio"), record.audio[role], catalog
        ) or not _asset_matches(raw_song.get("notes_midi"), record.notes_midi, catalog):
            raise CatalogValidationError("manifest asset does not match the catalog")
        seen_source_ids.add(source_id)
        resolved.append(
            {
                "source_id": source_id,
                "split": raw_song.get("split"),
                "audio_path": str(record.audio[role].path),
                "midi_path": str(record.notes_midi.path),
                "audio_kind": role,
            }
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Guitar task manifest from an OCTAVE catalog."
    )
    parser.add_argument("catalog", type=Path, help="OCTAVE catalog root")
    parser.add_argument("--output", type=Path, required=True, help="path for a new task manifest")
    parser.add_argument("--audio-role", default="guitar", help="preferred catalog audio role")
    parser.add_argument(
        "--fallback-audio-role", default="mix", help="fallback role; use none to disable"
    )
    parser.add_argument("--required-difficulty", default="expert")
    args = parser.parse_args()
    fallback = None if args.fallback_audio_role.lower() == "none" else args.fallback_audio_role
    manifest = build_guitar_manifest(
        args.catalog,
        audio_role=args.audio_role,
        fallback_audio_role=fallback,
        required_difficulty=args.required_difficulty,
    )
    output = write_guitar_manifest(args.output, manifest)
    print(json.dumps({"output": output.name, "record_count": manifest["summary"]["record_count"]}))


if __name__ == "__main__":
    main()
