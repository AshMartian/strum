"""Validate OCTAVE's explicit source policy for Vocal harmony experiments.

``PART HARM1``/``HARM2``/``HARM3`` are independent chart targets.  A normal
``vocals`` or ``mix`` asset may contain any combination of those voices and
the lead, so STRUM must never derive Harmony supervision from it.  OCTAVE
materializes an optional, content-addressed sidecar next to its canonical
catalog.  This module turns only that sidecar into a path-free task view.

The sidecar is deliberately a source-policy contract, not a model or an
execution profile.  It can attest either an original isolated stem or a
separation output with pinned input/model/configuration identities.  STRUM can
verify those identities and fails closed whenever they are absent; it cannot
infer isolation from audio role names or a chart track.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import mido

from src.catalog_task_manifest import (
    CATALOG_FORMAT,
    SPLIT_ALGORITHM,
    _catalog_fingerprint,
    _relative_asset_reference,
    deterministic_split,
)
from src.song_source_catalog import (
    CatalogAsset,
    CatalogValidationError,
    SongSourceCatalog,
    load_catalog,
)

HARMONY_SOURCE_POLICY_FILENAME = "vocal-harmony-sources.json"
HARMONY_SOURCE_POLICY_FORMAT = "octave-vocal-harmony-source-policy/v1"
HARMONY_SOURCE_TASK_FORMAT = "strum-vocal-harmony-source-task/v1"
HARMONY_SOURCE_PIPELINE_ID = "vocals.harmony-source-policy/v1"
HARMONY_TRACK_ROLES = {"HARM1": "harm1", "HARM2": "harm2", "HARM3": "harm3"}
HARMONY_TRACKS = tuple(HARMONY_TRACK_ROLES)
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,127}$")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _safe_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise CatalogValidationError(f"vocal harmony policy {label} is invalid")
    return value


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CatalogValidationError(f"vocal harmony policy {label} must be a SHA-256")
    return value


def _asset_identity(raw: object, asset: CatalogAsset, label: str) -> dict[str, str]:
    if not isinstance(raw, dict) or set(raw) != {"asset_id", "sha256"}:
        raise CatalogValidationError(f"vocal harmony policy {label} asset identity is invalid")
    asset_id = raw.get("asset_id")
    sha256 = raw.get("sha256")
    if asset_id != asset.asset_id or sha256 != asset.sha256:
        raise CatalogValidationError(f"vocal harmony policy {label} asset does not match catalog")
    return {"asset_id": asset.asset_id, "sha256": asset.sha256}


def _parse_provenance(raw: object, record: object, track: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise CatalogValidationError("vocal harmony policy isolation provenance is invalid")
    kind = raw.get("kind")
    if kind == "isolated_source_stem/v1":
        if set(raw) != {"kind", "timeline", "attestation_id"}:
            raise CatalogValidationError("source-stem provenance contains unsupported fields")
        if raw.get("timeline") != "same-master-timeline/v1":
            raise CatalogValidationError("source-stem provenance requires same-master timeline")
        return {
            "kind": kind,
            "timeline": "same-master-timeline/v1",
            "attestation_id": _safe_identifier(raw.get("attestation_id"), "attestation_id"),
        }
    if kind == "isolated_separation_output/v1":
        if set(raw) != {"kind", "timeline", "input", "separator"}:
            raise CatalogValidationError("separation provenance contains unsupported fields")
        if raw.get("timeline") != "same-master-timeline/v1":
            raise CatalogValidationError("separation provenance requires same-master timeline")
        mix = getattr(record, "audio", {}).get("mix")
        if mix is None:
            raise CatalogValidationError("separated harmony output requires a catalog mix input")
        input_identity = _asset_identity(raw.get("input"), mix, "separation input")
        separator = raw.get("separator")
        if not isinstance(separator, dict) or set(separator) != {
            "id",
            "version",
            "model_sha256",
            "configuration_sha256",
        }:
            raise CatalogValidationError("separation provenance requires pinned separator metadata")
        return {
            "kind": kind,
            "timeline": "same-master-timeline/v1",
            "input": input_identity,
            "separator": {
                "id": _safe_identifier(separator.get("id"), "separator id"),
                "version": _safe_identifier(separator.get("version"), "separator version"),
                "model_sha256": _sha(separator.get("model_sha256"), "separator model_sha256"),
                "configuration_sha256": _sha(
                    separator.get("configuration_sha256"), "separator configuration_sha256"
                ),
            },
        }
    raise CatalogValidationError(
        f"vocal harmony policy {track} requires isolated-stem or pinned separation provenance"
    )


def _midi_has_track(path: Path, track_name: str) -> bool:
    try:
        midi = mido.MidiFile(path)
    except (EOFError, OSError, ValueError) as error:
        raise CatalogValidationError("vocal harmony notes MIDI is unreadable") from error
    return any(track.name == track_name for track in midi.tracks)


def _read_policy(catalog: SongSourceCatalog) -> tuple[str, dict[str, object]]:
    path = catalog.root / HARMONY_SOURCE_POLICY_FILENAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise CatalogValidationError("vocal harmony source policy is missing") from error
    except (OSError, json.JSONDecodeError) as error:
        raise CatalogValidationError("vocal harmony source policy is unreadable") from error
    if not isinstance(raw, dict) or set(raw) != {
        "schema_version",
        "format",
        "policy_id",
        "catalog_id",
        "catalog_control_sha256",
        "records",
    }:
        raise CatalogValidationError("vocal harmony source policy contains unsupported fields")
    if raw.get("schema_version") != 1 or raw.get("format") != HARMONY_SOURCE_POLICY_FORMAT:
        raise CatalogValidationError(
            f"vocal harmony source policy must use {HARMONY_SOURCE_POLICY_FORMAT}"
        )
    _safe_identifier(raw.get("policy_id"), "policy_id")
    if raw.get("catalog_id") != catalog.catalog_id or raw.get(
        "catalog_control_sha256"
    ) != _catalog_fingerprint(catalog):
        raise CatalogValidationError("vocal harmony source policy catalog lineage does not match")
    if not isinstance(raw.get("records"), list):
        raise CatalogValidationError("vocal harmony source policy records must be a list")
    return _sha256_json(raw), raw


def _selected_tracks(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return HARMONY_TRACKS
    if not isinstance(value, Sequence) or isinstance(value, str) or not value:
        raise CatalogValidationError("harmony_tracks must be a non-empty array")
    tracks = tuple(value)
    if len(set(tracks)) != len(tracks) or any(track not in HARMONY_TRACK_ROLES for track in tracks):
        raise CatalogValidationError(
            "harmony_tracks must contain unique HARM1, HARM2, or HARM3 values"
        )
    return tracks


def _policy_rows(
    catalog: SongSourceCatalog, policy: Mapping[str, object], tracks: tuple[str, ...]
) -> list[dict[str, object]]:
    records = {record.source_id: record for record in catalog.records}
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    raw_records = policy["records"]
    assert isinstance(raw_records, list)
    for raw in raw_records:
        if not isinstance(raw, dict) or set(raw) != {
            "source_id",
            "track_name",
            "audio",
            "provenance",
        }:
            raise CatalogValidationError(
                "vocal harmony source policy record contains unsupported fields"
            )
        source_id = raw.get("source_id")
        track = raw.get("track_name")
        if not isinstance(source_id, str) or track not in HARMONY_TRACK_ROLES:
            raise CatalogValidationError("vocal harmony source policy track identity is invalid")
        key = (source_id, track)
        if key in seen:
            raise CatalogValidationError(
                "vocal harmony source policy has duplicate source/track entries"
            )
        seen.add(key)
        record = records.get(source_id)
        if record is None:
            raise CatalogValidationError(
                "vocal harmony source policy references an unknown catalog source"
            )
        expected_role = HARMONY_TRACK_ROLES[track]
        audio = raw.get("audio")
        if not isinstance(audio, dict) or set(audio) != {"role", "asset_id", "sha256"}:
            raise CatalogValidationError("vocal harmony source policy audio declaration is invalid")
        if audio.get("role") != expected_role:
            raise CatalogValidationError(
                "vocal harmony source policy audio role does not match HARM track"
            )
        asset = record.audio.get(expected_role)
        if asset is None:
            raise CatalogValidationError(
                "vocal harmony source policy requires an isolated HARM audio role"
            )
        _asset_identity(
            {"asset_id": audio.get("asset_id"), "sha256": audio.get("sha256")}, asset, "output"
        )
        provenance = _parse_provenance(raw.get("provenance"), record, track)
        # Coverage asserts OCTAVE recognized the Vocal authoring family, while
        # the decoded MIDI check prevents a dishonest/incomplete coverage list
        # from becoming Harmony labels.
        coverage = record.instruments.get("vocals")
        if (
            coverage is None
            or coverage.status != "present"
            or "expert" not in coverage.difficulties
        ):
            raise CatalogValidationError("vocal harmony source requires Expert Vocal coverage")
        if track not in coverage.track_names or not _midi_has_track(record.notes_midi.path, track):
            raise CatalogValidationError(
                "vocal harmony source requires the exact declared HARM MIDI track"
            )
        if track in tracks and record.training_use == "allowed":
            rows.append(
                {
                    "source_id": source_id,
                    "track_name": track,
                    "audio_role": expected_role,
                    "audio": _relative_asset_reference(catalog, asset),
                    "notes_midi": _relative_asset_reference(catalog, record.notes_midi),
                    "provenance": provenance,
                }
            )
    return sorted(rows, key=lambda row: (str(row["source_id"]), str(row["track_name"])))


def build_vocal_harmony_source_task(
    catalog_root: str | Path,
    *,
    harmony_tracks: Sequence[str] | None = None,
    split_ratios: tuple[int, int, int] = (80, 10, 10),
    split_seed: str = "catalog-source-id/v1",
) -> dict[str, object]:
    """Build a path-free task view for *approved isolated* HARM sources only."""
    tracks = _selected_tracks(harmony_tracks)
    # Match the shared catalog-task contract: existing immutable seed IDs use
    # ``catalog-source-id/v1``.  A seed is metadata, not a file location, but
    # do reject strings that could introduce a local path into the task view.
    if (
        not isinstance(split_seed, str)
        or not split_seed
        or len(split_seed) > 128
        or split_seed.startswith("/")
        or "\\" in split_seed
        or "://" in split_seed
    ):
        raise CatalogValidationError("vocal harmony split_seed is invalid")
    # Reuse the common split validator and avoid silently creating a per-track
    # split that would leak one song between train and validation.
    deterministic_split("octave-src-00000000", split_ratios, seed=split_seed)
    catalog = load_catalog(catalog_root)
    policy_sha256, policy = _read_policy(catalog)
    rows = _policy_rows(catalog, policy, tracks)
    for row in rows:
        row["split"] = deterministic_split(str(row["source_id"]), split_ratios, seed=split_seed)
    counts = Counter(row["split"] for row in rows)
    source_count = len({str(row["source_id"]) for row in rows})
    return {
        "schema_version": 1,
        "format": HARMONY_SOURCE_TASK_FORMAT,
        "lineage": {
            "catalog_id": catalog.catalog_id,
            "catalog_format": CATALOG_FORMAT,
            "catalog_control_sha256": _catalog_fingerprint(catalog),
            "source_policy_format": HARMONY_SOURCE_POLICY_FORMAT,
            "source_policy_id": policy["policy_id"],
            "source_policy_sha256": policy_sha256,
            "pipeline_id": HARMONY_SOURCE_PIPELINE_ID,
            "pipeline_version": 1,
            "split_algorithm": SPLIT_ALGORITHM,
        },
        "task": {
            "kind": "vocals_harmony_source_policy",
            "pipeline_id": HARMONY_SOURCE_PIPELINE_ID,
            "instrument": "vocals",
            "harmony_tracks": list(tracks),
            "required_difficulty": "expert",
            "audio_policy": {
                "id": "isolated-harmony-stem-only/v1",
                "fallback": None,
                "shared_roles_forbidden": ["mix", "vocals"],
                "required_provenance": [
                    "isolated_source_stem/v1",
                    "isolated_separation_output/v1",
                ],
            },
            "label_schema": {
                "id": "vocal-harmony-midi/v1",
                "track_names": list(tracks),
                "difficulty_encoding": "vocal-pitch-phrase-lyrics-events/v1",
            },
            "split_algorithm": SPLIT_ALGORITHM,
            "split_seed": split_seed,
            "split_ratios": list(split_ratios),
        },
        "sources": rows,
        "summary": {
            "record_count": len(rows),
            "source_count": source_count,
            "by_split": dict(sorted(counts.items())),
        },
    }


def write_vocal_harmony_source_task(output: str | Path, task: Mapping[str, object]) -> Path:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def resolve_vocal_harmony_source_task(
    task: Mapping[str, Any], catalog_root: str | Path
) -> list[dict[str, object]]:
    """Rebuild and compare a task before returning worker-private managed paths."""
    if task.get("schema_version") != 1 or task.get("format") != HARMONY_SOURCE_TASK_FORMAT:
        raise CatalogValidationError(f"vocal harmony task must use {HARMONY_SOURCE_TASK_FORMAT}")
    raw_task = task.get("task")
    if not isinstance(raw_task, dict) or raw_task.get("pipeline_id") != HARMONY_SOURCE_PIPELINE_ID:
        raise CatalogValidationError("vocal harmony task pipeline is invalid")
    tracks = raw_task.get("harmony_tracks")
    ratios = raw_task.get("split_ratios")
    seed = raw_task.get("split_seed")
    if (
        not isinstance(tracks, list)
        or not isinstance(ratios, list)
        or len(ratios) != 3
        or not all(isinstance(value, int) for value in ratios)
        or not isinstance(seed, str)
    ):
        raise CatalogValidationError("vocal harmony task settings are invalid")
    rebuilt = build_vocal_harmony_source_task(
        catalog_root, harmony_tracks=tracks, split_ratios=tuple(ratios), split_seed=seed
    )
    if _canonical(task) != _canonical(rebuilt):
        raise CatalogValidationError(
            "vocal harmony task does not match current approved catalog policy"
        )
    catalog = load_catalog(catalog_root)
    records = {record.source_id: record for record in catalog.records}
    resolved: list[dict[str, object]] = []
    for source in rebuilt["sources"]:
        assert isinstance(source, dict)
        record = records[str(source["source_id"])]
        role = str(source["audio_role"])
        resolved.append(
            {
                "source_id": source["source_id"],
                "track_name": source["track_name"],
                "split": source["split"],
                "audio_path": str(record.audio[role].path),
                "midi_path": str(record.notes_midi.path),
                "audio_role": role,
                "provenance": source["provenance"],
            }
        )
    return resolved


def inspect_vocal_harmony_source_catalog(
    catalog_root: str | Path, *, harmony_tracks: Sequence[str] | None = None
) -> dict[str, object]:
    """Return a bounded path-free readiness summary without exposing policy data."""
    try:
        task = build_vocal_harmony_source_task(catalog_root, harmony_tracks=harmony_tracks)
    except CatalogValidationError:
        catalog = load_catalog(catalog_root)
        return {
            "eligible_count": 0,
            "exclusion_reason_counts": {
                "isolated_harmony_policy_unavailable": len(catalog.records)
            },
            "audio_policy": {
                "kind": "isolated_harmony_stem_only",
                "fallback_role": None,
                "shared_vocals_or_mix_allowed": False,
                "source_policy_required": True,
            },
            "estimated_storage_bytes": 0,
            "storage_estimate_capped": False,
        }
    sources = task["sources"]
    assert isinstance(sources, list)
    unique_assets = {
        str(item["audio"]["asset_id"]): item["audio"] for item in sources if isinstance(item, dict)
    }
    return {
        "eligible_count": len(sources),
        "exclusion_reason_counts": {"isolated_harmony_policy_unavailable": 0},
        "audio_policy": {
            "kind": "isolated_harmony_stem_only",
            "fallback_role": None,
            "shared_vocals_or_mix_allowed": False,
            "source_policy_required": True,
        },
        "estimated_storage_bytes": sum(
            int(asset["byte_length"]) for asset in unique_assets.values() if isinstance(asset, dict)
        ),
        "storage_estimate_capped": False,
    }
