"""Build catalog-backed, five-lane chart-transform task views.

This is the only chart-pair preparation path intended for OCTAVE integration.
It consumes the strict song-source catalog reader rather than source packages
or arbitrary folders, so every persisted task view is path-free and can be
revalidated from catalog IDs and content hashes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from scripts.prepare_guitar_chart_pairs import (
    DIFFICULTY_BASE_NOTES,
    FIVE_LANE_INSTRUMENT_TRACKS,
    PreparationError,
    parse_instrument_difficulties,
)
from src.song_source_catalog import (
    AUDIO_ROLES,
    CATALOG_FILENAME,
    CatalogAsset,
    CatalogRecord,
    SongSourceCatalog,
    load_catalog,
)

PAIR_DATASET_FORMAT = "strum-chart-pairs/v1"
PIPELINE_ID = "chart_transform.five_lane"
PIPELINE_VERSION = 1
PREPROCESSING_ID = "midi-five-lane-events"
PREPROCESSING_VERSION = 1
SPLIT_ALGORITHM = "sha256-source-id-rank/v2-three-way"


@dataclass(frozen=True)
class CatalogChartPairOptions:
    """Stable configuration for one instrument and one difficulty transform."""

    instrument: str
    target_difficulty: str
    split_seed: int = 20260814
    calibration_fraction: float = 0.1
    test_fraction: float = 0.1
    dataset_id: str | None = None
    audio_feature_mode: str = "none"
    audio_role: str | None = None
    fallback_audio_role: str | None = None

    def __post_init__(self) -> None:
        if self.instrument not in FIVE_LANE_INSTRUMENT_TRACKS:
            raise PreparationError(
                f"instrument must be one of: {', '.join(sorted(FIVE_LANE_INSTRUMENT_TRACKS))}"
            )
        if self.target_difficulty not in {"Hard", "Medium", "Easy"}:
            raise PreparationError("target_difficulty must be Hard, Medium, or Easy")
        if not isinstance(self.split_seed, int) or isinstance(self.split_seed, bool):
            raise PreparationError("split_seed must be an integer")
        if (
            not isinstance(self.calibration_fraction, (int, float))
            or isinstance(self.calibration_fraction, bool)
            or not isinstance(self.test_fraction, (int, float))
            or isinstance(self.test_fraction, bool)
            or not 0 < self.calibration_fraction < 1
            or not 0 < self.test_fraction < 1
        ):
            raise PreparationError("calibration_fraction and test_fraction must be between 0 and 1")
        if self.calibration_fraction + self.test_fraction >= 1:
            raise PreparationError("calibration_fraction + test_fraction must leave training data")
        if self.dataset_id is not None and not self.dataset_id.strip():
            raise PreparationError("dataset_id must be non-empty when provided")
        if self.audio_feature_mode not in {"none", "rms_onset_v1"}:
            raise PreparationError("audio_feature_mode must be none or rms_onset_v1")
        if self.audio_feature_mode == "none" and (
            self.audio_role is not None or self.fallback_audio_role is not None
        ):
            raise PreparationError("audio roles require audio_feature_mode")
        if self.audio_feature_mode != "none" and (
            self.audio_role is not None
            and self.audio_role not in AUDIO_ROLES
            or self.fallback_audio_role is not None
            and self.fallback_audio_role not in AUDIO_ROLES
        ):
            raise PreparationError("audio role is unsupported")


def pipeline_descriptor() -> dict[str, object]:
    """Return the path-free STRUM contract OCTAVE can use to configure this task."""
    return {
        "pipeline_id": PIPELINE_ID,
        "pipeline_version": PIPELINE_VERSION,
        "required_catalog": {
            "format": "octave-song-source-catalog/v1",
            "training_use": "allowed",
            "instrument": "one of guitar, bass, keys, drums",
            "difficulties": ["expert", "hard", "medium", "easy"],
        },
        "preprocessing": {
            "id": PREPROCESSING_ID,
            "version": PREPROCESSING_VERSION,
            "lanes": 5,
            "source_difficulty": "Expert",
            "target_difficulties": ["Hard", "Medium", "Easy"],
        },
        "audio_conditioning": {
            "modes": ["none", "rms_onset_v1"],
            "selection": "task-view-declared catalog audio only",
        },
        "split": {"algorithm": SPLIT_ALGORITHM, "unit": "catalog source_id"},
        "training_requirements": [
            "source_disjoint_train_calibration_test/v2",
            "strum_owned_decoder_calibration/v1",
            "test_only_transform_promotion/v1",
        ],
    }


def prepare_catalog_chart_pairs(
    catalog_root: str | Path,
    output_dir: str | Path,
    options: CatalogChartPairOptions,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    """Materialize a trainable chart-pair task view from allowed catalog records.

    The catalog reader validates every managed asset before this function parses
    MIDI. Persisted pair records intentionally contain source IDs and asset
    hashes, never a raw source or local filesystem path.
    """
    catalog = load_catalog(catalog_root)
    required_difficulties = {"expert", options.target_difficulty.lower()}
    selected = tuple(
        record
        for record in catalog.records
        if _record_supports(record, options.instrument, required_difficulties)
    )
    parsed_records, skipped = _parse_selected_records(selected, options)
    if options.audio_feature_mode != "none":
        parsed_records, audio_skipped = _select_audio_conditioning_assets(
            parsed_records, catalog, options
        )
        skipped.extend(audio_skipped)
    if len(parsed_records) < 3:
        details = "; ".join(skipped) if skipped else "fewer than three eligible catalog records"
        raise PreparationError(
            "catalog task view requires at least three valid song records for train/calibration/test: "
            f"{details}"
        )

    split_assignments = _split_assignments(
        [record["source_id"] for record in parsed_records],
        options.split_seed,
        options.calibration_fraction,
        options.test_fraction,
    )
    for record in parsed_records:
        record["split"] = split_assignments[record["source_id"]]

    output = Path(output_dir).expanduser().resolve()
    records_path = output / "pairs.jsonl"
    manifest_path = output / "dataset-manifest.json"
    if not overwrite and (records_path.exists() or manifest_path.exists()):
        raise PreparationError(
            f"output already contains {records_path.name} or {manifest_path.name}; pass --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    records_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in parsed_records) + "\n",
        encoding="utf-8",
    )
    manifest = _dataset_manifest(catalog, options, parsed_records, split_assignments)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "manifest_path": manifest_path,
        "record_count": len(parsed_records),
        "skipped": skipped,
        "task_view_id": manifest["task_view"]["task_view_id"],
    }


def _record_supports(
    record: CatalogRecord, instrument: str, required_difficulties: set[str]
) -> bool:
    coverage = record.instruments.get(instrument)
    return bool(
        record.training_use == "allowed"
        and coverage is not None
        and coverage.status == "present"
        and required_difficulties <= coverage.difficulties
    )


def _parse_selected_records(
    selected: Iterable[CatalogRecord], options: CatalogChartPairOptions
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    skipped: list[str] = []
    for record in sorted(selected, key=lambda item: item.source_id):
        try:
            difficulties = parse_instrument_difficulties(record.notes_midi.path, options.instrument)
        except (OSError, PreparationError, ValueError) as error:
            skipped.append(f"{record.source_id}: {error}")
            continue
        if not difficulties["Expert"]:
            skipped.append(f"{record.source_id}: no Expert five-lane events")
            continue
        records.append(
            {
                "song_id": record.source_id,
                "source_id": record.source_id,
                "notes_midi_sha256": record.notes_midi.sha256,
                "instrument": options.instrument,
                "source_difficulty": "Expert",
                "target_difficulty": options.target_difficulty,
                "source_events": difficulties["Expert"],
                "target_events": difficulties[options.target_difficulty],
            }
        )
    return records, skipped


def _select_audio_conditioning_assets(
    parsed_records: list[dict[str, Any]],
    catalog: SongSourceCatalog,
    options: CatalogChartPairOptions,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Keep only chart pairs with a complete, approved aligned audio asset."""
    records = {record.source_id: record for record in catalog.records}
    selected: list[dict[str, Any]] = []
    skipped: list[str] = []
    preferred_role = options.audio_role or options.instrument
    fallback_role = "mix" if options.fallback_audio_role is None else options.fallback_audio_role
    for pair in parsed_records:
        source_id = pair["source_id"]
        record = records[source_id]
        role, asset = _select_audio_asset(record, preferred_role, fallback_role)
        if asset is None or role is None:
            skipped.append(f"{source_id}: no approved conditioning audio")
            continue
        try:
            duration_ms = sf.info(asset.path).duration * 1000.0
        except (OSError, RuntimeError):
            skipped.append(f"{source_id}: conditioning audio is unreadable")
            continue
        chart_end_ms = max(event["time_ms"] for event in pair["source_events"])
        if chart_end_ms >= duration_ms:
            skipped.append(f"{source_id}: conditioning audio ends before Expert chart")
            continue
        pair["audio_role"] = role
        pair["audio_sha256"] = asset.sha256
        pair["audio_byte_length"] = asset.byte_length
        selected.append(pair)
    return selected, skipped


def _select_audio_asset(
    record: CatalogRecord, preferred_role: str, fallback_role: str | None
) -> tuple[str | None, CatalogAsset | None]:
    for role in (preferred_role, fallback_role):
        if role is not None and role in record.audio:
            return role, record.audio[role]
    return None, None


def _split_assignments(
    source_ids: list[str], seed: int, calibration_fraction: float, test_fraction: float
) -> dict[str, str]:
    if len(set(source_ids)) != len(source_ids):
        raise PreparationError("catalog task view has duplicate source_id values")
    ordered = sorted(source_ids, key=lambda source_id: _split_key(source_id, seed))
    if len(ordered) < 3:
        # Migration-readable V1 task views remain useful for raw experiments,
        # but the promotion contract rejects them because they lack test data.
        return {
            source_id: "validation" if source_id == ordered[0] else "train"
            for source_id in source_ids
        }
    test_count = max(1, round(len(ordered) * test_fraction))
    calibration_count = max(1, round(len(ordered) * calibration_fraction))
    if test_count + calibration_count >= len(ordered):
        raise PreparationError("three-way split leaves no training songs")
    test = set(ordered[:test_count])
    calibration = set(ordered[test_count : test_count + calibration_count])
    return {
        source_id: (
            "test" if source_id in test else "calibration" if source_id in calibration else "train"
        )
        for source_id in source_ids
    }


def _split_key(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{source_id}".encode()).hexdigest()


def _dataset_manifest(
    catalog: SongSourceCatalog,
    options: CatalogChartPairOptions,
    records: list[dict[str, Any]],
    split_assignments: dict[str, str],
) -> dict[str, object]:
    catalog_manifest_path = catalog.root / CATALOG_FILENAME
    records_path = _catalog_records_path(catalog_manifest_path)
    source_inputs = [
        {
            "source_id": record["source_id"],
            "notes_midi_sha256": record["notes_midi_sha256"],
            **(
                {
                    "audio_role": record["audio_role"],
                    "audio_sha256": record["audio_sha256"],
                    "audio_byte_length": record["audio_byte_length"],
                }
                if options.audio_feature_mode != "none"
                else {}
            ),
        }
        for record in records
    ]
    preprocessing = {
        "id": PREPROCESSING_ID,
        "version": PREPROCESSING_VERSION,
        "instrument_track": FIVE_LANE_INSTRUMENT_TRACKS[options.instrument],
        "lane_notes": DIFFICULTY_BASE_NOTES,
        "source_difficulty": "Expert",
        "target_difficulty": options.target_difficulty,
    }
    legacy = "validation" in split_assignments.values()
    split = (
        {
            "algorithm": "sha256-source-id-rank/v1",
            "seed": options.split_seed,
            "validation_fraction": options.calibration_fraction + options.test_fraction,
            "assignments": {
                source_id: split_assignments[source_id] for source_id in sorted(split_assignments)
            },
        }
        if legacy
        else {
            "algorithm": SPLIT_ALGORITHM,
            "seed": options.split_seed,
            "calibration_fraction": options.calibration_fraction,
            "test_fraction": options.test_fraction,
            "assignments": {
                source_id: split_assignments[source_id] for source_id in sorted(split_assignments)
            },
        }
    )
    task_view = {
        "pipeline": {"id": PIPELINE_ID, "version": PIPELINE_VERSION},
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "manifest_sha256": _sha256(catalog_manifest_path),
            "records_sha256": _sha256(records_path),
        },
        "source_inputs": source_inputs,
        "split": split,
        "preprocessing": {
            **preprocessing,
            "config_sha256": _canonical_sha256(preprocessing),
        },
    }
    if options.audio_feature_mode != "none":
        task_view["audio_conditioning"] = {
            "mode": options.audio_feature_mode,
            "preferred_role": options.audio_role or options.instrument,
            "fallback_role": "mix"
            if options.fallback_audio_role is None
            else options.fallback_audio_role,
        }
    task_view["task_view_id"] = _canonical_sha256(task_view)
    return {
        "schema_version": 1,
        "format": PAIR_DATASET_FORMAT,
        "dataset_id": options.dataset_id
        or f"{catalog.catalog_id}-{options.instrument.lower()}-expert-{options.target_difficulty.lower()}",
        "records": "pairs.jsonl",
        "provenance": f"OCTAVE catalog {catalog.catalog_id}; allowed records only",
        "license": "OCTAVE catalog training-use decision; per-record rights retained by catalog",
        "instrument": options.instrument,
        "source_difficulty": "Expert",
        "target_difficulty": options.target_difficulty,
        "record_count": len(records),
        "task_view": task_view,
    }


def _catalog_records_path(catalog_manifest_path: Path) -> Path:
    raw = json.loads(catalog_manifest_path.read_text(encoding="utf-8"))
    records = raw.get("records")
    if not isinstance(records, str):
        raise PreparationError("validated catalog manifest has no records path")
    return (catalog_manifest_path.parent / records).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
