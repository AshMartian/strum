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

from scripts.prepare_guitar_chart_pairs import (
    DIFFICULTY_BASE_NOTES,
    FIVE_LANE_INSTRUMENT_TRACKS,
    PreparationError,
    parse_instrument_difficulties,
)
from src.song_source_catalog import CATALOG_FILENAME, CatalogRecord, SongSourceCatalog, load_catalog

PAIR_DATASET_FORMAT = "strum-chart-pairs/v1"
PIPELINE_ID = "chart_transform.five_lane"
PIPELINE_VERSION = 1
PREPROCESSING_ID = "midi-five-lane-events"
PREPROCESSING_VERSION = 1
SPLIT_ALGORITHM = "sha256-source-id-rank/v1"


@dataclass(frozen=True)
class CatalogChartPairOptions:
    """Stable configuration for one instrument and one difficulty transform."""

    instrument: str
    target_difficulty: str
    split_seed: int = 20260814
    validation_fraction: float = 0.2
    dataset_id: str | None = None

    def __post_init__(self) -> None:
        if self.instrument not in FIVE_LANE_INSTRUMENT_TRACKS:
            raise PreparationError(
                f"instrument must be one of: {', '.join(sorted(FIVE_LANE_INSTRUMENT_TRACKS))}"
            )
        if self.target_difficulty not in {"Hard", "Medium", "Easy"}:
            raise PreparationError("target_difficulty must be Hard, Medium, or Easy")
        if not isinstance(self.split_seed, int):
            raise PreparationError("split_seed must be an integer")
        if not 0 < self.validation_fraction < 1:
            raise PreparationError("validation_fraction must be between 0 and 1")
        if self.dataset_id is not None and not self.dataset_id.strip():
            raise PreparationError("dataset_id must be non-empty when provided")


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
        "split": {"algorithm": SPLIT_ALGORITHM, "unit": "catalog source_id"},
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
    if len(parsed_records) < 2:
        details = "; ".join(skipped) if skipped else "fewer than two eligible catalog records"
        raise PreparationError(
            "catalog task view requires at least two valid song records for song-level validation: "
            f"{details}"
        )

    split_assignments = _split_assignments(
        [record["source_id"] for record in parsed_records],
        options.split_seed,
        options.validation_fraction,
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


def _split_assignments(
    source_ids: list[str], seed: int, validation_fraction: float
) -> dict[str, str]:
    if len(set(source_ids)) != len(source_ids):
        raise PreparationError("catalog task view has duplicate source_id values")
    ordered = sorted(source_ids, key=lambda source_id: _split_key(source_id, seed))
    validation_count = min(len(ordered) - 1, max(1, round(len(ordered) * validation_fraction)))
    validation = set(ordered[:validation_count])
    return {
        source_id: "validation" if source_id in validation else "train" for source_id in source_ids
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
        {"source_id": record["source_id"], "notes_midi_sha256": record["notes_midi_sha256"]}
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
    task_view = {
        "pipeline": {"id": PIPELINE_ID, "version": PIPELINE_VERSION},
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "manifest_sha256": _sha256(catalog_manifest_path),
            "records_sha256": _sha256(records_path),
        },
        "source_inputs": source_inputs,
        "split": {
            "algorithm": SPLIT_ALGORITHM,
            "seed": options.split_seed,
            "validation_fraction": options.validation_fraction,
            "assignments": {
                source_id: split_assignments[source_id] for source_id in sorted(split_assignments)
            },
        },
        "preprocessing": {
            **preprocessing,
            "config_sha256": _canonical_sha256(preprocessing),
        },
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
