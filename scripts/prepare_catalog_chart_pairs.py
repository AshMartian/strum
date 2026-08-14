#!/usr/bin/env python3
"""Create a catalog-backed five-lane chart-transform task view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.prepare_guitar_chart_pairs import FIVE_LANE_INSTRUMENT_TRACKS, PreparationError
except ModuleNotFoundError:  # Support ``python scripts/prepare_catalog_chart_pairs.py``.
    from prepare_guitar_chart_pairs import FIVE_LANE_INSTRUMENT_TRACKS, PreparationError
from src.catalog_chart_pairs import (
    CatalogChartPairOptions,
    pipeline_descriptor,
    prepare_catalog_chart_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a STRUM chart-transform task view from an OCTAVE catalog."
    )
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--instrument", choices=sorted(FIVE_LANE_INSTRUMENT_TRACKS))
    parser.add_argument("--target-difficulty", choices=["Hard", "Medium", "Easy"])
    parser.add_argument("--dataset-id")
    parser.add_argument("--split-seed", type=int, default=20260814)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--describe-pipeline",
        action="store_true",
        help="print the pipeline contract instead of creating a task view",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.describe_pipeline:
        print(json.dumps(pipeline_descriptor(), indent=2, sort_keys=True))
        return
    if not all((args.catalog_root, args.output_dir, args.instrument, args.target_difficulty)):
        raise SystemExit(
            "--catalog-root, --output-dir, --instrument, and --target-difficulty are required"
        )
    try:
        result = prepare_catalog_chart_pairs(
            args.catalog_root,
            args.output_dir,
            CatalogChartPairOptions(
                instrument=args.instrument,
                target_difficulty=args.target_difficulty,
                split_seed=args.split_seed,
                validation_fraction=args.validation_fraction,
                dataset_id=args.dataset_id,
            ),
            overwrite=args.overwrite,
        )
    except PreparationError as error:
        raise SystemExit(str(error)) from error
    print(
        json.dumps(
            {
                "manifest_path": str(result["manifest_path"]),
                "record_count": result["record_count"],
                "skipped": result["skipped"],
                "task_view_id": result["task_view_id"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
