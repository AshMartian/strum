#!/usr/bin/env python3
"""CLI for a versioned, catalog-backed STRUM task view."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog_task_manifest import (  # noqa: E402
    available_task_kinds,
    build_catalog_task_manifest,
    write_catalog_task_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a STRUM task manifest from an OCTAVE catalog."
    )
    parser.add_argument("catalog", type=Path, help="OCTAVE catalog root")
    parser.add_argument("--task", choices=available_task_kinds(), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audio-role", help="preferred managed audio role")
    parser.add_argument(
        "--fallback-audio-role", help="fallback managed audio role; use none to disable"
    )
    parser.add_argument("--required-difficulty", default="expert")
    parser.add_argument("--split-seed", default="catalog-source-id/v1")
    parser.add_argument(
        "--preprocessing-json",
        default="{}",
        help="portable JSON preprocessing settings, persisted and fingerprinted",
    )
    args = parser.parse_args()
    try:
        preprocessing = json.loads(args.preprocessing_json)
    except json.JSONDecodeError as error:
        parser.error(f"--preprocessing-json must be JSON: {error.msg}")
    if not isinstance(preprocessing, dict):
        parser.error("--preprocessing-json must be an object")
    disable_fallback = args.fallback_audio_role == "none"
    manifest = build_catalog_task_manifest(
        args.catalog,
        args.task,
        audio_role=args.audio_role,
        fallback_audio_role=args.fallback_audio_role,
        disable_fallback=disable_fallback,
        required_difficulty=args.required_difficulty,
        split_seed=args.split_seed,
        preprocessing=preprocessing,
    )
    output = write_catalog_task_manifest(args.output, manifest)
    print(json.dumps({"output": output.name, "record_count": manifest["summary"]["record_count"]}))


if __name__ == "__main__":
    main()
