#!/usr/bin/env python3
"""Materialize exact-Pro catalog audio windows for research training only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.pro_audio_preprocessing import (  # noqa: E402
    ProAudioPreprocessError,
    prepare_pro_audio_windows,
)
from src.song_source_catalog import CatalogValidationError  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit-songs", type=int, default=0)
    args = parser.parse_args()
    try:
        result = prepare_pro_audio_windows(
            manifest_path=args.manifest,
            catalog_root=args.catalog_root,
            cache_dir=args.cache_dir,
            splits=tuple(args.splits),
            limit_songs=args.limit_songs,
        )
    except (CatalogValidationError, ProAudioPreprocessError) as error:
        parser.error(str(error))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
