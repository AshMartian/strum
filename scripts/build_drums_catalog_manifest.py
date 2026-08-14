#!/usr/bin/env python3
"""CLI entry point for OCTAVE catalog-backed Drums task manifests."""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog_drums_manifest import main  # noqa: E402

if __name__ == "__main__":
    main()
