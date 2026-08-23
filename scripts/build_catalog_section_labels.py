#!/usr/bin/env python3
"""Derive path-free section labels from a catalog-backed section task view."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import soundfile as sf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_section_labels import HOP_S, WINDOW_S, label_window  # noqa: E402
from scripts.preprocess_guitar_windows import parse_onsets_from_manifest  # noqa: E402
from src.catalog_task_manifest import resolve_catalog_task_manifest_songs  # noqa: E402


def _manifest_hash(manifest: object) -> str:
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _parse_chart_onsets(midi_path: Path, label_track: str) -> list[tuple[float, frozenset[int]]]:
    """Return Expert five-lane events from exactly one task-view track.

    The legacy :class:`GuitarParser` can fall back to another guitar-shaped
    MIDI track when its preferred track is absent.  That is useful for an
    interactive legacy importer, but it would silently change the label
    source of a catalog worker.  Reuse the five-lane worker parser instead:
    it requires the exact named track and uses the same 25 ms chord grouping
    as the Guitar/Bass worker data path.
    """
    events = parse_onsets_from_manifest(midi_path, label_track=label_track)
    return [(time_ms, frozenset(frets)) for time_ms, frets in events]


def build_labels(manifest: dict[str, object], catalog_root: Path) -> dict[str, object]:
    """Create labels from MIDI, retaining source IDs but no local asset paths."""
    task = manifest.get("task")
    if not isinstance(task, dict) or task.get("kind") not in {"section_guitar", "section_bass"}:
        raise ValueError("catalog manifest must use section_guitar or section_bass")
    expected_track = "PART GUITAR" if task["instrument"] == "guitar" else "PART BASS"
    records: list[dict[str, object]] = []
    for song in resolve_catalog_task_manifest_songs(manifest, catalog_root):
        # A section label is derived from one five-lane performance stream.
        # Multiple matching tracks can be alternate arrangements, not a
        # union.  Do not invent union semantics while making a dataset.
        if song.get("label_tracks") != [expected_track]:
            raise ValueError(
                "section labels require exactly one declared PART GUITAR or PART BASS track"
            )
        onsets = _parse_chart_onsets(Path(song["midi_path"]), expected_track)
        if not onsets:
            continue
        duration = sf.info(song["audio_path"]).duration
        t = 0.0
        while t + WINDOW_S <= duration:
            in_window = [frets for time_ms, frets in onsets if t <= time_ms / 1000.0 < t + WINDOW_S]
            label, features = label_window(in_window, WINDOW_S)
            records.append(
                {
                    "source_id": song["source_id"],
                    "t_start_s": round(t, 3),
                    "t_end_s": round(t + WINDOW_S, 3),
                    "label": label,
                    "features": features,
                    "split": song["split"],
                }
            )
            t += HOP_S
    return {
        "schema_version": 1,
        "format": "strum-section-labels/v1",
        "lineage": {
            "task_manifest_sha256": _manifest_hash(manifest),
            "pipeline_id": task["pipeline_id"],
            "catalog_id": manifest["lineage"]["catalog_id"],
        },
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive STRUM section labels from an OCTAVE catalog task."
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--catalog-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = build_labels(manifest, args.catalog_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": args.out.name, "record_count": len(labels["records"])}))


if __name__ == "__main__":
    main()
