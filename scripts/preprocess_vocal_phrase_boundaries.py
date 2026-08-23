#!/usr/bin/env python3
"""Prepare approved ``PART VOCALS`` phrase-boundary labels for STRUM.

This bounded dataset path learns only lead-vocal phrase starts and ends from
the chart's established 105/106-or-105-span marker convention.  It does not
infer lyric text, talkies, harmony tracks, or a playable Vocal chart.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.preprocess_vocals_frames import (  # noqa: E402
    HOP_LENGTH,
    SAMPLE_RATE,
    _load_audio,
    _log_mel,
    _segments,
    parse_vocal_events,
)
from src.catalog_task_manifest import (  # noqa: E402
    MANIFEST_FORMAT,
    resolve_catalog_task_manifest_songs,
)

PIPELINE_ID = "vocals.phrase-boundaries/v1"
TASK_KIND = "vocals_phrase_boundaries"
PREPROCESSING_ID = "vocals-logmel-phrase-boundaries/v1"
_EXPECTED_LABEL_SCHEMA = {
    "id": "vocals-pitch-phrase-lyrics-midi/v1",
    "track_names": ["PART VOCALS"],
    "difficulty_encoding": "vocal-phrase-boundary-events/v1",
}
BOUNDARY_TOLERANCE_FRAMES = 1


class VocalPhrasePreprocessError(ValueError):
    """Raised when a catalog task cannot yield safe phrase-boundary labels."""


def _boundary_labels(
    starts: list[float], ends: list[float], frames: int
) -> tuple[np.ndarray, np.ndarray]:
    """Make narrow frame neighborhoods around observed marker boundaries."""
    start_targets = np.zeros(frames, dtype=np.uint8)
    end_targets = np.zeros(frames, dtype=np.uint8)
    for values, targets in ((starts, start_targets), (ends, end_targets)):
        for seconds in values:
            center = int(round(seconds * SAMPLE_RATE / HOP_LENGTH))
            low = max(0, center - BOUNDARY_TOLERANCE_FRAMES)
            high = min(frames, center + BOUNDARY_TOLERANCE_FRAMES + 1)
            targets[low:high] = 1
    return start_targets, end_targets


def prepare_vocal_phrase_boundaries(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...],
    limit_songs: int,
) -> dict[str, object]:
    """Revalidate one approved catalog and write local, path-free caches."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalPhrasePreprocessError("Vocal phrase task view is unreadable") from error
    task = manifest.get("task") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != TASK_KIND
        or task.get("pipeline_id") != PIPELINE_ID
        or task.get("instrument") != "vocals"
        or task.get("label_schema") != _EXPECTED_LABEL_SCHEMA
    ):
        raise VocalPhrasePreprocessError(
            "Vocal phrase preprocessing requires the phrase-boundary catalog task view"
        )
    songs = resolve_catalog_task_manifest_songs(manifest, catalog_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "preprocessing": PREPROCESSING_ID,
        "splits": {},
    }
    for split in splits:
        selected = [song for song in songs if song["split"] == split]
        if limit_songs:
            selected = selected[:limit_songs]
        frames: list[np.ndarray] = []
        starts: list[np.ndarray] = []
        ends: list[np.ndarray] = []
        song_metadata: list[dict[str, object]] = []
        for song in selected:
            if song.get("label_tracks") != ["PART VOCALS"]:
                raise VocalPhrasePreprocessError(
                    "Vocal phrase task view selected an undeclared MIDI track"
                )
            events = parse_vocal_events(Path(str(song["midi_path"])))
            phrase_starts = events["phrase_start_events"]
            phrase_ends = events["phrase_end_events"]
            if not isinstance(phrase_starts, list) or not isinstance(phrase_ends, list):
                raise VocalPhrasePreprocessError("Vocal phrase markers are invalid")
            if not phrase_starts or not phrase_ends:
                continue
            mel = _log_mel(_load_audio(Path(str(song["audio_path"]))))
            start_targets, end_targets = _boundary_labels(
                [float(value) for value in phrase_starts],
                [float(value) for value in phrase_ends],
                mel.shape[1],
            )
            song_segments = _segments(mel, start_targets, end_targets)
            frames.extend(item[0] for item in song_segments)
            starts.extend(item[1] for item in song_segments)
            ends.extend(item[2] for item in song_segments)
            song_metadata.append(
                {
                    "source_id": song["source_id"],
                    "segment_count": len(song_segments),
                    "phrase_start_count": len(phrase_starts),
                    "phrase_end_count": len(phrase_ends),
                }
            )
        if not frames:
            raise VocalPhrasePreprocessError(
                f"Vocal phrase {split} split has no usable phrase boundaries"
            )
        np.save(cache_dir / f"{split}_mel.npy", np.stack(frames).astype(np.float16))
        np.save(cache_dir / f"{split}_phrase_start.npy", np.stack(starts))
        np.save(cache_dir / f"{split}_phrase_end.npy", np.stack(ends))
        summary["splits"][split] = {
            "song_count": len(song_metadata),
            "segment_count": len(frames),
            "songs": song_metadata,
        }
    (cache_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--catalog-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--limit-songs", type=int, default=0)
    args = parser.parse_args()
    if args.limit_songs < 0:
        parser.error("--limit-songs must be non-negative")
    try:
        prepare_vocal_phrase_boundaries(
            manifest_path=args.manifest,
            catalog_root=args.catalog_root,
            cache_dir=args.cache_dir,
            splits=tuple(args.splits),
            limit_songs=args.limit_songs,
        )
    except (VocalPhrasePreprocessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
