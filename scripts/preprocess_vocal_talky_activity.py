#!/usr/bin/env python3
"""Prepare observed lead-Vocal pitchless/talky span targets from catalog MIDI.

The sole label authority is note 96 on exact ``PART VOCALS``.  The cache is
local; task views and packaged artifacts retain catalog-safe identity only.
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
    N_MELS,
    SAMPLE_RATE,
    SEGMENT_FRAMES,
    _load_audio,
    _log_mel,
    parse_vocal_events,
)
from src.catalog_task_manifest import (  # noqa: E402
    MANIFEST_FORMAT,
    resolve_catalog_task_manifest_songs,
)

PIPELINE_ID = "vocals.talky-activity/v1"
TASK_KIND = "vocals_talky_activity"
PREPROCESSING_ID = "vocals-logmel-pitchless-talky-note-96-spans/v1"
_EXPECTED_LABEL_SCHEMA = {
    "id": "vocals-pitch-phrase-lyrics-midi/v1",
    "track_names": ["PART VOCALS"],
    "difficulty_encoding": "vocal-pitchless-talky-note-96-spans/v1",
}


class VocalTalkyPreprocessError(ValueError):
    """Raised when a task view cannot yield truthful talky targets."""


def _labels_for_frames(spans: list[dict[str, float]], frames: int) -> np.ndarray:
    labels = np.zeros(frames, dtype=np.uint8)
    for span in spans:
        start, end = span.get("start"), span.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise VocalTalkyPreprocessError("Vocal talky span is invalid")
        low = max(0, int(np.floor(float(start) * SAMPLE_RATE / HOP_LENGTH)))
        high = min(frames, max(low + 1, int(np.ceil(float(end) * SAMPLE_RATE / HOP_LENGTH))))
        labels[low:high] = 1
    return labels


def _segments(mel: np.ndarray, labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(0, mel.shape[1], SEGMENT_FRAMES // 2):
        if start >= mel.shape[1]:
            break
        end = min(start + SEGMENT_FRAMES, mel.shape[1])
        segment = np.full((N_MELS, SEGMENT_FRAMES), float(mel.min()), dtype=np.float32)
        target = np.zeros(SEGMENT_FRAMES, dtype=np.uint8)
        segment[:, : end - start] = mel[:, start:end]
        target[: end - start] = labels[start:end]
        output.append((segment, target))
        if end == mel.shape[1]:
            break
    return output


def prepare_vocal_talky_activity(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...],
    limit_songs: int,
) -> dict[str, object]:
    """Revalidate the catalog task and materialize local note-96 span labels."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalTalkyPreprocessError("Vocal talky task view is unreadable") from error
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
        raise VocalTalkyPreprocessError(
            "Vocal talky preprocessing requires the talky-activity catalog task view"
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
        targets: list[np.ndarray] = []
        metadata: list[dict[str, object]] = []
        for song in selected:
            if song.get("label_tracks") != ["PART VOCALS"]:
                raise VocalTalkyPreprocessError(
                    "Vocal talky task selected an undeclared MIDI track"
                )
            events = parse_vocal_events(Path(str(song["midi_path"])))
            spans = events.get("talky_spans")
            if not isinstance(spans, list):
                raise VocalTalkyPreprocessError("Vocal talky events are invalid")
            # Retain negative songs: a classifier needs non-talky context, but
            # the trainer separately enforces positives in every held-out split.
            mel = _log_mel(_load_audio(Path(str(song["audio_path"]))))
            labels = _labels_for_frames(spans, mel.shape[1])
            song_segments = _segments(mel, labels)
            frames.extend(item[0] for item in song_segments)
            targets.extend(item[1] for item in song_segments)
            metadata.append(
                {
                    "source_id": song["source_id"],
                    "segment_count": len(song_segments),
                    "talky_span_count": len(spans),
                }
            )
        if not frames:
            raise VocalTalkyPreprocessError(f"Vocal talky {split} split has no usable audio")
        np.save(cache_dir / f"{split}_mel.npy", np.stack(frames).astype(np.float16))
        np.save(cache_dir / f"{split}_talky.npy", np.stack(targets))
        summary["splits"][split] = {
            "song_count": len(metadata),
            "segment_count": len(frames),
            "talky_span_count": sum(int(song["talky_span_count"]) for song in metadata),
            "songs": metadata,
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
        prepare_vocal_talky_activity(
            manifest_path=args.manifest,
            catalog_root=args.catalog_root,
            cache_dir=args.cache_dir,
            splits=tuple(args.splits),
            limit_songs=args.limit_songs,
        )
    except (VocalTalkyPreprocessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
