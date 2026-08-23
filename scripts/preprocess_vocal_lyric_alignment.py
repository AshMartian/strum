#!/usr/bin/env python3
"""Prepare approved lead-Vocal lyric CTC targets from ``PART VOCALS``.

The chart is the sole lyric authority: each target comes from a ``lyrics`` or
``text`` meta event on the exact lead track.  This is deliberately separate
from pitch, phrase, talky, harmony, and chart-composition training.  Caches
are local implementation details; the task view and packaged experiment keep
only catalog-safe identity and aggregate lineage.
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
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

PIPELINE_ID = "vocals.lyric-alignment/v1"
TASK_KIND = "vocals_lyric_alignment"
PREPROCESSING_ID = "vocals-logmel-observed-lyric-ctc/v1"
TOKENIZER_ID = "vocal-lyric-character-tokenizer/v1"
# Index zero is CTC's required blank.  The remaining tokens are deliberately
# compact and deterministic.  Unknown Unicode is represented explicitly,
# rather than silently discarded from an approved chart label.
VOCABULARY = (
    "<blank>",
    "<space>",
    "<unknown>",
    *tuple("abcdefghijklmnopqrstuvwxyz0123456789'-.+/#^"),
)
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
MAX_TARGET_TOKENS = 128
_EXPECTED_LABEL_SCHEMA = {
    "id": "vocals-pitch-phrase-lyrics-midi/v1",
    "track_names": ["PART VOCALS"],
    "difficulty_encoding": "vocal-lyric-meta-events/v1",
}


class VocalLyricPreprocessError(ValueError):
    """Raised when an immutable Vocal lyric task cannot yield CTC data."""


def tokenize_lyric(text: str) -> list[int]:
    """Encode chart lyric text without applying language-model corrections."""
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    tokens: list[int] = []
    for character in normalized:
        if character.isspace():
            token = "<space>"
        elif character in TOKEN_TO_ID:
            token = character
        else:
            token = "<unknown>"
        tokens.append(TOKEN_TO_ID[token])
    return tokens


def _segments_for_song(
    mel: np.ndarray, lyric_events: list[dict[str, object]]
) -> list[tuple[np.ndarray, list[int], list[int]]]:
    """Partition one song into non-overlapping, timestamped CTC windows.

    Event timestamps stay beside targets so a future forced-alignment/eval
    stage can compare timing without recovering any original path.  Windows
    with no lyric label are omitted: blank-only CTC examples carry no text
    supervision and would dominate a compact curated corpus.
    """
    output: list[tuple[np.ndarray, list[int], list[int]]] = []
    for start in range(0, mel.shape[1], SEGMENT_FRAMES):
        end = min(start + SEGMENT_FRAMES, mel.shape[1])
        tokens: list[int] = []
        event_frames: list[int] = []
        for event in lyric_events:
            time = event.get("time")
            text = event.get("text")
            if not isinstance(time, (int, float)) or not isinstance(text, str):
                raise VocalLyricPreprocessError("Vocal lyric event is invalid")
            frame = int(round(float(time) * SAMPLE_RATE / HOP_LENGTH))
            if start <= frame < end:
                encoded = tokenize_lyric(text)
                if encoded:
                    tokens.extend(encoded)
                    event_frames.append(frame - start)
        if not tokens:
            continue
        if len(tokens) > MAX_TARGET_TOKENS:
            # Preserve target integrity: no truncation or split-point guessing
            # is permitted for source lyrics in this first bounded component.
            continue
        segment = np.full((N_MELS, SEGMENT_FRAMES), float(mel.min()), dtype=np.float32)
        segment[:, : end - start] = mel[:, start:end]
        output.append((segment, tokens, event_frames))
    return output


def prepare_vocal_lyric_alignment(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...],
    limit_songs: int,
) -> dict[str, object]:
    """Revalidate a catalog task view and materialize local CTC caches."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalLyricPreprocessError("Vocal lyric task view is unreadable") from error
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
        raise VocalLyricPreprocessError(
            "Vocal lyric preprocessing requires the lyric-alignment catalog task view"
        )
    songs = resolve_catalog_task_manifest_songs(manifest, catalog_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "preprocessing": PREPROCESSING_ID,
        "tokenizer": {"id": TOKENIZER_ID, "vocabulary_size": len(VOCABULARY)},
        "splits": {},
    }
    for split in splits:
        selected = [song for song in songs if song["split"] == split]
        if limit_songs:
            selected = selected[:limit_songs]
        frames: list[np.ndarray] = []
        targets: list[list[int]] = []
        event_positions: list[list[int]] = []
        song_metadata: list[dict[str, object]] = []
        skipped_overlong = 0
        for song in selected:
            if song.get("label_tracks") != ["PART VOCALS"]:
                raise VocalLyricPreprocessError(
                    "Vocal lyric task view selected an undeclared MIDI track"
                )
            events = parse_vocal_events(Path(str(song["midi_path"])))
            lyric_events = events.get("lyric_events")
            if not isinstance(lyric_events, list):
                raise VocalLyricPreprocessError("Vocal lyric events are invalid")
            mel = _log_mel(_load_audio(Path(str(song["audio_path"]))))
            song_segments = _segments_for_song(mel, lyric_events)
            # An observed lyric too long for the bounded CTC target is reported
            # in aggregate rather than shortened into a false label.
            raw_token_count = sum(
                len(tokenize_lyric(str(event.get("text", "")))) for event in lyric_events
            )
            if lyric_events and not song_segments and raw_token_count > MAX_TARGET_TOKENS:
                skipped_overlong += 1
                continue
            if not song_segments:
                continue
            frames.extend(item[0] for item in song_segments)
            targets.extend(item[1] for item in song_segments)
            event_positions.extend(item[2] for item in song_segments)
            song_metadata.append(
                {
                    "source_id": song["source_id"],
                    "segment_count": len(song_segments),
                    "lyric_event_count": len(lyric_events),
                    "target_token_count": sum(len(item[1]) for item in song_segments),
                }
            )
        if not frames:
            raise VocalLyricPreprocessError(
                f"Vocal lyric {split} split has no usable observed lyric events"
            )
        max_tokens = max(len(item) for item in targets)
        target_matrix = np.zeros((len(targets), max_tokens), dtype=np.int16)
        target_lengths = np.zeros(len(targets), dtype=np.int16)
        max_events = max(len(item) for item in event_positions)
        event_matrix = np.full((len(event_positions), max_events), -1, dtype=np.int16)
        for index, (target, positions) in enumerate(zip(targets, event_positions, strict=True)):
            target_matrix[index, : len(target)] = target
            target_lengths[index] = len(target)
            event_matrix[index, : len(positions)] = positions
        np.save(cache_dir / f"{split}_mel.npy", np.stack(frames).astype(np.float16))
        np.save(cache_dir / f"{split}_targets.npy", target_matrix)
        np.save(cache_dir / f"{split}_target_lengths.npy", target_lengths)
        np.save(cache_dir / f"{split}_event_frames.npy", event_matrix)
        summary["splits"][split] = {
            "song_count": len(song_metadata),
            "segment_count": len(frames),
            "skipped_overlong_song_count": skipped_overlong,
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
        prepare_vocal_lyric_alignment(
            manifest_path=args.manifest,
            catalog_root=args.catalog_root,
            cache_dir=args.cache_dir,
            splits=tuple(args.splits),
            limit_songs=args.limit_songs,
        )
    except (VocalLyricPreprocessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
