#!/usr/bin/env python3
"""Prepare catalog-approved ``PART VOCALS`` labels for a bounded experiment.

This is intentionally not a replacement for :mod:`scripts.vocals_charter`.
It derives only frame-level lead-vocal activity and sung-pitch targets from
managed ``PART VOCALS`` MIDI tracks.  Phrase-marker and lyric-event counts are
retained in the cache metadata so a later lyric/phrase model can be evaluated
against the same immutable catalog task view.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mido
import numpy as np
import soundfile as sf
import torch
import torchaudio

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.catalog_task_manifest import (  # noqa: E402
    MANIFEST_FORMAT,
    resolve_catalog_task_manifest_songs,
)

SAMPLE_RATE = 22_050
N_MELS = 128
N_FFT = 2_048
HOP_LENGTH = 512
SEGMENT_SECONDS = 5.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH) + 1
SEGMENT_HOP_FRAMES = SEGMENT_FRAMES // 2
VOCAL_MIN_MIDI = 36
VOCAL_MAX_MIDI = 84
PITCH_CLASS_COUNT = VOCAL_MAX_MIDI - VOCAL_MIN_MIDI + 2  # 0 is unvoiced.


class VocalsPreprocessError(ValueError):
    """Raised when a task view cannot safely yield Vocal frame targets."""


def _tempo_map(midi: mido.MidiFile) -> list[tuple[int, int]]:
    """Return absolute-tick tempo changes, preserving the final value at a tick."""
    changes: list[tuple[int, int]] = [(0, 500_000)]
    for track in midi.tracks:
        tick = 0
        for message in track:
            tick += message.time
            if message.type == "set_tempo":
                changes.append((tick, message.tempo))
    result: list[tuple[int, int]] = []
    for tick, tempo in sorted(changes):
        if result and result[-1][0] == tick:
            result[-1] = (tick, tempo)
        else:
            result.append((tick, tempo))
    return result


def _seconds_at_tick(tick: int, changes: list[tuple[int, int]], ticks_per_beat: int) -> float:
    elapsed = 0.0
    previous_tick, tempo = changes[0]
    for change_tick, change_tempo in changes[1:]:
        if tick <= change_tick:
            return elapsed + (tick - previous_tick) * tempo / ticks_per_beat / 1_000_000
        elapsed += (change_tick - previous_tick) * tempo / ticks_per_beat / 1_000_000
        previous_tick, tempo = change_tick, change_tempo
    return elapsed + (tick - previous_tick) * tempo / ticks_per_beat / 1_000_000


def parse_vocal_events(midi_path: Path, *, label_track: str = "PART VOCALS") -> dict[str, object]:
    """Parse lead vocal notes plus phrase/lyric metadata from exactly one track."""
    try:
        midi = mido.MidiFile(midi_path)
    except (OSError, ValueError, EOFError) as error:
        raise VocalsPreprocessError("Vocal notes MIDI is unreadable") from error
    track = next((candidate for candidate in midi.tracks if candidate.name == label_track), None)
    if track is None:
        raise VocalsPreprocessError("PART VOCALS label track is missing")
    changes = _tempo_map(midi)
    active: dict[int, list[float]] = {}
    notes: list[dict[str, float | int]] = []
    lyric_events = 0
    phrase_markers = 0
    tick = 0
    for message in track:
        tick += message.time
        at_seconds = _seconds_at_tick(tick, changes, midi.ticks_per_beat)
        if message.type in {"lyrics", "text"} and getattr(message, "text", "").strip():
            lyric_events += 1
        if message.type == "note_on" and message.velocity > 0:
            if message.note in {105, 106}:
                phrase_markers += 1
            elif VOCAL_MIN_MIDI <= message.note <= VOCAL_MAX_MIDI:
                active.setdefault(message.note, []).append(at_seconds)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            starts = active.get(message.note)
            if starts:
                start = starts.pop(0)
                if at_seconds > start:
                    notes.append({"start": start, "end": at_seconds, "pitch": message.note})
    # Broken source charts occasionally omit note-offs.  Do not invent their
    # end times: fail them out of the target corpus rather than leaking a
    # silently broad label into training.
    notes.sort(key=lambda item: (float(item["start"]), int(item["pitch"])))
    return {
        "notes": notes,
        "lyric_event_count": lyric_events,
        "phrase_marker_count": phrase_markers,
    }


def _load_audio(path: Path) -> np.ndarray:
    try:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=False)
    except (OSError, RuntimeError) as error:
        raise VocalsPreprocessError("Vocal audio is unreadable") from error
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not len(audio):
        raise VocalsPreprocessError("Vocal audio is empty")
    if source_rate != SAMPLE_RATE:
        source = torch.from_numpy(np.asarray(audio)).unsqueeze(0)
        audio = torchaudio.functional.resample(source, source_rate, SAMPLE_RATE).squeeze(0).numpy()
    return np.asarray(audio, dtype=np.float32)


def _log_mel(audio: np.ndarray) -> np.ndarray:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=50.0,
        f_max=8_000.0,
        power=2.0,
    )(torch.from_numpy(audio).unsqueeze(0)).squeeze(0)
    return torch.log(mel + 1e-8).numpy().astype(np.float32)


def _labels_for_frames(
    notes: list[dict[str, float | int]], frames: int
) -> tuple[np.ndarray, np.ndarray]:
    activity = np.zeros(frames, dtype=np.uint8)
    pitch = np.zeros(frames, dtype=np.uint8)
    for item in notes:
        start = max(0, int(np.floor(float(item["start"]) * SAMPLE_RATE / HOP_LENGTH)))
        end = min(
            frames, max(start + 1, int(np.ceil(float(item["end"]) * SAMPLE_RATE / HOP_LENGTH)))
        )
        activity[start:end] = 1
        pitch[start:end] = int(item["pitch"]) - VOCAL_MIN_MIDI + 1
    return activity, pitch


def _segments(
    mel: np.ndarray, activity: np.ndarray, pitch: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    starts = range(0, max(mel.shape[1] - 1, 1), SEGMENT_HOP_FRAMES)
    for start in starts:
        if start >= mel.shape[1]:
            break
        end = min(start + SEGMENT_FRAMES, mel.shape[1])
        segment_mel = np.full((N_MELS, SEGMENT_FRAMES), float(mel.min()), dtype=np.float32)
        segment_activity = np.zeros(SEGMENT_FRAMES, dtype=np.uint8)
        segment_pitch = np.zeros(SEGMENT_FRAMES, dtype=np.uint8)
        width = end - start
        segment_mel[:, :width] = mel[:, start:end]
        segment_activity[:width] = activity[start:end]
        segment_pitch[:width] = pitch[start:end]
        output.append((segment_mel, segment_activity, segment_pitch))
        if end == mel.shape[1]:
            break
    return output


def prepare_vocals_frames(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...],
    limit_songs: int,
) -> dict[str, object]:
    """Resolve approved catalog records and materialize portable local caches."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise VocalsPreprocessError("Vocal task view is unreadable") from error
    task = manifest.get("task") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != MANIFEST_FORMAT
        or not isinstance(task, dict)
        or task.get("kind") != "vocals_activity"
        or task.get("pipeline_id") != "vocals.note-activity/v1"
        or task.get("instrument") != "vocals"
        or task.get("label_schema", {}).get("track_prefixes") != ["PART VOCALS"]
    ):
        raise VocalsPreprocessError(
            "Vocal preprocessing requires the Vocal activity catalog task view"
        )
    songs = resolve_catalog_task_manifest_songs(manifest, catalog_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"schema_version": 1, "splits": {}}
    for split in splits:
        selected = [song for song in songs if song["split"] == split]
        if limit_songs:
            selected = selected[:limit_songs]
        frame_segments: list[np.ndarray] = []
        activities: list[np.ndarray] = []
        pitches: list[np.ndarray] = []
        song_metadata: list[dict[str, object]] = []
        for song in selected:
            if song.get("label_tracks") != ["PART VOCALS"]:
                raise VocalsPreprocessError("Vocal task view selected an undeclared MIDI track")
            events = parse_vocal_events(Path(str(song["midi_path"])))
            notes = events["notes"]
            assert isinstance(notes, list)
            if not notes:
                continue
            mel = _log_mel(_load_audio(Path(str(song["audio_path"]))))
            activity, pitch = _labels_for_frames(notes, mel.shape[1])
            song_segments = _segments(mel, activity, pitch)
            frame_segments.extend(item[0] for item in song_segments)
            activities.extend(item[1] for item in song_segments)
            pitches.extend(item[2] for item in song_segments)
            song_metadata.append(
                {
                    "source_id": song["source_id"],
                    "segment_count": len(song_segments),
                    "note_count": len(notes),
                    "lyric_event_count": events["lyric_event_count"],
                    "phrase_marker_count": events["phrase_marker_count"],
                }
            )
        if not frame_segments:
            raise VocalsPreprocessError(f"Vocal {split} split has no usable pitched vocal notes")
        np.save(cache_dir / f"{split}_mel.npy", np.stack(frame_segments).astype(np.float16))
        np.save(cache_dir / f"{split}_activity.npy", np.stack(activities))
        np.save(cache_dir / f"{split}_pitch.npy", np.stack(pitches))
        summary["splits"][split] = {
            "song_count": len(song_metadata),
            "segment_count": len(frame_segments),
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
        prepare_vocals_frames(
            manifest_path=args.manifest,
            catalog_root=args.catalog_root,
            cache_dir=args.cache_dir,
            splits=tuple(args.splits),
            limit_songs=args.limit_songs,
        )
    except (VocalsPreprocessError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
