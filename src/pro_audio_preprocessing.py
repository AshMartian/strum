"""Catalog-revalidated audio features for exact Pro MIDI target events.

This is deliberately not the five-lane window cache with a different name.
Each cached example retains the exact Pro target track and its semantic label
language: standard versus ``_22`` string/fret/technique events, or chromatic
Pro Keys pitch/channel/range-shift events.  The cache is useful for research
trainers, but it is *not* an auto-chart model or a deployable profile.
"""

from __future__ import annotations

import bisect
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mido
import numpy as np

from src.pro_target_manifest import (
    PRO_TARGET_MANIFEST_FORMAT,
    normalize_pro_audio_preprocessing,
    resolve_catalog_pro_target_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError


class ProAudioPreprocessError(ValueError):
    """Raised when a Pro target view cannot yield an auditable feature cache."""


@dataclass(frozen=True)
class _TempoSegment:
    tick: int
    seconds: float
    tempo: int


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProAudioPreprocessError("Pro target task view is unreadable") from error
    if not isinstance(raw, dict) or raw.get("format") != PRO_TARGET_MANIFEST_FORMAT:
        raise ProAudioPreprocessError("task view is not a Pro target manifest")
    return raw


def _tempo_segments(midi_path: Path) -> tuple[int, tuple[_TempoSegment, ...]]:
    """Read a whole-song tempo map, including tempo changes outside REAL_* tracks."""
    try:
        midi = mido.MidiFile(midi_path)
    except (EOFError, OSError, ValueError) as error:
        raise ProAudioPreprocessError("Pro target MIDI asset is unreadable") from error
    tempo_by_tick: dict[int, int] = {0: 500_000}
    tick = 0
    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.type == "set_tempo":
            tempo_by_tick[tick] = message.tempo
    current_tick = 0
    current_seconds = 0.0
    current_tempo = tempo_by_tick[0]
    segments = [_TempoSegment(0, 0.0, current_tempo)]
    for change_tick, change_tempo in sorted(tempo_by_tick.items()):
        if change_tick == 0:
            continue
        current_seconds += mido.tick2second(
            change_tick - current_tick, midi.ticks_per_beat, current_tempo
        )
        current_tick = change_tick
        current_tempo = change_tempo
        segments.append(_TempoSegment(current_tick, current_seconds, current_tempo))
    return midi.ticks_per_beat, tuple(segments)


def _tick_seconds(tick: int, ticks_per_beat: int, segments: Sequence[_TempoSegment]) -> float:
    if tick < 0:
        raise ProAudioPreprocessError("Pro target event has an invalid tick")
    if not segments:
        raise ProAudioPreprocessError("Pro target MIDI has no tempo map")
    position = bisect.bisect_right([segment.tick for segment in segments], tick) - 1
    segment = segments[max(position, 0)]
    return segment.seconds + mido.tick2second(tick - segment.tick, ticks_per_beat, segment.tempo)


def _load_log_mel(audio_path: Path, settings: Mapping[str, object]) -> np.ndarray:
    """Decode one approved asset into deterministic, mono log-mel features."""
    try:
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415

        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except Exception as error:  # soundfile errors vary by installed codec support.
        raise ProAudioPreprocessError("Pro target audio asset is unreadable") from error
    if audio.size == 0:
        raise ProAudioPreprocessError("Pro target audio asset is empty")
    mono = audio.mean(axis=1, dtype=np.float32)
    target_rate = int(settings["sample_rate"])
    samples = torch.from_numpy(mono).unsqueeze(0)
    if sample_rate != target_rate:
        samples = torchaudio.functional.resample(samples, sample_rate, target_rate)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_rate,
        n_fft=int(settings["n_fft"]),
        hop_length=int(settings["hop_length"]),
        n_mels=int(settings["n_mels"]),
        f_min=30.0,
        f_max=min(8000.0, target_rate / 2),
        power=2.0,
    )
    with torch.no_grad():
        return torch.log(transform(samples).squeeze(0) + 1e-8).cpu().numpy()


def _event_windows(
    targets: Sequence[object],
    *,
    task_kind: str,
    source_id: str,
    split: str,
    midi_path: Path,
    mel: np.ndarray,
    settings: Mapping[str, object],
) -> list[tuple[np.ndarray, dict[str, object]]]:
    """Create fixed windows while retaining the complete exact label payload."""
    if task_kind not in {"pro_guitar", "pro_bass", "pro_keys"}:
        raise ProAudioPreprocessError("Pro target task kind is invalid")
    ticks_per_beat, tempos = _tempo_segments(midi_path)
    sample_rate = int(settings["sample_rate"])
    hop_length = int(settings["hop_length"])
    before = round(int(settings["window_before_ms"]) * sample_rate / (1000 * hop_length))
    after = round(int(settings["window_after_ms"]) * sample_rate / (1000 * hop_length))
    window_frames = before + after + 1
    examples: list[tuple[np.ndarray, dict[str, object]]] = []
    for target_track in targets:
        if not isinstance(target_track, dict):
            raise ProAudioPreprocessError("Pro target record has an invalid target track")
        track_name = target_track.get("track_name")
        events = target_track.get("events")
        if not isinstance(track_name, str) or not isinstance(events, list):
            raise ProAudioPreprocessError("Pro target record has invalid target events")
        by_tick: dict[int, list[dict[str, object]]] = defaultdict(list)
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("tick"), int):
                raise ProAudioPreprocessError("Pro target event is invalid")
            by_tick[event["tick"]].append(event)
        for tick, simultaneous in sorted(by_tick.items()):
            seconds = _tick_seconds(tick, ticks_per_beat, tempos)
            center = round(seconds * sample_rate / hop_length)
            # A target at/after the decoded feature tail has no evidence and
            # must never become a padded all-zero positive sample.
            if center >= mel.shape[1]:
                continue
            feature = np.zeros((mel.shape[0], window_frames), dtype=np.float32)
            source_start = max(0, center - before)
            source_end = min(mel.shape[1], center + after + 1)
            destination_start = source_start - (center - before)
            feature[:, destination_start : destination_start + source_end - source_start] = mel[
                :, source_start:source_end
            ]
            label: dict[str, object] = {
                "source_id": source_id,
                "split": split,
                "target_track": track_name,
                "event_tick": tick,
                "event_schema": target_track.get("event_schema"),
                "events": simultaneous,
            }
            if task_kind in {"pro_guitar", "pro_bass"}:
                variant = target_track.get("track_variant")
                if variant not in {"standard", "22_fret"}:
                    raise ProAudioPreprocessError("Pro string target lost its track variant")
                label["track_variant"] = variant
                label["target_language"] = "string_fret_technique/v1"
            else:
                shifts = target_track.get("range_shifts")
                if not isinstance(shifts, list):
                    raise ProAudioPreprocessError("Pro Keys target lost its range shifts")
                label["range_shifts"] = shifts
                label["target_language"] = "pitch_channel_range_shift/v1"
            examples.append((feature, label))
    return examples


def _safe_source_lineage(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    task_view = manifest.get("task_view")
    if not isinstance(task_view, dict) or not isinstance(task_view.get("songs"), list):
        raise ProAudioPreprocessError("Pro target manifest task view is invalid")
    result: dict[str, dict[str, object]] = {}
    for song in task_view["songs"]:
        if not isinstance(song, dict) or not isinstance(song.get("source_id"), str):
            raise ProAudioPreprocessError("Pro target manifest task source is invalid")
        audio = song.get("audio")
        midi = song.get("notes_midi")
        if not isinstance(audio, dict) or not isinstance(midi, dict):
            raise ProAudioPreprocessError("Pro target manifest asset lineage is invalid")
        result[song["source_id"]] = {
            "audio_sha256": audio.get("sha256"),
            "audio_role": song.get("audio_role"),
            "notes_midi_sha256": midi.get("sha256"),
        }
    return result


def prepare_pro_audio_windows(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    limit_songs: int = 0,
) -> dict[str, object]:
    """Materialize a private, path-free event cache from an immutable Pro view."""
    if not splits or any(split not in {"train", "val", "test"} for split in splits):
        raise ProAudioPreprocessError("Pro audio preprocessing has invalid splits")
    if limit_songs < 0:
        raise ProAudioPreprocessError("Pro audio preprocessing limit_songs is invalid")
    if cache_dir.exists() and (not cache_dir.is_dir() or any(cache_dir.iterdir())):
        raise ProAudioPreprocessError("Pro audio cache directory must be empty")
    manifest = _load_manifest(manifest_path)
    task_view = manifest["task_view"]
    assert isinstance(task_view, dict)
    task = task_view.get("task")
    task_kind = task.get("kind") if isinstance(task, dict) else None
    if task_kind not in {"pro_guitar", "pro_bass", "pro_keys"}:
        raise ProAudioPreprocessError("Pro target manifest task kind is invalid")
    settings = normalize_pro_audio_preprocessing(manifest.get("audio_preprocessing"))
    try:
        resolved = resolve_catalog_pro_target_manifest_songs(manifest, catalog_root)
    except CatalogValidationError:
        raise
    except (TypeError, ValueError) as error:
        raise ProAudioPreprocessError("Pro target manifest cannot be revalidated") from error
    safe_lineage = _safe_source_lineage(manifest)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "format": "strum-pro-audio-feature-cache/v1",
        "preprocessing": settings,
        "task_kind": task_kind,
        "splits": {},
    }
    for split in splits:
        selected = [song for song in resolved if song.get("split") == split]
        if limit_songs:
            selected = selected[:limit_songs]
        features: list[np.ndarray] = []
        labels: list[dict[str, object]] = []
        skipped: Counter[str] = Counter()
        for song in selected:
            source_id = song.get("source_id")
            if not isinstance(source_id, str):
                raise ProAudioPreprocessError("resolved Pro target source is invalid")
            try:
                mel = _load_log_mel(Path(str(song["audio_path"])), settings)
                examples = _event_windows(
                    song.get("targets", []),
                    task_kind=task_kind,
                    source_id=source_id,
                    split=split,
                    midi_path=Path(str(song["midi_path"])),
                    mel=mel,
                    settings=settings,
                )
            except ProAudioPreprocessError as error:
                skipped[str(error)] += 1
                continue
            if not examples:
                skipped["Pro target audio ends before every target event"] += 1
                continue
            source_lineage = safe_lineage.get(source_id)
            if source_lineage is None:
                raise ProAudioPreprocessError("Pro target source lineage is missing")
            for feature, label in examples:
                labels.append({**label, "source": source_lineage})
                features.append(feature)
        if not features:
            raise ProAudioPreprocessError(f"Pro audio {split} split has no usable target windows")
        feature_path = cache_dir / f"{split}_logmel.npy"
        label_path = cache_dir / f"{split}_targets.jsonl"
        np.save(feature_path, np.stack(features).astype(np.float16))
        label_path.write_text(
            "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels), encoding="utf-8"
        )
        summary["splits"][split] = {
            "feature_name": feature_path.name,
            "target_name": label_path.name,
            "song_count": len({label["source_id"] for label in labels}),
            "event_window_count": len(labels),
            "skipped_source_counts": dict(sorted(skipped.items())),
        }
    (cache_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
