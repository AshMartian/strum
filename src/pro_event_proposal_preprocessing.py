"""Catalog-revalidated audio windows for a bounded Pro event-proposal candidate.

The exact-REAL_* target manifest remains the only label authority.  Unlike
``pro_audio_preprocessing``, this cache contains deterministic negative audio
windows as well as windows centred on authored event times.  At inference the
candidate can score arbitrary offline audio windows; it does *not* decode a
Pro sequence, choose string/fret attributes, or write MIDI.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.pro_audio_preprocessing import (
    ProAudioPreprocessError,
    _load_log_mel,
    _load_manifest,
    _safe_source_lineage,
    _tempo_segments,
    _tick_seconds,
)
from src.pro_target_manifest import PRO_TARGET_MANIFEST_FORMAT, normalize_pro_audio_preprocessing
from src.song_source_catalog import CatalogValidationError

PRO_EVENT_PROPOSAL_CACHE_FORMAT = "strum-pro-event-proposal-feature-cache/v1"
PRO_EVENT_PROPOSAL_PREPROCESSING_ID = "pro-logmel-event-proposal-windows/v1"
_MAX_NEGATIVE_RATIO = 32
_MAX_NEGATIVE_EXCLUSION_MS = 5000


class ProEventProposalPreprocessError(ValueError):
    """Raised when a Pro proposal cache cannot prove its label boundary."""


def normalize_pro_event_proposal_options(raw: Mapping[str, object] | None) -> dict[str, int]:
    """Validate bounded negative sampling settings with no path-bearing input."""
    requested = {} if raw is None else dict(raw)
    permitted = {"negative_ratio", "negative_exclusion_ms", "negative_seed"}
    if set(requested) - permitted:
        raise ProEventProposalPreprocessError("Pro event proposal options are unsupported")
    defaults = {
        "negative_ratio": 4,
        "negative_exclusion_ms": 80,
        "negative_seed": 20260822,
    }
    defaults.update(requested)
    for key, value in defaults.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ProEventProposalPreprocessError(f"Pro event proposal {key} is invalid")
    if not 1 <= defaults["negative_ratio"] <= _MAX_NEGATIVE_RATIO:
        raise ProEventProposalPreprocessError("Pro event proposal negative_ratio is invalid")
    if defaults["negative_exclusion_ms"] > _MAX_NEGATIVE_EXCLUSION_MS:
        raise ProEventProposalPreprocessError("Pro event proposal exclusion window is too large")
    return defaults


def _window(mel: np.ndarray, center: int, before: int, after: int) -> np.ndarray:
    if not 0 <= center < mel.shape[1]:
        raise ProEventProposalPreprocessError("Pro proposal center is outside audio features")
    result = np.zeros((mel.shape[0], before + after + 1), dtype=np.float32)
    source_start = max(0, center - before)
    source_end = min(mel.shape[1], center + after + 1)
    destination_start = source_start - (center - before)
    result[:, destination_start : destination_start + source_end - source_start] = mel[
        :, source_start:source_end
    ]
    return result


def _negative_centers(
    *,
    source_id: str,
    frame_count: int,
    positives: Sequence[int],
    count: int,
    exclusion: int,
    seed: int,
) -> list[int]:
    """Return deterministic non-event centers, excluding the local onset tolerance."""
    allowed = [
        frame
        for frame in range(frame_count)
        if all(abs(frame - positive) > exclusion for positive in positives)
    ]
    ranked = sorted(
        allowed,
        key=lambda frame: hashlib.sha256(f"{seed}:{source_id}:{frame}".encode()).digest(),
    )
    return ranked[:count]


def _positive_events(
    targets: Sequence[object],
    *,
    ticks_per_beat: int,
    tempos: Sequence[object],
    settings: Mapping[str, object],
    frame_count: int,
) -> dict[int, list[dict[str, object]]]:
    sample_rate = int(settings["sample_rate"])
    hop_length = int(settings["hop_length"])
    by_center: dict[int, list[dict[str, object]]] = defaultdict(list)
    for target in targets:
        if not isinstance(target, dict):
            raise ProEventProposalPreprocessError("Pro proposal target track is invalid")
        track_name = target.get("track_name")
        events = target.get("events")
        if not isinstance(track_name, str) or not isinstance(events, list):
            raise ProEventProposalPreprocessError("Pro proposal target track events are invalid")
        for event in events:
            if not isinstance(event, dict) or not isinstance(event.get("tick"), int):
                raise ProEventProposalPreprocessError("Pro proposal target event is invalid")
            seconds = _tick_seconds(event["tick"], ticks_per_beat, tempos)  # type: ignore[arg-type]
            center = round(seconds * sample_rate / hop_length)
            if not 0 <= center < frame_count:
                continue
            by_center[center].append(
                {
                    "track_name": track_name,
                    "track_variant": target.get("track_variant"),
                    "event_schema": target.get("event_schema"),
                    "event": event,
                }
            )
    return dict(sorted(by_center.items()))


def prepare_pro_event_proposal_windows(
    *,
    manifest_path: Path,
    catalog_root: Path,
    cache_dir: Path,
    splits: tuple[str, ...] = ("train", "val"),
    limit_songs: int = 0,
    negative_ratio: int = 4,
    negative_exclusion_ms: int = 80,
    negative_seed: int = 20260822,
) -> dict[str, object]:
    """Materialize a private, audio-only event-proposal training cache.

    Catalog and exact target revalidation happens before any managed path is
    read.  The returned cache/summary persists IDs and hashes only.
    """
    from src.pro_target_manifest import resolve_catalog_pro_target_manifest_songs  # noqa: PLC0415

    options = normalize_pro_event_proposal_options(
        {
            "negative_ratio": negative_ratio,
            "negative_exclusion_ms": negative_exclusion_ms,
            "negative_seed": negative_seed,
        }
    )
    if not splits or any(split not in {"train", "val", "test"} for split in splits):
        raise ProEventProposalPreprocessError("Pro event proposal splits are invalid")
    if limit_songs < 0:
        raise ProEventProposalPreprocessError("Pro event proposal limit_songs is invalid")
    if cache_dir.exists() and (not cache_dir.is_dir() or any(cache_dir.iterdir())):
        raise ProEventProposalPreprocessError("Pro event proposal cache directory must be empty")
    try:
        manifest = _load_manifest(manifest_path)
        resolved = resolve_catalog_pro_target_manifest_songs(manifest, catalog_root)
        settings = normalize_pro_audio_preprocessing(manifest.get("audio_preprocessing"))
    except (CatalogValidationError, ProAudioPreprocessError) as error:
        raise ProEventProposalPreprocessError(
            "Pro proposal target view cannot be revalidated"
        ) from error
    if manifest.get("format") != PRO_TARGET_MANIFEST_FORMAT:
        raise ProEventProposalPreprocessError("Pro proposal target view format is invalid")
    safe_lineage = _safe_source_lineage(manifest)
    before = round(
        int(settings["window_before_ms"])
        * int(settings["sample_rate"])
        / (1000 * int(settings["hop_length"]))
    )
    after = round(
        int(settings["window_after_ms"])
        * int(settings["sample_rate"])
        / (1000 * int(settings["hop_length"]))
    )
    exclusion = math.ceil(
        options["negative_exclusion_ms"]
        * int(settings["sample_rate"])
        / (1000 * int(settings["hop_length"]))
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "format": PRO_EVENT_PROPOSAL_CACHE_FORMAT,
        "preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "audio_features": settings,
            "negative_sampling": options,
        },
        "task_kind": manifest["task_view"]["task"]["kind"],
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
                raise ProEventProposalPreprocessError("resolved Pro proposal source is invalid")
            try:
                mel = _load_log_mel(Path(str(song["audio_path"])), settings)
                ticks_per_beat, tempos = _tempo_segments(Path(str(song["midi_path"])))
                positives = _positive_events(
                    song.get("targets", []),
                    ticks_per_beat=ticks_per_beat,
                    tempos=tempos,
                    settings=settings,
                    frame_count=mel.shape[1],
                )
            except (ProAudioPreprocessError, ProEventProposalPreprocessError) as error:
                skipped[str(error)] += 1
                continue
            if not positives:
                skipped["Pro proposal audio ends before every target event"] += 1
                continue
            lineage = safe_lineage.get(source_id)
            if lineage is None:
                raise ProEventProposalPreprocessError("Pro proposal source lineage is missing")
            negative_count = len(positives) * options["negative_ratio"]
            negatives = _negative_centers(
                source_id=source_id,
                frame_count=mel.shape[1],
                positives=list(positives),
                count=negative_count,
                exclusion=exclusion,
                seed=options["negative_seed"],
            )
            if not negatives:
                skipped["Pro proposal source has no negative audio frames"] += 1
                continue
            for center, targets in positives.items():
                features.append(_window(mel, center, before, after))
                labels.append(
                    {
                        "source_id": source_id,
                        "split": split,
                        "center_frame": center,
                        "is_event": True,
                        "exact_real_targets": targets,
                        "source": lineage,
                    }
                )
            for center in negatives:
                features.append(_window(mel, center, before, after))
                labels.append(
                    {
                        "source_id": source_id,
                        "split": split,
                        "center_frame": center,
                        "is_event": False,
                        "source": lineage,
                    }
                )
        positives_count = sum(label["is_event"] is True for label in labels)
        negatives_count = sum(label["is_event"] is False for label in labels)
        if not features or not positives_count or not negatives_count:
            raise ProEventProposalPreprocessError(
                f"Pro proposal {split} split requires positive and negative audio windows"
            )
        feature_path = cache_dir / f"{split}_logmel.npy"
        target_path = cache_dir / f"{split}_proposal_targets.jsonl"
        np.save(feature_path, np.stack(features).astype(np.float16))
        target_path.write_text(
            "".join(json.dumps(label, sort_keys=True) + "\n" for label in labels), encoding="utf-8"
        )
        summary["splits"][split] = {
            "feature_name": feature_path.name,
            "target_name": target_path.name,
            "song_count": len({label["source_id"] for label in labels}),
            "positive_window_count": positives_count,
            "negative_window_count": negatives_count,
            "skipped_source_counts": dict(sorted(skipped.items())),
        }
    (cache_dir / "preprocess_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary
