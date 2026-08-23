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
PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID = "pro-event-proposal-asymmetric-window-exclusion/v1"
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
    before: int,
    after: int,
    local_exclusion: int,
    seed: int,
) -> list[int]:
    """Return deterministic centers whose complete feature windows are negative.

    A proposal feature is asymmetric: ``[center - before, center + after]``.
    Therefore a candidate center that appears *before* a REAL onset can still
    expose that onset in its right-hand context.  Exclusion is deliberately
    asymmetric too: for every REAL onset, reject centers in the inclusive
    interval ``[-max(after, local), +max(before, local)]``.  This enforces
    both the configured local tolerance and the stronger invariant that no
    negative feature window contains a REAL event onset.
    """
    if before < 0 or after < 0 or local_exclusion < 0:
        raise ProEventProposalPreprocessError("Pro proposal negative geometry is invalid")
    excluded_before_event = max(after, local_exclusion)
    excluded_after_event = max(before, local_exclusion)
    allowed = [
        frame
        for frame in range(frame_count)
        if all(
            frame < positive - excluded_before_event or frame > positive + excluded_after_event
            for positive in positives
        )
    ]
    ranked = sorted(
        allowed,
        key=lambda frame: hashlib.sha256(f"{seed}:{source_id}:{frame}".encode()).digest(),
    )
    return ranked[:count]


def canonical_pro_event_proposal_negative_policy(
    *,
    audio_features: Mapping[str, object],
    requested_options: Mapping[str, object] | None,
) -> dict[str, object]:
    """Build the one permitted path-free effective negative-sampling policy.

    The proposal bundle must preserve this complete policy, rather than just
    its identifier.  Keeping the constructor beside the cache prevents a
    trainer and a bundle validator from independently reimplementing its
    feature-window arithmetic.
    """
    try:
        settings = normalize_pro_audio_preprocessing(audio_features)
    except CatalogValidationError as error:
        raise ProEventProposalPreprocessError("Pro proposal audio features are invalid") from error
    options = normalize_pro_event_proposal_options(requested_options)
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
    local_exclusion = math.ceil(
        options["negative_exclusion_ms"]
        * int(settings["sample_rate"])
        / (1000 * int(settings["hop_length"]))
    )
    return {
        "id": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
        "selection": "deterministic-sha256-ranked-audio-frames/v1",
        "requested_options": dict(options),
        "feature_window": {
            "center": "candidate_center_frame",
            "window_before_ms": settings["window_before_ms"],
            "window_after_ms": settings["window_after_ms"],
            "window_before_frames": before,
            "window_after_frames": after,
        },
        "real_event_exclusion": {
            "rule": "negative-feature-window-must-not-contain-real-event-onset/v1",
            "local_onset_exclusion_frames": local_exclusion,
            # For a REAL event at frame E, a negative center N is rejected
            # whenever E + minimum <= N <= E + maximum.  The larger right
            # feature context is intentionally reflected by the longer
            # negative offset on the earlier side of an event.
            "invalid_negative_center_offset_frames": {
                "minimum": -max(after, local_exclusion),
                "maximum": max(before, local_exclusion),
                "inclusive": True,
            },
        },
    }


def validate_pro_event_proposal_negative_policy(
    raw: object, *, audio_features: Mapping[str, object]
) -> dict[str, object]:
    """Require a serialized policy to be the exact cache-produced policy.

    This is deliberately equality-based rather than a permissive schema
    check: no caller may omit the selection method, truncate requested
    options, or weaken the feature-window/exclusion geometry in a portable
    candidate configuration.
    """
    if not isinstance(raw, Mapping):
        raise ProEventProposalPreprocessError("Pro proposal negative policy is invalid")
    requested_options = raw.get("requested_options")
    if not isinstance(requested_options, Mapping):
        raise ProEventProposalPreprocessError("Pro proposal negative policy is invalid")
    canonical = canonical_pro_event_proposal_negative_policy(
        audio_features=audio_features,
        requested_options=requested_options,
    )
    if dict(raw) != canonical:
        raise ProEventProposalPreprocessError("Pro proposal negative policy is not canonical")
    return canonical


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
    negative_policy = canonical_pro_event_proposal_negative_policy(
        audio_features=settings,
        requested_options=options,
    )
    before = int(negative_policy["feature_window"]["window_before_frames"])
    after = int(negative_policy["feature_window"]["window_after_frames"])
    local_exclusion = int(negative_policy["real_event_exclusion"]["local_onset_exclusion_frames"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "schema_version": 1,
        "format": PRO_EVENT_PROPOSAL_CACHE_FORMAT,
        "preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "audio_features": settings,
            "negative_policy": negative_policy,
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
                before=before,
                after=after,
                local_exclusion=local_exclusion,
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
