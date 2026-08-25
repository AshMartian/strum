"""Shared runtime admission checks for five-lane audio-to-chart task views.

Catalog coverage is intentionally lightweight.  These checks are the stricter
boundary used by training and held-out profile evaluation: an admitted asset
must fully decode and its exact chart track must contain an Expert five-lane
event.  They return aggregate-safe reason codes; callers never serialize a
path or source identity into a public worker result.
"""

from __future__ import annotations

from pathlib import Path

import mido
import soundfile as sf

RUNTIME_ADMISSION_FORMAT = "strum-five-lane-runtime-admission/v1"
RUNTIME_ADMISSION = {"format": RUNTIME_ADMISSION_FORMAT}


def classify_five_lane_runtime_source(
    audio_path: Path, midi_path: Path, *, label_track: str
) -> str | None:
    """Return an aggregate-safe exclusion code, or ``None`` when usable.

    ``sf.read`` deliberately consumes the whole stream.  Header inspection is
    insufficient: a corrupt OPUS may open successfully and fail later, which
    would otherwise let training silently skip an item that evaluation rejects.
    """
    try:
        audio, _sample_rate = sf.read(str(audio_path), dtype="float32")
    except Exception:
        return "runtime_audio_unreadable"
    if getattr(audio, "size", 0) < 1:
        return "runtime_audio_unreadable"
    try:
        midi = mido.MidiFile(str(midi_path))
    except Exception:
        return "exact_expert_label_missing"
    track = next((item for item in midi.tracks if item.name == label_track), None)
    if track is None:
        return "exact_expert_label_missing"
    if not any(
        message.type == "note_on"
        and message.velocity > 0
        and message.note in {95, 96, 97, 98, 99, 100}
        for message in track
    ):
        return "exact_expert_label_missing"
    return None
