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

# A five-lane neural profile must learn from the isolated instrument stem it
# will be asked to interpret.  This is intentionally separate from runtime
# admission: a file can be fully decodable and have exact Expert labels while
# still being a mix fallback that is unsuitable for a profile-grade attempt.
# The minimums are source-disjoint split counts, not window/event counts.
PROFILE_GRADE_ADMISSION_FORMAT = "strum-five-lane-profile-audibility-admission/v1"
PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT = {"train": 20, "val": 5, "test": 5}
PROFILE_GRADE_ADMISSION = {
    "format": PROFILE_GRADE_ADMISSION_FORMAT,
    "audio_selection": "exact_dedicated_instrument_role",
    "source_disjoint_minimums": PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT,
    "runtime_admission": RUNTIME_ADMISSION_FORMAT,
}


def profile_grade_admission_is_met(by_split: dict[str, int]) -> bool:
    """Return whether aggregate source-disjoint coverage meets the contract."""
    return all(
        by_split.get(split, 0) >= minimum
        for split, minimum in PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT.items()
    )


def validate_profile_grade_audio_selection(
    *, instrument: str, audio_role: str, fallback_audio_role: str | None
) -> None:
    """Reject profile-grade selection that would admit a mix fallback."""
    if audio_role != instrument or fallback_audio_role is not None:
        raise ValueError(
            "profile-grade five-lane preparation requires the exact dedicated "
            "instrument audio role and no fallback audio role"
        )


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
