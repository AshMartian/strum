"""Full-stream decoder compatibility checks for catalog-backed Vocal datasets.

Every implemented lead-Vocal preprocessor uses :mod:`soundfile` to decode an
approved audio asset before deriving log-mel features.  Catalog import proves
content identity, but an asset can still be a format that libsndfile cannot
decode.  This module keeps that implementation boundary in one place so a
task view can reject incompatible audio before a multi-stage training job
starts.

The check streams the complete asset in bounded chunks.  Content hashing
remains the catalog's integrity check; completing the same libsndfile decode
that the preprocessor relies on prevents a corrupt middle section from
surviving catalog admission merely because its edge windows open correctly.
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf

VOCAL_AUDIO_COMPATIBILITY = "soundfile-full-stream-decode/v1"
_PROBE_FRAMES = 4_096


def has_compatible_vocal_audio(path: Path) -> bool:
    """Return whether STRUM's Vocal decoder can read a non-empty signal.

    Read the complete stream in fixed-size chunks without retaining the whole
    song.  This uses the preprocessor's decoder and float32 conversion while
    keeping preparation memory-bounded.  The boolean API prevents local paths
    or decoder details escaping into a portable task view.
    """
    try:
        with sf.SoundFile(path, mode="r") as source:
            frames = source.frames
            if frames <= 0 or source.samplerate <= 0 or source.channels <= 0:
                return False
            decoded_frames = 0
            while decoded_frames < frames:
                chunk = source.read(
                    min(_PROBE_FRAMES, frames - decoded_frames),
                    dtype="float32",
                    always_2d=False,
                )
                if chunk.size == 0:
                    return False
                decoded_frames += len(chunk)
    except (OSError, RuntimeError, ValueError):
        return False
    return True
