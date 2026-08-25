"""Shared runtime admission checks for five-lane audio-to-chart task views.

Catalog coverage is intentionally lightweight.  These checks are the stricter
boundary used by training and held-out profile evaluation: an admitted asset
must fully decode and its exact chart track must contain an Expert five-lane
event.  They return aggregate-safe reason codes; callers never serialize a
path or source identity into a public worker result.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from typing import Any

import mido
import soundfile as sf

RUNTIME_ADMISSION_FORMAT = "strum-five-lane-runtime-admission/v1"
RUNTIME_ADMISSION = {"format": RUNTIME_ADMISSION_FORMAT}
# Catalog stems are bounded song assets; two seconds leaves room for a normal
# full-stream decode while preventing several wedged native decoders from
# making a catalog inspection unresponsive.
FULL_STREAM_AUDIO_DECODE_TIMEOUT_SECONDS = 2.0
FULL_STREAM_AUDIO_DECODE_CLEANUP_SECONDS = 0.1

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


def _full_stream_decode_child(
    audio_path: str, receive_connection: Any, send_connection: Any
) -> None:
    """Decode in an isolated process and send only a boolean result.

    Some native decoders can terminate a process rather than raising a Python
    exception for malformed input.  No path or decoder detail crosses this
    boundary, so callers can publish only the established aggregate reason.
    """
    receive_connection.close()
    try:
        audio, _sample_rate = sf.read(audio_path, dtype="float32")
        send_connection.send(bool(getattr(audio, "size", 0) >= 1))
    except BaseException:
        try:
            send_connection.send(False)
        except BaseException:
            pass
    finally:
        send_connection.close()


def _process_is_alive(process: Any) -> bool:
    """Return process liveness without turning cleanup failures into admission failures."""
    try:
        return bool(process.is_alive())
    except (AssertionError, OSError, ValueError):
        return False


def _stop_decode_process(process: Any) -> None:
    """Bound cleanup even when a native child ignores termination signals."""
    if not _process_is_alive(process):
        return
    try:
        process.terminate()
    except (OSError, ValueError):
        return
    try:
        process.join(FULL_STREAM_AUDIO_DECODE_CLEANUP_SECONDS)
    except (OSError, ValueError):
        return
    if not _process_is_alive(process):
        return
    kill = getattr(process, "kill", None)
    if not callable(kill):
        return
    try:
        kill()
        process.join(FULL_STREAM_AUDIO_DECODE_CLEANUP_SECONDS)
    except (OSError, ValueError):
        return


def _full_stream_audio_decodes(
    audio_path: Path,
    *,
    timeout_seconds: float = FULL_STREAM_AUDIO_DECODE_TIMEOUT_SECONDS,
    child_target: Callable[[str, Any, Any], None] = _full_stream_decode_child,
    process_context: Any | None = None,
) -> bool:
    """Return whether a bounded child process fully decodes one audio asset."""
    if timeout_seconds <= 0:
        raise ValueError("audio decode timeout must be positive")
    # Do not fork a process that already imported libsndfile: native decoder
    # locks can be inherited in a permanently-held state.  Forkserver starts
    # from a clean interpreter and avoids that state; spawn is the portable
    # fallback where forkserver is unavailable.
    if process_context is None:
        try:
            context = mp.get_context("forkserver")
        except ValueError:
            context = mp.get_context("spawn")
    else:
        context = process_context
    receive_connection, send_connection = context.Pipe(duplex=False)
    process: Any | None = None
    cleanup_attempted = False
    try:
        process = context.Process(
            target=child_target,
            args=(str(audio_path), receive_connection, send_connection),
        )
        process.start()
        send_connection.close()
        process.join(timeout_seconds)
        if _process_is_alive(process):
            _stop_decode_process(process)
            cleanup_attempted = True
            return False
        if process.exitcode != 0 or not receive_connection.poll():
            return False
        try:
            return receive_connection.recv() is True
        except (EOFError, OSError):
            return False
    except (OSError, RuntimeError):
        return False
    finally:
        for connection in (send_connection, receive_connection):
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            if not cleanup_attempted:
                _stop_decode_process(process)
            if not _process_is_alive(process):
                try:
                    process.close()
                except (OSError, ValueError):
                    pass


def classify_five_lane_runtime_source(
    audio_path: Path, midi_path: Path, *, label_track: str
) -> str | None:
    """Return an aggregate-safe exclusion code, or ``None`` when usable.

    ``sf.read`` deliberately consumes the whole stream.  Header inspection is
    insufficient: a corrupt OPUS may open successfully and fail later, which
    would otherwise let training silently skip an item that evaluation rejects.
    """
    if not _full_stream_audio_decodes(audio_path):
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
