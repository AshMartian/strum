from __future__ import annotations

import os
import time
from pathlib import Path

from src import five_lane_runtime_admission as admission


def _terminate_child(*_args: object) -> None:
    os._exit(23)


def _stall_child(*_args: object) -> None:
    time.sleep(1)


class _Connection:
    def close(self) -> None:
        return None

    def poll(self) -> bool:
        return False


class _SignalResistantProcess:
    exitcode = None

    def __init__(self) -> None:
        self.terminate_calls = 0
        self.kill_calls = 0
        self.close_calls = 0

    def start(self) -> None:
        return None

    def join(self, _timeout: float) -> None:
        return None

    def is_alive(self) -> bool:
        return True

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    def close(self) -> None:
        self.close_calls += 1
        raise AssertionError("an active process must never be closed")


class _SignalResistantContext:
    def __init__(self) -> None:
        self.process = _SignalResistantProcess()

    def Pipe(self, *, duplex: bool) -> tuple[_Connection, _Connection]:
        assert duplex is False
        return _Connection(), _Connection()

    def Process(self, **_kwargs: object) -> _SignalResistantProcess:
        return self.process


def test_isolated_decode_maps_child_termination_to_false(tmp_path: Path) -> None:
    assert (
        admission._full_stream_audio_decodes(
            tmp_path / "decoder-crash.ogg", timeout_seconds=1, child_target=_terminate_child
        )
        is False
    )


def test_isolated_decode_maps_child_timeout_to_false(tmp_path: Path) -> None:
    assert (
        admission._full_stream_audio_decodes(
            tmp_path / "decoder-timeout.ogg", timeout_seconds=0.01, child_target=_stall_child
        )
        is False
    )


def test_isolated_decode_bounds_sigterm_and_sigkill_resistant_child(tmp_path: Path) -> None:
    context = _SignalResistantContext()
    started = time.monotonic()

    assert (
        admission._full_stream_audio_decodes(
            tmp_path / "signal-resistant.ogg",
            timeout_seconds=0.01,
            process_context=context,
        )
        is False
    )

    assert time.monotonic() - started < 0.5
    assert context.process.terminate_calls == 1
    assert context.process.kill_calls == 1
    assert context.process.close_calls == 0


def test_runtime_admission_maps_isolated_decode_failure_to_unreadable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(admission, "_full_stream_audio_decodes", lambda _path: False)

    assert (
        admission.classify_five_lane_runtime_source(
            tmp_path / "decoder-failure.ogg",
            tmp_path / "unused.mid",
            label_track="PART GUITAR",
        )
        == "runtime_audio_unreadable"
    )
