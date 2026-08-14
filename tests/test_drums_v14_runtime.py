from types import SimpleNamespace

import pytest
import torch

from src.inference import drums_v14_runtime
from src.inference.drums_v14_runtime import (
    DrumsV14Runtime,
    DrumsV14RuntimeError,
    _select_peak_frames,
)


def _profile() -> SimpleNamespace:
    return SimpleNamespace(
        onset_threshold=0.5,
        class_thresholds=(0.5,) * 8,
        min_distance_ms=20.0,
        segment_duration_seconds=10.0,
        overlap=0.5,
        class_to_midi=(96, 97, 98, 98, 99, 99, 100, 100),
    )


class _OutputModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        frames = x.shape[-1]
        onset = torch.zeros((1, frames, 1), device=x.device)
        classes = torch.zeros((1, frames, 8), device=x.device)
        velocities = torch.zeros((1, frames, 8), device=x.device)
        onset[0, 2, 0] = 0.9
        classes[0, 2, 0] = 0.8  # kick
        classes[0, 2, 2] = 0.7  # hi-hat, lane 2
        classes[0, 2, 3] = 0.9  # high tom, same lane; wins deterministically
        velocities[0, 2, 0] = 0.5
        velocities[0, 2, 3] = 0.9
        return {"onset_probs": onset, "class_probs": classes, "velocities": velocities}


def test_direct_tensor_runtime_emits_safe_expert_events() -> None:
    runtime = DrumsV14Runtime(_profile(), _OutputModel(), torch.device("cpu"))

    events = runtime.transcribe_tensor(torch.zeros(44_100), 44_100)

    assert [(event.lane, event.midi_note, event.velocity) for event in events] == [
        (0, 96, 64),
        (2, 98, 114),
    ]
    assert all(event.time_ms >= 0 for event in events)


def test_peak_selection_is_deterministic_and_respects_distance() -> None:
    peaks = _select_peak_frames(
        torch.tensor([0.0, 0.8, 0.7, 0.9, 0.0]), threshold=0.5, min_distance_frames=3
    )

    assert peaks == [3]


def test_loader_requires_a_strict_state_dict(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    seen: dict[str, object] = {}

    class _LoadedModel(torch.nn.Module):
        def __init__(self, **parameters: object) -> None:
            super().__init__()
            seen["parameters"] = parameters

        def load_state_dict(self, state_dict, strict: bool = True):
            seen["state_dict"] = state_dict
            seen["strict"] = strict
            return torch.nn.modules.module._IncompatibleKeys([], [])

    monkeypatch.setattr(drums_v14_runtime, "TwoStageDrumsCRNN", _LoadedModel)
    monkeypatch.setattr(
        drums_v14_runtime.torch,
        "load",
        lambda *args, **kwargs: {"model_state_dict": {"weight": torch.tensor([1.0])}},
    )
    parameters = {
        "n_mels": 128,
        "conv_channels": [64, 128, 256, 512],
        "freq_subbands": [32, 64, 96, 128],
        "subband_proj_dim": 256,
        "lstm_hidden": 640,
        "lstm_layers": 3,
        "attention_heads": 10,
        "attention_type": "flash",
        "attention_window": 512,
        "dropout": 0.0,
        "onset_detector_hidden": 320,
        "classifier_hidden": 640,
        "num_classes": 8,
        "predict_velocity": True,
    }

    DrumsV14Runtime.from_profile(
        _profile(), checkpoint_path=tmp_path / "v14.pt", model_parameters=parameters, device="cpu"
    )

    assert seen["strict"] is True
    assert seen["parameters"] == parameters


def test_loader_rejects_partial_or_legacy_model_settings(tmp_path) -> None:
    with pytest.raises(DrumsV14RuntimeError, match="complete and exact"):
        DrumsV14Runtime.from_profile(
            _profile(), checkpoint_path=tmp_path / "v14.pt", model_parameters={}, device="cpu"
        )
