"""Direct, fail-closed Expert Drums V14 inference.

This module intentionally does not import ``batch_infer_hybrid``.  It consumes
the profile already validated by the model-bundle layer, one verified V14
checkpoint, and runtime-only managed audio.  It has no ensemble, environment
switches, postprocessing, or availability-based companion model fallbacks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torchaudio

from src.models.drums_v13 import TwoStageDrumsCRNN

if TYPE_CHECKING:
    from src.inference.drums_v14_profile import DrumsV14ExpertProfile


SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
CLASS_COUNT = 8
LOG_EPSILON = 1e-8
MODEL_PARAMETER_FIELDS = frozenset(
    {
        "n_mels",
        "conv_channels",
        "freq_subbands",
        "subband_proj_dim",
        "lstm_hidden",
        "lstm_layers",
        "attention_heads",
        "attention_type",
        "attention_window",
        "dropout",
        "onset_detector_hidden",
        "classifier_hidden",
        "num_classes",
        "predict_velocity",
    }
)


class DrumsV14RuntimeError(RuntimeError):
    """Raised when a verified profile cannot execute exactly as declared."""


@dataclass(frozen=True)
class DrumsV14Event:
    """One safe Expert Drums event; no source path or untrusted metadata."""

    time_ms: float
    lane: int
    midi_note: int
    velocity: int


class DrumsV14Runtime:
    """Tensor-only V14 execution configured exclusively by a typed profile."""

    def __init__(
        self,
        profile: DrumsV14ExpertProfile,
        model: torch.nn.Module,
        device: torch.device,
    ) -> None:
        self.profile = profile
        self.model = model.to(device).eval()
        self.device = device
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )

    @classmethod
    def from_profile(
        cls,
        profile: DrumsV14ExpertProfile,
        *,
        checkpoint_path: str | Path,
        model_parameters: Mapping[str, object],
        device: str | torch.device,
    ) -> DrumsV14Runtime:
        """Strictly load the one V14 checkpoint declared by the profile.

        The worker resolves and hash-verifies ``checkpoint_path`` before this
        call.  This method still verifies the state-dict key and requires an
        exact state-dict match so a wrong architecture cannot partially load.
        """
        parameters = _validated_model_parameters(model_parameters)
        target_device = torch.device(device)
        try:
            checkpoint = torch.load(
                Path(checkpoint_path), map_location=target_device, weights_only=True
            )
        except Exception as error:
            raise DrumsV14RuntimeError("unable to load verified V14 checkpoint") from error
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model_state_dict"), dict
        ):
            raise DrumsV14RuntimeError("V14 checkpoint must contain model_state_dict")
        model = TwoStageDrumsCRNN(**parameters)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise DrumsV14RuntimeError(
                "V14 checkpoint is incompatible with its declared model"
            ) from error
        return cls(profile, model, target_device)

    def transcribe_audio_file(self, audio_path: str | Path) -> list[DrumsV14Event]:
        """Load a managed catalog audio asset and return only Expert Drums events."""
        try:
            audio, sample_rate = torchaudio.load(str(audio_path))
        except Exception as error:
            raise DrumsV14RuntimeError("unable to read managed Drums audio") from error
        return self.transcribe_tensor(audio, sample_rate)

    @torch.inference_mode()
    def transcribe_tensor(self, audio: torch.Tensor, sample_rate: int) -> list[DrumsV14Event]:
        """Run V14 on an in-memory audio tensor with no mutable global state."""
        mono = _normalise_audio(audio, sample_rate)
        mel = torch.log(self._mel(mono.unsqueeze(0)) + LOG_EPSILON)
        if mel.shape[-1] == 0:
            raise DrumsV14RuntimeError("managed Drums audio has no frames")
        onset_probs, class_probs, velocities = self._predict(mel)
        events: list[DrumsV14Event] = []
        for frame in _select_peak_frames(
            onset_probs,
            threshold=self.profile.onset_threshold,
            min_distance_frames=max(
                1, round(self.profile.min_distance_ms / 1000 * SAMPLE_RATE / HOP_LENGTH)
            ),
        ):
            events.extend(self._events_for_frame(frame, class_probs[frame], velocities[frame]))
        return events

    def _predict(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_frames = mel.shape[-1]
        segment_frames = max(
            1, round(self.profile.segment_duration_seconds * SAMPLE_RATE / HOP_LENGTH)
        )
        hop_frames = max(1, round(segment_frames * (1 - self.profile.overlap)))
        starts = list(range(0, max(1, total_frames - segment_frames + 1), hop_frames))
        final_start = max(0, total_frames - segment_frames)
        if starts[-1] != final_start:
            starts.append(final_start)
        starts = sorted(set(starts))
        onset_sum = torch.zeros(total_frames, dtype=torch.float32)
        class_sum = torch.zeros((total_frames, CLASS_COUNT), dtype=torch.float32)
        velocity_sum = torch.zeros((total_frames, CLASS_COUNT), dtype=torch.float32)
        counts = torch.zeros(total_frames, dtype=torch.float32)
        for start in starts:
            end = min(start + segment_frames, total_frames)
            segment = mel[:, :, start:end]
            if segment.shape[-1] < segment_frames:
                segment = torch.nn.functional.pad(segment, (0, segment_frames - segment.shape[-1]))
            output = self.model(segment.unsqueeze(0).to(self.device))
            if not isinstance(output, dict) or not {
                "onset_probs",
                "class_probs",
                "velocities",
            } <= set(output):
                raise DrumsV14RuntimeError("V14 model does not expose the required direct outputs")
            actual = end - start
            onset_sum[start:end] += output["onset_probs"].squeeze(0).squeeze(-1).cpu()[:actual]
            class_sum[start:end] += output["class_probs"].squeeze(0).cpu()[:actual]
            velocity_sum[start:end] += output["velocities"].squeeze(0).cpu()[:actual]
            counts[start:end] += 1
        if torch.any(counts == 0):
            raise DrumsV14RuntimeError("V14 segment plan left uncovered audio frames")
        return (
            onset_sum / counts,
            class_sum / counts.unsqueeze(1),
            velocity_sum / counts.unsqueeze(1),
        )

    def _events_for_frame(
        self, frame: int, class_probs: torch.Tensor, velocities: torch.Tensor
    ) -> list[DrumsV14Event]:
        best_for_lane: dict[int, tuple[float, int]] = {}
        for class_index, threshold in enumerate(self.profile.class_thresholds):
            probability = float(class_probs[class_index])
            if probability < threshold:
                continue
            midi_note = self.profile.class_to_midi[class_index]
            lane = midi_note - 96
            current = best_for_lane.get(lane)
            if current is None or probability > current[0]:
                best_for_lane[lane] = (probability, class_index)
        time_ms = frame * HOP_LENGTH / SAMPLE_RATE * 1000
        events: list[DrumsV14Event] = []
        for lane, (_probability, class_index) in sorted(best_for_lane.items()):
            velocity = max(1, min(127, round(float(velocities[class_index]) * 127)))
            events.append(
                DrumsV14Event(
                    time_ms=time_ms,
                    lane=lane,
                    midi_note=self.profile.class_to_midi[class_index],
                    velocity=velocity,
                )
            )
        return events


def _validated_model_parameters(raw: Mapping[str, object]) -> dict[str, object]:
    if set(raw) != MODEL_PARAMETER_FIELDS:
        raise DrumsV14RuntimeError("V14 model parameters must be complete and exact")
    if (
        raw["n_mels"] != N_MELS
        or raw["num_classes"] != CLASS_COUNT
        or raw["predict_velocity"] is not True
        or raw["attention_type"] not in {"sliding", "flash"}
        or not isinstance(raw["conv_channels"], list)
        or not isinstance(raw["freq_subbands"], list)
        or not all(isinstance(value, int) and value > 0 for value in raw["conv_channels"])
        or not all(isinstance(value, int) and value > 0 for value in raw["freq_subbands"])
        or any(
            not isinstance(raw[key], int) or raw[key] <= 0
            for key in (
                "subband_proj_dim",
                "lstm_hidden",
                "lstm_layers",
                "attention_heads",
                "attention_window",
                "onset_detector_hidden",
                "classifier_hidden",
            )
        )
        or not isinstance(raw["dropout"], (int, float))
        or not 0 <= raw["dropout"] < 1
    ):
        raise DrumsV14RuntimeError("V14 model parameters are incompatible with direct inference")
    return dict(raw)


def _normalise_audio(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        raise DrumsV14RuntimeError("managed Drums audio has an invalid sample rate")
    if audio.ndim == 2:
        audio = audio.mean(dim=0)
    elif audio.ndim != 1:
        raise DrumsV14RuntimeError("managed Drums audio must have one or two dimensions")
    if audio.numel() == 0:
        raise DrumsV14RuntimeError("managed Drums audio is empty")
    audio = audio.to(dtype=torch.float32, device="cpu")
    if sample_rate != SAMPLE_RATE:
        audio = torchaudio.functional.resample(
            audio.unsqueeze(0), sample_rate, SAMPLE_RATE
        ).squeeze(0)
    return audio


def _select_peak_frames(
    probabilities: torch.Tensor, *, threshold: float, min_distance_frames: int
) -> list[int]:
    """Deterministic local-max peak selection without SciPy or legacy helpers."""
    if probabilities.ndim != 1:
        raise DrumsV14RuntimeError("V14 onset output must be one-dimensional")
    candidates: list[int] = []
    for index, value in enumerate(probabilities):
        if float(value) < threshold:
            continue
        left = float(probabilities[index - 1]) if index else float("-inf")
        right = float(probabilities[index + 1]) if index + 1 < len(probabilities) else float("-inf")
        if float(value) >= left and float(value) > right:
            candidates.append(index)
    candidates.sort(key=lambda index: (-float(probabilities[index]), index))
    selected: list[int] = []
    for index in candidates:
        if all(abs(index - existing) >= min_distance_frames for existing in selected):
            selected.append(index)
    return sorted(selected)
