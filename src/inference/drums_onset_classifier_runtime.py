"""Tensor-only executor for the evaluation-only Drums V2 classifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from src.models.onset_classifier import OnsetClassifier

from .drums_onset_classifier_profile import (
    COARSE_MEL_SHAPE,
    CONTEXT_SHAPE,
    FINE_MEL_SHAPE,
)

if TYPE_CHECKING:
    from .drums_onset_classifier_profile import DrumsOnsetClassifierEvaluationProfile


class DrumsOnsetClassifierRuntimeError(RuntimeError):
    """Raised when an evaluation-only V2 classifier cannot execute exactly."""


@dataclass(frozen=True)
class DrumsOnsetClassification:
    """One eight-class probability vector for a caller-supplied onset window."""

    probabilities: tuple[float, ...]


class DrumsOnsetClassifierRuntime:
    """Execute Stage 2 only; onset detection and chart writing are absent."""

    def __init__(self, profile: DrumsOnsetClassifierEvaluationProfile, model: OnsetClassifier, device: torch.device) -> None:
        self.profile = profile
        self.model = model.to(device).eval()
        self.device = device

    @classmethod
    def from_profile(
        cls,
        profile: DrumsOnsetClassifierEvaluationProfile,
        *,
        checkpoint_path: str | Path,
        device: str | torch.device,
    ) -> DrumsOnsetClassifierRuntime:
        target_device = torch.device(device)
        try:
            checkpoint = torch.load(Path(checkpoint_path), map_location=target_device, weights_only=True)
        except Exception as error:
            raise DrumsOnsetClassifierRuntimeError("unable to load verified Drums V2 checkpoint") from error
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model_state_dict"), dict):
            raise DrumsOnsetClassifierRuntimeError("Drums V2 checkpoint must contain model_state_dict")
        model = OnsetClassifier(**profile.model_parameters)
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as error:
            raise DrumsOnsetClassifierRuntimeError("Drums V2 checkpoint is incompatible with its declared model") from error
        return cls(profile, model, target_device)

    @torch.inference_mode()
    def classify_windows(
        self, mel_fine: torch.Tensor, mel_coarse: torch.Tensor, context: torch.Tensor
    ) -> list[DrumsOnsetClassification]:
        fine = _validate_windows(mel_fine, FINE_MEL_SHAPE, "fine mel")
        coarse = _validate_windows(mel_coarse, COARSE_MEL_SHAPE, "coarse mel")
        contexts = _validate_context(context)
        if fine.shape[0] != coarse.shape[0] or fine.shape[0] != contexts.shape[0]:
            raise DrumsOnsetClassifierRuntimeError("Drums V2 inputs must have the same batch size")
        logits = self.model(fine.to(self.device), coarse.to(self.device), contexts.to(self.device))
        if not isinstance(logits, torch.Tensor) or logits.shape != (fine.shape[0], 8):
            raise DrumsOnsetClassifierRuntimeError("Drums V2 model produced an invalid classification tensor")
        probabilities = torch.sigmoid(logits).cpu()
        return [DrumsOnsetClassification(tuple(float(value) for value in row)) for row in probabilities]


def _validate_windows(value: torch.Tensor, expected: tuple[int, int, int], label: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or tuple(value.shape[1:]) != expected:
        raise DrumsOnsetClassifierRuntimeError(f"Drums V2 {label} must have shape [N, {', '.join(map(str, expected))}]")
    if not value.is_floating_point():
        raise DrumsOnsetClassifierRuntimeError(f"Drums V2 {label} must be floating point")
    return value.to(dtype=torch.float32, device="cpu")


def _validate_context(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or tuple(value.shape[1:]) != CONTEXT_SHAPE:
        raise DrumsOnsetClassifierRuntimeError("Drums V2 context must have shape [N, 64]")
    if not value.is_floating_point():
        raise DrumsOnsetClassifierRuntimeError("Drums V2 context must be floating point")
    return value.to(dtype=torch.float32, device="cpu")
