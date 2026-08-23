"""
SectionRouter — predicts per-1s section labels and routes onset detection.

Wraps the trained SectionClassifier. For each 1-s hop in audio, predicts a
label in {silence, constant_strum, chord_stab, lead_line, single_notes, mixed}
plus per-class confidences.

Used by guitar_bass.py to switch onset-detection strategies per section.

Set STRUM_GB_USE_ROUTER=0 to disable; the classifier checkpoint is optional
and the router degrades gracefully to "mixed" everywhere if not present.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src import section_frontend
from src.section_frontend import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    normalize_router_patches,
    resample_router_audio,
    router_patches,
)

logger = logging.getLogger(__name__)

# Backward-compatible public location for artifact inspection.  The object is
# defined by the shared executable frontend rather than duplicated here.
ROUTER_FEATURE_EXTRACTOR = section_frontend.ROUTER_FEATURE_EXTRACTOR

LABELS = ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"]
LABEL_TO_IDX = {label: index for index, label in enumerate(LABELS)}

DEFAULT_CKPT = "checkpoints/section_classifier/best.pt"


@dataclass
class Section:
    t_start_s: float
    t_end_s: float
    label: str
    probs: np.ndarray  # shape (6,)


_ROUTER_CACHE: dict[str, SectionRouter | None] = {}


def get_router(checkpoint: str = DEFAULT_CKPT) -> SectionRouter | None:
    """Singleton accessor; returns None if disabled or checkpoint missing."""
    if os.environ.get("STRUM_GB_USE_ROUTER", "1") == "0":
        return None
    key = checkpoint
    if key in _ROUTER_CACHE:
        return _ROUTER_CACHE[key]
    if not Path(checkpoint).exists():
        logger.warning("section classifier checkpoint not found: %s (router disabled)", checkpoint)
        _ROUTER_CACHE[key] = None
        return None
    try:
        router = SectionRouter(checkpoint)
        _ROUTER_CACHE[key] = router
        return router
    except Exception as exc:
        logger.warning("failed to load section router: %s", exc)
        _ROUTER_CACHE[key] = None
        return None


class SectionRouter:
    """Loads SectionClassifier and predicts section labels over an audio array."""

    def __init__(self, checkpoint: str = DEFAULT_CKPT, device: str | None = None):
        from src.models.section_classifier import SectionClassifier

        # Force CPU by default — model is tiny (162k params), and torchaudio's
        # GPU STFT JIT-compile fails on GB10 (NVRTC sm-arch error). CPU is
        # fast enough (~50ms/song). Override with STRUM_SECTION_DEVICE=cuda.
        env_dev = os.environ.get("STRUM_SECTION_DEVICE")
        if device is None:
            device = env_dev or "cpu"
        self.device = torch.device(device)
        self.model = SectionClassifier().to(self.device)
        # This legacy path remains opt-in and is never an OCTAVE-deployable
        # profile.  Still avoid executing pickle payloads while loading its
        # state dictionary.
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        logger.info(
            "SectionRouter loaded: %s (val_acc=%.3f) on %s",
            checkpoint,
            ckpt.get("val_acc", float("nan")),
            self.device,
        )

    @torch.no_grad()
    def predict(self, audio: np.ndarray, sr: int) -> list[Section]:
        """Predict per-1s-hop section labels for an audio array."""
        # This shared frontend is also used for catalog training.  Keep the
        # router input behavior stable while making the contract executable.
        audio = resample_router_audio(audio, sr)

        if len(audio) < WINDOW_SAMPLES:
            return [Section(0.0, len(audio) / SAMPLE_RATE, "mixed", np.zeros(len(LABELS)))]

        starts, patches = router_patches(audio)
        patches = normalize_router_patches(patches)

        mel_t = torch.from_numpy(patches).unsqueeze(1).to(self.device)  # (N,1,M,T)
        logits = self.model(mel_t)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=-1)

        sections = []
        for i, s in enumerate(starts):
            sections.append(
                Section(
                    t_start_s=s / SAMPLE_RATE,
                    t_end_s=(s + WINDOW_SAMPLES) / SAMPLE_RATE,
                    label=LABELS[int(preds[i])],
                    probs=probs[i],
                )
            )
        return sections


def label_at_time(sections: list[Section], t_s: float) -> str:
    """Return the label of the section containing time t_s (or nearest)."""
    if not sections:
        return "mixed"
    # Linear scan; sections are short enough this is fine.
    best_label = sections[0].label
    for sec in sections:
        if sec.t_start_s <= t_s < sec.t_end_s:
            return sec.label
        if sec.t_start_s <= t_s:
            best_label = sec.label
    return best_label
