"""The exact log-mel frontend used by the legacy :mod:`section_router`.

This module is intentionally small and NumPy/librosa based.  It is shared by
the legacy router and the catalog section preprocessor so a catalog-trained
checkpoint is trained on the same feature values the router would receive.
Tensor shape is not considered sufficient compatibility evidence: the audio
decode, resampling, Mel scale, STFT padding, frame slicing, and normalization
are all part of this contract.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 30.0
FMAX = 8000.0
WINDOW_S = 2.0
HOP_S = 1.0
WINDOW_SAMPLES = int(WINDOW_S * SAMPLE_RATE)
WINDOW_FRAMES = WINDOW_SAMPLES // HOP_LENGTH + 1
HOP_SAMPLES = int(HOP_S * SAMPLE_RATE)
LOG_OFFSET = 1e-8
NORMALIZATION_EPS = 1e-5

# This object is portable manifest data, not a best-effort description.  Keep
# it coupled to the functions below and use it in both runtime and bundles.
ROUTER_FEATURE_EXTRACTOR: dict[str, object] = {
    "format": "strum-section-feature-extractor/v1",
    "backend": "librosa",
    "sample_rate": SAMPLE_RATE,
    "channel_mixdown": "librosa.load(mono=True)",
    "audio_decode": "librosa.load(sr=22050, mono=True)",
    "resampler": "librosa.load/default",
    "n_mels": N_MELS,
    "n_fft": N_FFT,
    "hop_length": HOP_LENGTH,
    "fmin": FMIN,
    "fmax": FMAX,
    "power": 2.0,
    "window": "hann",
    "center": True,
    "pad_mode": "constant",
    "mel_scale": "slaney",
    "mel_norm": "slaney",
    "normalization": "per_window_mean_std_eps_1e-5",
    "log_offset": LOG_OFFSET,
    "window_seconds": WINDOW_S,
    "hop_seconds": HOP_S,
    "window_frames": WINDOW_FRAMES,
}


def load_router_audio(path: Path) -> np.ndarray:
    """Decode exactly as ``transcribe_guitar`` supplies router audio."""
    import librosa  # noqa: PLC0415

    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return np.asarray(audio, dtype=np.float32)


def resample_router_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Apply the router's in-memory resampling policy when necessary."""
    mono = np.asarray(audio, dtype=np.float32)
    if sample_rate == SAMPLE_RATE:
        return mono
    import librosa  # noqa: PLC0415

    return np.asarray(
        librosa.resample(mono, orig_sr=sample_rate, target_sr=SAMPLE_RATE), dtype=np.float32
    )


def compute_router_log_mel(audio: np.ndarray) -> np.ndarray:
    """Return the legacy full-song Slaney log-mel matrix as ``float32``."""
    import librosa  # noqa: PLC0415

    mel = librosa.feature.melspectrogram(
        y=np.asarray(audio, dtype=np.float32),
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
        window="hann",
        center=True,
        pad_mode="constant",
        htk=False,
        norm="slaney",
    )
    return np.log(mel + LOG_OFFSET).astype(np.float32)


def router_window_starts(audio_length: int) -> list[int]:
    """Return the router's complete-window starts, including its final tail."""
    if audio_length < WINDOW_SAMPLES:
        return []
    starts = list(range(0, audio_length - WINDOW_SAMPLES + 1, HOP_SAMPLES))
    if starts[-1] + WINDOW_SAMPLES < audio_length:
        starts.append(audio_length - WINDOW_SAMPLES)
    return starts


def router_patch_from_log_mel(log_mel: np.ndarray, start_sample: int) -> np.ndarray:
    """Slice one patch with the router's frame and trailing-edge behavior."""
    if start_sample < 0:
        raise ValueError("section window start must be non-negative")
    frame_start = start_sample // HOP_LENGTH
    frame_end = frame_start + WINDOW_FRAMES
    if frame_end <= log_mel.shape[1]:
        return np.asarray(log_mel[:, frame_start:frame_end], dtype=np.float32)
    slab = log_mel[:, frame_start:]
    if slab.shape[1] == 0:
        raise ValueError("section window starts beyond the decoded audio")
    return np.pad(
        slab,
        ((0, 0), (0, frame_end - slab.shape[1])),
        mode="edge",
    ).astype(np.float32)


def router_patches(audio: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Extract all runtime router patches before per-patch normalization."""
    starts = router_window_starts(len(audio))
    if not starts:
        return starts, np.empty((0, N_MELS, WINDOW_FRAMES), dtype=np.float32)
    log_mel = compute_router_log_mel(audio)
    patches = np.stack([router_patch_from_log_mel(log_mel, start) for start in starts])
    return starts, patches


def normalize_router_patches(patches: np.ndarray) -> np.ndarray:
    """Apply the per-window normalization used immediately before inference."""
    values = np.asarray(patches, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (N_MELS, WINDOW_FRAMES):
        raise ValueError("section patches have an incompatible shape")
    mean = values.mean(axis=(1, 2), keepdims=True)
    std = values.std(axis=(1, 2), keepdims=True) + NORMALIZATION_EPS
    return (values - mean) / std
