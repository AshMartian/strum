from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from scripts import preprocess_section_windows
from scripts.build_catalog_section_labels import _parse_chart_onsets
from src.inference.section_router import ROUTER_FEATURE_EXTRACTOR
from src.section_frontend import (
    FMAX,
    FMIN,
    HOP_LENGTH,
    LOG_OFFSET,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
    WINDOW_FRAMES,
    WINDOW_SAMPLES,
    compute_router_log_mel,
    load_router_audio,
    normalize_router_patches,
    router_patch_from_log_mel,
    router_patches,
)
from src.section_worker_training import SECTION_FEATURE_EXTRACTOR


def _write_two_part_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    guitar = mido.MidiTrack()
    guitar.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    # Two Expert lanes ten ticks apart are one chord under the declared 25 ms
    # grouping rule at 120 BPM.
    guitar.append(mido.Message("note_on", note=96, velocity=100, time=0))
    guitar.append(mido.Message("note_on", note=97, velocity=100, time=10))
    guitar.append(mido.Message("note_off", note=96, velocity=0, time=230))
    guitar.append(mido.Message("note_off", note=97, velocity=0, time=0))
    midi.tracks.append(guitar)

    bass = mido.MidiTrack()
    bass.append(mido.MetaMessage("track_name", name="PART BASS", time=0))
    bass.append(mido.Message("note_on", note=100, velocity=100, time=0))
    bass.append(mido.Message("note_off", note=100, velocity=0, time=240))
    midi.tracks.append(bass)
    midi.save(path)


def test_catalog_section_labels_use_only_the_exact_declared_track(tmp_path: Path) -> None:
    midi_path = tmp_path / "notes.mid"
    _write_two_part_midi(midi_path)

    guitar = _parse_chart_onsets(midi_path, "PART GUITAR")
    bass = _parse_chart_onsets(midi_path, "PART BASS")

    assert len(guitar) == 1
    assert guitar[0][1] == frozenset({0, 1})
    assert len(bass) == 1
    assert bass[0][1] == frozenset({4})
    assert _parse_chart_onsets(midi_path, "PART KEYS") == []


def _legacy_router_reference(audio: np.ndarray) -> tuple[list[int], np.ndarray]:
    """Independent copy of the frontend that predated the shared contract."""
    import librosa

    starts = list(range(0, len(audio) - WINDOW_SAMPLES + 1, SAMPLE_RATE))
    if starts[-1] + WINDOW_SAMPLES < len(audio):
        starts.append(len(audio) - WINDOW_SAMPLES)
    full = librosa.feature.melspectrogram(
        y=audio,
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
    full = np.log(full + LOG_OFFSET).astype(np.float32)
    patches = np.zeros((len(starts), N_MELS, WINDOW_FRAMES), dtype=np.float32)
    for index, start in enumerate(starts):
        first = start // HOP_LENGTH
        final = first + WINDOW_FRAMES
        if final > full.shape[1]:
            slab = full[:, first:]
            patches[index] = np.pad(slab, ((0, 0), (0, final - slab.shape[1])), mode="edge")
        else:
            patches[index] = full[:, first:final]
    return starts, normalize_router_patches(patches)


def test_catalog_section_frontend_is_exactly_the_legacy_router_frontend() -> None:
    assert SECTION_FEATURE_EXTRACTOR is ROUTER_FEATURE_EXTRACTOR
    assert SECTION_FEATURE_EXTRACTOR["backend"] == "librosa"
    assert SECTION_FEATURE_EXTRACTOR["mel_scale"] == "slaney"
    assert SECTION_FEATURE_EXTRACTOR["pad_mode"] == "constant"


def test_shared_section_frontend_matches_the_prior_router_calculation() -> None:
    samples = np.arange(SAMPLE_RATE * 3 + 1000, dtype=np.float32)
    audio = (np.sin(samples * 0.011) + 0.25 * np.sin(samples * 0.073)).astype(np.float32)

    expected_starts, expected = _legacy_router_reference(audio)
    starts, patches = router_patches(audio)

    assert starts == expected_starts
    np.testing.assert_allclose(normalize_router_patches(patches), expected, rtol=0, atol=0)


def test_catalog_section_cache_preserves_source_id_and_router_features(tmp_path: Path) -> None:
    audio_path = tmp_path / "catalog-asset.wav"
    raw = np.sin(np.arange(SAMPLE_RATE * 3, dtype=np.float32) * 0.02).astype(np.float32)
    sf.write(audio_path, raw, SAMPLE_RATE, subtype="FLOAT")

    preprocess_section_windows.process_split(
        [
            {
                "source_id": "octave-src-1234",
                "audio_path": str(audio_path),
                "split": "train",
                "t_start_s": 0.0,
                "label": "single_notes",
            }
        ],
        "train",
        tmp_path,
    )

    metadata = (tmp_path / "train_section_meta.json").read_text(encoding="utf-8")
    assert '"source_id": "octave-src-1234"' in metadata
    assert "song_id" not in metadata
    cache = np.load(tmp_path / "train_section_mel.npy")
    decoded = load_router_audio(audio_path)
    expected = router_patch_from_log_mel(compute_router_log_mel(decoded), 0)
    assert cache.dtype == np.float32
    np.testing.assert_allclose(cache[0], expected, rtol=0, atol=0)
