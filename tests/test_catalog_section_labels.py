from __future__ import annotations

from pathlib import Path

import mido
import numpy as np
import torch

from scripts import preprocess_section_windows
from scripts.build_catalog_section_labels import _parse_chart_onsets
from src.inference.section_router import ROUTER_FEATURE_EXTRACTOR
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


def test_catalog_section_frontend_is_explicitly_not_the_legacy_router_frontend() -> None:
    """A matching shape/label set must not create an implicit profile bridge."""
    assert SECTION_FEATURE_EXTRACTOR["backend"] == "torchaudio"
    assert ROUTER_FEATURE_EXTRACTOR["backend"] == "librosa"
    assert SECTION_FEATURE_EXTRACTOR["mel_scale"] != ROUTER_FEATURE_EXTRACTOR["mel_scale"]
    assert SECTION_FEATURE_EXTRACTOR["pad_mode"] != ROUTER_FEATURE_EXTRACTOR["pad_mode"]
    assert SECTION_FEATURE_EXTRACTOR["mel_norm"] != ROUTER_FEATURE_EXTRACTOR["mel_norm"]


def test_catalog_section_cache_preserves_source_id_not_legacy_song_id(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        preprocess_section_windows,
        "load_audio_mono22k",
        lambda _path: np.zeros(preprocess_section_windows.WINDOW_SAMPLES, dtype=np.float32),
    )
    monkeypatch.setattr(
        preprocess_section_windows,
        "compute_log_mel",
        lambda _audio: torch.zeros((128, preprocess_section_windows.WINDOW_FRAMES)),
    )

    preprocess_section_windows.process_split(
        [
            {
                "source_id": "octave-src-1234",
                "audio_path": "/ephemeral/catalog-asset.wav",
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
