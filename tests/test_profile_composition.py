from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest

import src.worker as worker_module
from src.inference.drums_v14_runtime import DrumsV14Event
from src.inference.guitar_bass import GuitarChart, GuitarNote
from src.inference import guitar_hybrid_profile
from src.inference import guitar_hybrid_v2
from src.inference import drums_v14_runtime
from src.model_bundle import BundleValidationError
import src.profile_composition as composition_module
from src.profile_composition import compose_profile_request
from src.worker import (
    WorkerRequestError,
    _merge_composition_midis,
    discover_model_bundles,
    preflight_chart_request,
    run_chart_request,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _child_bundle(root: Path, *, instrument: str, capability: str) -> tuple[Path, str]:
    profile_id = f"{instrument}-expert"
    checkpoint = root / "weights" / f"{instrument}.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{instrument} weights".encode())
    configuration = root / "profiles" / f"{profile_id}.json"
    configuration.parent.mkdir()
    configuration.write_text(json.dumps({"instrument": instrument}) + "\n", encoding="utf-8")
    component = f"{instrument}.component"
    (root / "strum-model-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": f"{instrument}-fixture",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    component: {
                        "checkpoint": checkpoint.relative_to(root).as_posix(),
                        "sha256": _sha256(checkpoint),
                        "byte_length": checkpoint.stat().st_size,
                    }
                },
                "profiles": {
                    profile_id: {
                        "capability": capability,
                        "instruments": [instrument],
                        "required_components": [component],
                        "difficulty_policies": ["expert_only"],
                        "configuration": configuration.relative_to(root).as_posix(),
                        "configuration_sha256": _sha256(configuration),
                        "configuration_byte_length": configuration.stat().st_size,
                    }
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root, profile_id


def _typed_guitar_bundle(root: Path) -> tuple[Path, str]:
    """Small but fully typed direct profile, with inference patched only at its edge."""
    checkpoint = root / "weights" / "onset.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified onset")
    model_config = root / "configs" / "guitar.yaml"
    model_config.parent.mkdir()
    model_config.write_text("onset: {}\n", encoding="utf-8")
    configuration = root / "profiles" / "guitar-rule.json"
    configuration.parent.mkdir()
    configuration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-guitar-hybrid-rule-profile/v1",
                "onset_threshold": 0.4,
                "latency_offset_ms": 25,
                "min_pitch_amplitude": 0.3,
                "min_pitch": 36,
                "max_pitch": 88,
                "snap_window_ms": 75,
                "sustain_min_duration_ms": 400,
                "max_chord_size": 3,
                "voice_filter": True,
                "basic_pitch_version": "0.4.0",
            }
        ),
        encoding="utf-8",
    )
    (root / "strum-model-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "guitar-rule-fixture",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    "guitar.onset": {
                        "checkpoint": "weights/onset.pt",
                        "sha256": _sha256(checkpoint),
                        "byte_length": checkpoint.stat().st_size,
                        "config": "configs/guitar.yaml",
                        "config_sha256": _sha256(model_config),
                        "config_byte_length": model_config.stat().st_size,
                        "architecture": "GuitarOnsetCRNN/v2",
                    }
                },
                "profiles": {
                    "guitar-rule": {
                        "capability": "guitar.hybrid-v2-rule/v1",
                        "instruments": ["guitar"],
                        "required_components": ["guitar.onset"],
                        "difficulty_policies": ["expert_only"],
                        "configuration": "profiles/guitar-rule.json",
                        "configuration_sha256": _sha256(configuration),
                        "configuration_byte_length": configuration.stat().st_size,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root, "guitar-rule"


def _typed_drums_bundle(root: Path) -> tuple[Path, str]:
    checkpoint = root / "weights" / "v14.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified V14 checkpoint")
    model_config = root / "configs" / "drums-v14.yaml"
    model_config.parent.mkdir()
    model_config.write_text("model: drums-v14\n", encoding="utf-8")
    configuration = root / "profiles" / "drums-v14.json"
    configuration.parent.mkdir()
    configuration.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-drums-v14-expert-profile/v1",
                "model_architecture": "TwoStageDrumsCRNN/v14",
                "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                "segment_duration_seconds": 10,
                "overlap": 0.5,
                "onset_threshold": 0.4,
                "class_thresholds": [0.3, 0.25, 0.35, 0.12, 0.28, 0.12, 0.35, 0.12],
                "min_distance_ms": 20,
                "postprocess": "none",
                "class_to_midi": [96, 97, 98, 98, 99, 99, 100, 100],
                "model_parameters": {
                    "n_mels": 128, "conv_channels": [64, 128, 256, 512],
                    "freq_subbands": [32, 64, 96, 128], "subband_proj_dim": 256,
                    "lstm_hidden": 640, "lstm_layers": 3, "attention_heads": 10,
                    "attention_type": "flash", "attention_window": 512, "dropout": 0.0,
                    "onset_detector_hidden": 320, "classifier_hidden": 640,
                    "num_classes": 8, "predict_velocity": True,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "strum-model-bundle.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "drums-v14-fixture",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    "drums.v14": {
                        "checkpoint": "weights/v14.pt", "sha256": _sha256(checkpoint),
                        "byte_length": checkpoint.stat().st_size,
                        "config": "configs/drums-v14.yaml", "config_sha256": _sha256(model_config),
                        "config_byte_length": model_config.stat().st_size,
                        "architecture": "TwoStageDrumsCRNN/v14",
                        "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                    }
                },
                "profiles": {
                    "drums-v14-expert": {
                        "capability": "drums.v14-expert/v1", "instruments": ["drums"],
                        "required_components": ["drums.v14"], "difficulty_policies": ["expert_only"],
                        "configuration": "profiles/drums-v14.json",
                        "configuration_sha256": _sha256(configuration),
                        "configuration_byte_length": configuration.stat().st_size,
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return root, "drums-v14-expert"


def test_compose_profile_creates_discoverable_multitrack_bundle_without_child_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guitar_root, guitar_profile = _child_bundle(
        tmp_path / "guitar", instrument="guitar", capability="guitar.neural-v1-expert/v1"
    )
    bass_root, bass_profile = _child_bundle(
        tmp_path / "bass", instrument="bass", capability="bass.neural-v1-expert/v1"
    )
    output = tmp_path / "models" / "composition"
    output.parent.mkdir()
    request = tmp_path / "compose.json"
    request.write_text(
        json.dumps(
            {
                "output": str(output),
                "profiles": [
                    {"model_root": str(guitar_root), "profile_id": guitar_profile},
                    {"model_root": str(bass_root), "profile_id": bass_profile},
                ],
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(composition_module, "_validate_child_execution_profile", lambda *_: None)
    result = compose_profile_request(request)

    assert result["status"] == "packaged"
    assert result["instruments"] == ["guitar", "bass"]
    discovered = discover_model_bundles(output.parent)
    assert discovered["candidate_count"] == 1
    profile = discovered["candidates"][0]["profiles"][0]
    assert profile["capability"] == "five-lane.composition/v1"
    assert profile["execution"]["status"] == "available"
    assert str(tmp_path) not in json.dumps(discovered)

    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "five-lane-composition",
                "difficulty_policy": "expert_only",
                "instruments": ["bass"],
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    plan = preflight_chart_request(preflight)
    assert plan["execution"] == "available"
    assert plan["instruments"] == ["bass"]
    assert plan["instrument_results"]["bass"]["status"] == "ready"
    assert plan["composition"]["stages"][0]["status"] == "not_requested"
    assert plan["composition"]["stages"][1]["status"] == "ready"


def test_compose_profile_rejects_untyped_child_profile(tmp_path: Path) -> None:
    guitar_root, guitar_profile = _child_bundle(
        tmp_path / "guitar", instrument="guitar", capability="guitar.neural-v1-expert/v1"
    )
    bass_root, bass_profile = _child_bundle(
        tmp_path / "bass", instrument="bass", capability="bass.neural-v1-expert/v1"
    )
    output = tmp_path / "composition"
    request = tmp_path / "compose.json"
    request.write_text(
        json.dumps(
            {
                "output": str(output),
                "profiles": [
                    {"model_root": str(guitar_root), "profile_id": guitar_profile},
                    {"model_root": str(bass_root), "profile_id": bass_profile},
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BundleValidationError, match="Guitar neural profile"):
        compose_profile_request(request)


def test_composed_profile_rejects_tampered_child_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guitar_root, guitar_profile = _child_bundle(
        tmp_path / "guitar", instrument="guitar", capability="guitar.neural-v1-expert/v1"
    )
    bass_root, bass_profile = _child_bundle(
        tmp_path / "bass", instrument="bass", capability="bass.neural-v1-expert/v1"
    )
    output = tmp_path / "composition"
    request = tmp_path / "compose.json"
    request.write_text(
        json.dumps(
            {
                "output": str(output),
                "profiles": [
                    {"model_root": str(guitar_root), "profile_id": guitar_profile},
                    {"model_root": str(bass_root), "profile_id": bass_profile},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(composition_module, "_validate_child_execution_profile", lambda *_: None)
    compose_profile_request(request)
    (output / "_composition_assets" / "bass" / "weights" / "bass.pt").write_bytes(b"tampered")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "five-lane-composition",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar", "bass"],
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BundleValidationError, match="checkpoint byte length mismatch"):
        preflight_chart_request(preflight)


def test_composition_run_merges_every_selected_child_track(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    guitar_root, guitar_profile = _child_bundle(
        tmp_path / "guitar", instrument="guitar", capability="guitar.neural-v1-expert/v1"
    )
    bass_root, bass_profile = _child_bundle(
        tmp_path / "bass", instrument="bass", capability="bass.neural-v1-expert/v1"
    )
    output = tmp_path / "composition"
    compose_request = tmp_path / "compose.json"
    compose_request.write_text(
        json.dumps(
            {
                "output": str(output),
                "profiles": [
                    {"model_root": str(guitar_root), "profile_id": guitar_profile},
                    {"model_root": str(bass_root), "profile_id": bass_profile},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(composition_module, "_validate_child_execution_profile", lambda *_: None)
    compose_profile_request(compose_request)
    original_preflight = worker_module.preflight_chart_request

    def fake_preflight(path: Path) -> dict[str, object]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["profile_id"] == "five-lane-composition":
            return original_preflight(path)
        return {
            "execution": "available",
            "instruments": raw["instruments"],
            "device": raw["device"],
        }

    def fake_child_run(path: Path) -> dict[str, object]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        child_preflight = json.loads(Path(raw["preflight_request"]).read_text(encoding="utf-8"))
        instrument = child_preflight["instruments"][0]
        track_name = {"guitar": "PART GUITAR", "bass": "PART BASS"}[instrument]
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack([mido.MetaMessage("track_name", name=track_name, time=0)])
        track.extend(
            [
                mido.Message("note_on", note=96, velocity=100, time=0),
                mido.Message("note_off", note=96, velocity=0, time=120),
            ]
        )
        midi.tracks.append(track)
        target = Path(raw["output_dir"])
        target.mkdir(parents=True)
        midi.save(target / "notes.mid")
        return {
            "instrument_results": {instrument: {"status": "succeeded"}},
            "expert_event_count": 1,
        }

    monkeypatch.setattr(worker_module, "preflight_chart_request", fake_preflight)
    monkeypatch.setattr(worker_module, "run_chart_request", fake_child_run)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "five-lane-composition",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar", "bass"],
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "result"
    request = tmp_path / "run.json"
    request.write_text(
        json.dumps(
            {
                "preflight_request": str(preflight),
                "audio_path": str(audio),
                "output_dir": str(result_dir),
            }
        ),
        encoding="utf-8",
    )

    result = run_chart_request(request)

    assert result["instrument_results"] == {
        "guitar": {
            "status": "succeeded",
            "stages": {
                "chart-guitar": {
                    "status": "succeeded",
                    "required": True,
                    "component_ids": ["guitar.guitar.component"],
                    "companion_ids": [],
                    "depends_on": [],
                    "difficulty": "Expert",
                    "artifact_ids": ["notes_midi"],
                }
            },
        },
        "bass": {
            "status": "succeeded",
            "stages": {
                "chart-bass": {
                    "status": "succeeded",
                    "required": True,
                    "component_ids": ["bass.bass.component"],
                    "companion_ids": [],
                    "depends_on": [],
                    "difficulty": "Expert",
                    "artifact_ids": ["notes_midi"],
                }
            },
        },
    }
    midi = mido.MidiFile(result_dir / "notes.mid")
    assert [track.name for track in midi.tracks] == ["PART GUITAR", "PART BASS"]
    run_manifest = json.loads((result_dir / "run.json").read_text(encoding="utf-8"))
    assert run_manifest["difficulty"]["status"] == "succeeded"
    assert str(tmp_path) not in json.dumps(run_manifest)


def test_composition_executes_real_typed_children_before_merging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the outer recursion; inference itself is the only mocked layer."""
    guitar_root, guitar_profile = _typed_guitar_bundle(tmp_path / "guitar")
    drums_root, drums_profile = _typed_drums_bundle(tmp_path / "drums")
    monkeypatch.setattr(guitar_hybrid_profile.importlib.util, "find_spec", lambda _: object())
    monkeypatch.setattr(guitar_hybrid_profile.importlib.metadata, "version", lambda _: "0.4.0")
    monkeypatch.setattr(
        guitar_hybrid_v2,
        "transcribe_guitar_hybrid",
        lambda *_args, **_kwargs: GuitarChart(notes=[GuitarNote(time_ms=0, fret=0)]),
    )

    class FakeDrumsRuntime:
        @classmethod
        def from_profile(cls, *_args: object, **_kwargs: object) -> "FakeDrumsRuntime":
            return cls()

        def transcribe_audio_file(self, _audio: Path) -> list[DrumsV14Event]:
            return [DrumsV14Event(time_ms=0, lane=0, midi_note=96, velocity=100)]

    monkeypatch.setattr(drums_v14_runtime, "DrumsV14Runtime", FakeDrumsRuntime)
    output = tmp_path / "composition"
    compose_request = tmp_path / "compose.json"
    compose_request.write_text(
        json.dumps(
            {
                "output": str(output),
                "profiles": [
                    {"model_root": str(guitar_root), "profile_id": guitar_profile},
                    {"model_root": str(drums_root), "profile_id": drums_profile},
                ],
            }
        ),
        encoding="utf-8",
    )
    compose_profile_request(compose_request)
    assert discover_model_bundles(output.parent)["candidates"][0]["profiles"][0]["execution"][
        "status"
    ] == "available"

    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"fixture")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(output),
                "profile_id": "five-lane-composition",
                "difficulty_policy": "expert_only",
                "instruments": ["drums", "guitar"],
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "result"
    run_request = tmp_path / "run.json"
    run_request.write_text(
        json.dumps(
            {
                "preflight_request": str(preflight),
                "audio_path": str(audio),
                "output_dir": str(result_dir),
            }
        ),
        encoding="utf-8",
    )
    result = run_chart_request(run_request)

    assert result["instrument_results"]["drums"]["status"] == "succeeded"
    assert result["instrument_results"]["guitar"]["status"] == "succeeded"
    assert result["instrument_event_counts"] == {"drums": 1, "guitar": 1}
    midi = mido.MidiFile(result_dir / "notes.mid")
    assert [track.name for track in midi.tracks] == ["PART DRUMS", "PART GUITAR"]


def test_composition_rejects_selected_name_only_track(tmp_path: Path) -> None:
    source = tmp_path / "guitar.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("track_name", name="PART GUITAR", time=0)]))
    midi.save(source)

    with pytest.raises(WorkerRequestError, match="no playable Expert notes"):
        _merge_composition_midis({"guitar": source}, tmp_path / "notes.mid", instruments=["guitar"])
