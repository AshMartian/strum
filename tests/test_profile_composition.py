from __future__ import annotations

import hashlib
import json
from pathlib import Path

import mido
import pytest

import src.worker as worker_module
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


def test_composition_rejects_selected_name_only_track(tmp_path: Path) -> None:
    source = tmp_path / "guitar.mid"
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(mido.MidiTrack([mido.MetaMessage("track_name", name="PART GUITAR", time=0)]))
    midi.save(source)

    with pytest.raises(WorkerRequestError, match="no playable Expert notes"):
        _merge_composition_midis({"guitar": source}, tmp_path / "notes.mid", instruments=["guitar"])
