from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest

from src.model_bundle import MANIFEST_FILENAME, BundleValidationError
from src.worker import (
    PROTOCOL_VERSION,
    _run_without_legacy_output,
    _runtime_payload,
    _write_expert_guitar_midi,
    inspect_catalog,
    preflight_bundle,
    preflight_chart_request,
    prepare_dataset_request,
    validate_inference_profile,
)


def _bundle(root: Path, component: dict[str, object]) -> Path:
    checkpoint = root / "weights" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"verified weights")
    component.update(
        {
            "checkpoint": "weights/model.pt",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "byte_length": checkpoint.stat().st_size,
        }
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "verified-test-bundle",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {"guitar.onset": component},
            }
        )
    )
    return root


def test_probe_declares_versioned_runtime_and_available_pipelines() -> None:
    payload = _runtime_payload()

    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["python_requires"] == ">=3.11"
    assert "guitar.onset-fret/v1" in payload["pipelines"]
    assert "dataset_prepare" in payload["capabilities"]
    assert "chart_run" in payload["capabilities"]
    assert isinstance(payload["optional_dependencies"]["basic_pitch"]["available"], bool)
    assert "model_bundle_preflight" in payload["capabilities"]


def test_legacy_inference_output_is_not_exposed_to_worker_clients(
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret_path = "/private/catalog/audio.opus"

    def legacy_inference() -> str:
        print(f"Predicting MIDI for {secret_path}")
        os.write(2, secret_path.encode())
        return "chart"

    assert _run_without_legacy_output(legacy_inference) == "chart"
    captured = capfd.readouterr()
    assert secret_path not in captured.out
    assert secret_path not in captured.err


def _catalog_asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    path = root / "assets" / "sha256" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _guitar_catalog(root: Path) -> None:
    source_id = "octave-src-aaaaaaaa"
    record = {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {"training_use": "allowed", "provenance": "Reviewed", "license": "test-only"},
        "metadata": {"name": "Fixture"},
        "chart": {
            "notes_midi": _catalog_asset(root, b"midi", "notes.mid"),
            "instruments": {
                "guitar": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART GUITAR"],
                }
            },
        },
        "audio": {"guitar": _catalog_asset(root, b"audio", "guitar.ogg")},
    }
    (root / "records.jsonl").write_text(json.dumps(record) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "worker-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_catalog_inspect_and_prepare_emit_path_free_task_view(tmp_path: Path) -> None:
    _guitar_catalog(tmp_path)
    inspection = inspect_catalog(tmp_path, "guitar.onset-fret/v1")
    assert inspection == {
        "status": "ready",
        "catalog_id": "worker-fixture",
        "record_count": 1,
        "allowed_record_count": 1,
        "pipeline_id": "guitar.onset-fret/v1",
    }
    request_path = tmp_path / "request.json"
    output = tmp_path / "views" / "guitar.json"
    request_path.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "guitar.onset-fret/v1",
                "output": str(output),
                "options": {"required_difficulty": "expert"},
            }
        )
    )

    result = prepare_dataset_request(request_path)

    assert result["status"] == "prepared"
    assert result["output_name"] == "guitar.json"
    assert result["record_count"] == 1
    assert str(tmp_path) not in output.read_text()


def test_preflight_requires_hash_and_length_for_deployable_components(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})

    result = preflight_bundle(root, required_components=["guitar.onset"])

    assert result["status"] == "ready"
    assert result["components"][0]["id"] == "guitar.onset"
    assert result["manifest_sha256"]


def test_preflight_rejects_incomplete_or_missing_components(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {})
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    del manifest["components"]["guitar.onset"]["byte_length"]
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    with pytest.raises(BundleValidationError, match="byte_length"):
        preflight_bundle(root, required_components=["guitar.onset"])
    with pytest.raises(BundleValidationError, match="not declared"):
        preflight_bundle(root, required_components=["guitar.fret"])


def test_profile_validation_requires_declared_companions_and_difficulty_policy(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"] = {
        "guitar-default": {
            "capability": "guitar.audio_to_chart/v1",
            "instruments": ["guitar"],
            "required_components": ["guitar.onset"],
            "difficulty_policies": ["expert_only", "deterministic-v1"],
        }
    }
    manifest_path.write_text(json.dumps(manifest))

    result = validate_inference_profile(
        root, profile_id="guitar-default", difficulty_policy="expert_only"
    )

    assert result["capability"] == "guitar.audio_to_chart/v1"
    assert result["instruments"] == ["guitar"]
    with pytest.raises(BundleValidationError, match="does not support"):
        validate_inference_profile(
            root, profile_id="guitar-default", difficulty_policy="learned:bad"
        )


def test_chart_preflight_returns_an_explicit_non_execution_plan(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"] = {
        "guitar-default": {
            "capability": "guitar.audio_to_chart/v1",
            "instruments": ["guitar"],
            "required_components": ["guitar.onset"],
            "difficulty_policies": ["expert_only"],
        }
    }
    manifest_path.write_text(json.dumps(manifest))
    request = tmp_path / "chart-request.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(root),
                "profile_id": "guitar-default",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
                "device": "cuda",
            }
        )
    )

    plan = preflight_chart_request(request)

    assert plan["status"] == "ready"
    assert plan["execution"] == "not_available"
    assert plan["components"][0]["id"] == "guitar.onset"


def test_expert_guitar_export_never_materializes_lower_difficulties(tmp_path: Path) -> None:
    chart = SimpleNamespace(
        tempo_bpm=120.0,
        notes=[SimpleNamespace(time_ms=100.0, duration_ms=200.0, fret=2)],
        chords=[SimpleNamespace(time_ms=500.0, duration_ms=100.0, frets=[0, 4])],
    )
    output = tmp_path / "notes.mid"

    _write_expert_guitar_midi(chart, output)

    note_ons = {
        message.note
        for track in mido.MidiFile(output).tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    assert note_ons == {96, 98, 100}
