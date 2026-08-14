from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.model_bundle import MANIFEST_FILENAME, BundleValidationError
from src.worker import PROTOCOL_VERSION, _runtime_payload, preflight_bundle


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
    assert payload["pipelines"] == []
    assert "model_bundle_preflight" in payload["capabilities"]


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
