from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.fret_mapper_candidate import load_fret_mapper_candidate
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.models.fret_mapper import FretMapperMLP


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    weights = bundle / "weights" / "guitar-fret-mapper.pt"
    config_path = bundle / "configs" / "guitar-fret-mapper.json"
    weights.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    model = FretMapperMLP(hidden=3)
    torch.save(
        {
            "format": "strum-fret-mapper-weights/v1",
            "model_state": model.state_dict(),
            "input_dimension": 95,
            "hidden": 3,
            "output_dimension": 5,
            "feature_mean": torch.zeros(95),
            "feature_std": torch.ones(95),
        },
        weights,
    )
    config = {
        "schema_version": 1,
        "format": "strum-fret-mapper-model-config/v1",
        "instrument": "guitar",
        "pipeline_id": "strum.fret-mapper/guitar/v1",
        "model_implementation": "FretMapperMLP/v1",
        "preprocessing": "basic-pitch-onset-features/v1",
        "feature_dimension": 95,
        "label_schema": "five-lane-fret-mapper-midi/v1",
        "basic_pitch": {
            "distribution": "basic-pitch",
            "version": "0.4.0",
            "onset_threshold": 0.5,
            "frame_threshold": 0.3,
            "min_note_length": 11,
        },
        "model": {
            "format": "strum-fret-mapper-mlp/v1",
            "input_dimension": 95,
            "hidden": 3,
            "output_dimension": 5,
            "dropout": 0.2,
        },
        "training": {"model_id": "candidate-fixture"},
    }
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_id": "candidate-fixture",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            "fret_mapper.guitar": {
                "checkpoint": "weights/guitar-fret-mapper.pt",
                "sha256": _sha256(weights),
                "byte_length": weights.stat().st_size,
                "config": "configs/guitar-fret-mapper.json",
                "config_sha256": _sha256(config_path),
                "config_byte_length": config_path.stat().st_size,
                "architecture": "FretMapperMLP/v1",
                "preprocessing": "basic-pitch-onset-features/v1",
            }
        },
    }
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest), encoding="utf-8")
    return bundle


def test_candidate_loader_requires_tensor_only_checkpoint_and_exact_architecture(
    tmp_path: Path,
) -> None:
    bundle = _candidate_bundle(tmp_path)

    candidate = load_fret_mapper_candidate(load_model_bundle(bundle, check_files=True), "guitar")

    assert candidate.component_id == "fret_mapper.guitar"
    assert candidate.feature_mean.shape == (95,)
    assert candidate.feature_std.shape == (95,)
    assert candidate.basic_pitch["version"] == "0.4.0"
    assert candidate.model(torch.zeros(1, 95)).shape == (1, 5)


def test_candidate_loader_rejects_legacy_numpy_checkpoint_payload(tmp_path: Path) -> None:
    bundle = _candidate_bundle(tmp_path)
    weights = bundle / "weights" / "guitar-fret-mapper.pt"
    model = FretMapperMLP(hidden=3)
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": 95,
            "hidden": 3,
            "feature_mean": np.zeros(95, dtype=np.float32),
            "feature_std": np.ones(95, dtype=np.float32),
            "best_val_f1": 0.2,
        },
        weights,
    )
    manifest_path = bundle / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component = manifest["components"]["fret_mapper.guitar"]
    component["sha256"] = _sha256(weights)
    component["byte_length"] = weights.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="tensor-only loadable"):
        load_fret_mapper_candidate(load_model_bundle(bundle, check_files=True), "guitar")


def test_candidate_loader_rejects_checkpoint_architecture_drift(tmp_path: Path) -> None:
    bundle = _candidate_bundle(tmp_path)
    weights = bundle / "weights" / "guitar-fret-mapper.pt"
    model = FretMapperMLP(hidden=4)
    torch.save(
        {
            "format": "strum-fret-mapper-weights/v1",
            "model_state": model.state_dict(),
            "input_dimension": 95,
            "hidden": 4,
            "output_dimension": 5,
            "feature_mean": torch.zeros(95),
            "feature_std": torch.ones(95),
        },
        weights,
    )
    manifest_path = bundle / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    component = manifest["components"]["fret_mapper.guitar"]
    component["sha256"] = _sha256(weights)
    component["byte_length"] = weights.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BundleValidationError, match="architecture does not match"):
        load_fret_mapper_candidate(load_model_bundle(bundle, check_files=True), "guitar")


def test_candidate_loader_rechecks_bundle_hashes(tmp_path: Path) -> None:
    bundle = _candidate_bundle(tmp_path)
    weights = bundle / "weights" / "guitar-fret-mapper.pt"
    weights.write_bytes(b"tampered")

    with pytest.raises(BundleValidationError, match="bundle failed verification"):
        load_fret_mapper_candidate(load_model_bundle(bundle), "guitar")


def test_trainer_writes_a_weights_only_checkpoint(tmp_path: Path) -> None:
    for source_id, split in (("train-song", "train"), ("val-song", "val")):
        np.savez_compressed(
            tmp_path / f"{source_id}.npz",
            X=np.zeros((2, 95), dtype=np.float32),
            Y=np.zeros((2, 5), dtype=np.float32),
            song_id=source_id,
            split=split,
        )
    checkpoint = tmp_path / "mapper.pt"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_fret_mapper.py",
            "--cache-dir",
            str(tmp_path),
            "--out",
            str(checkpoint),
            "--epochs",
            "1",
            "--batch",
            "2",
            "--hidden",
            "3",
            "--device",
            "cpu",
            "--use-catalog-splits",
        ],
        check=True,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)

    assert set(payload) == {
        "format",
        "model_state",
        "input_dimension",
        "hidden",
        "output_dimension",
        "feature_mean",
        "feature_std",
    }
    assert isinstance(payload["feature_mean"], torch.Tensor)
