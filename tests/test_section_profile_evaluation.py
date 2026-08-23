from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.inference.section_classifier_profile import (
    CAPABILITY,
    EVALUATION_FORMAT,
    load_section_classifier_candidate,
    load_section_classifier_evaluation_profile,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle
from src.models.section_classifier import SectionClassifier
from src.section_frontend import ROUTER_FEATURE_EXTRACTOR
from src.section_profile_evaluation import (
    SectionProfileEvaluationError,
    evaluate_section_candidate,
    package_section_evaluation_profile,
)
from src.worker import _chart_execution_available, validate_inference_profile


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_view(root: Path) -> Path:
    path = root / "section-guitar-task.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-catalog-task-manifest/v1",
                "task": {
                    "kind": "section_guitar",
                    "pipeline_id": "strum.section-classifier/guitar/v1",
                    "instrument": "guitar",
                    "label_schema": {
                        "id": "midi-section-events/v1",
                        "track_prefixes": ["PART GUITAR"],
                        "difficulty_encoding": "not-applicable",
                    },
                },
            }
        )
    )
    return path


def _candidate_experiment(root: Path) -> tuple[Path, Path]:
    experiment = root / "experiment"
    bundle = experiment / "bundle"
    config = {
        "schema_version": 1,
        "format": "strum-section-classifier-model-config/v1",
        "instrument": "guitar",
        "pipeline_id": "strum.section-classifier/guitar/v1",
        "model_implementation": "SectionClassifier/v1",
        "preprocessing": "section-logmel-librosa-router-windows/v1",
        "labels": ["silence", "constant_strum", "chord_stab", "lead_line", "single_notes", "mixed"],
        "window_seconds": 2.0,
        "hop_seconds": 1.0,
        "mel_shape": [128, 87],
        "feature_extractor": ROUTER_FEATURE_EXTRACTOR,
        "runtime_profile": {
            "format": "strum-section-router-deployment-requirements/v1",
            "status": "not_packageable",
            "reason": "section_router_execution_and_held_out_evaluation_not_proven",
            "requirements": [
                "section_router_profile_loader_tensor_only",
                "held_out_section_calibration_evaluation",
                "held_out_chart_impact_ablation",
                "composed_guitar_chart_profile_contract",
            ],
        },
        "training": {"model_id": "section-test"},
    }
    config_path = bundle / "configs" / "guitar-section.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps(config, sort_keys=True))
    weights = bundle / "weights" / "guitar-section.pt"
    weights.parent.mkdir()
    torch.save({"state_dict": SectionClassifier().state_dict(), "epoch": 1}, weights)
    manifest = {
        "schema_version": 1,
        "model_id": "catalog-section-guitar",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            "section_classifier.guitar": {
                "checkpoint": "weights/guitar-section.pt",
                "sha256": _sha256(weights),
                "byte_length": weights.stat().st_size,
                "config": "configs/guitar-section.json",
                "config_sha256": _sha256(config_path),
                "config_byte_length": config_path.stat().st_size,
                "architecture": "SectionClassifier/v1",
                "preprocessing": "section-logmel-librosa-router-windows/v1",
            }
        },
    }
    manifest_path = bundle / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest))
    experiment.mkdir(exist_ok=True)
    (experiment / "experiment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-experiment/v1",
                "lifecycle": "completed",
                "pipeline": {"id": "strum.section-classifier/guitar", "version": 1},
                "deployment_status": "requires_section_profile_evaluation",
            }
        )
    )
    return experiment, bundle


def _cache(*, cache_dir: Path, **_kwargs: object) -> dict[str, int]:
    cache_dir.mkdir(parents=True)
    generator = np.random.default_rng(8)
    for split in ("val", "test"):
        np.save(
            cache_dir / f"{split}_section_mel.npy",
            generator.normal(size=(6, 128, 87)).astype(np.float32),
        )
        np.save(cache_dir / f"{split}_section_label.npy", np.arange(6, dtype=np.int64))
    return {"val": 6, "test": 6}


def test_section_candidate_is_tensor_only_and_held_out_report_is_path_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _experiment, bundle = _candidate_experiment(tmp_path)
    candidate = load_section_classifier_candidate(
        load_model_bundle(bundle, check_files=True), "guitar"
    )
    assert candidate.component_id == "section_classifier.guitar"
    monkeypatch.setattr("src.section_profile_evaluation._materialize_held_out_cache", _cache)
    task = _task_view(tmp_path)
    report_path = tmp_path / "held-out.json"
    report = evaluate_section_candidate(
        bundle_root=bundle,
        task_view_path=task,
        catalog_root=tmp_path / "private-catalog",
        output_path=report_path,
        instrument="guitar",
    )
    assert report["format"] == EVALUATION_FORMAT
    assert report["split"] == "test"
    assert report["calibration"]["selection_split"] == "val"
    assert report["records_evaluated"] == 6
    assert str(tmp_path) not in report_path.read_text()


def test_section_evaluation_only_profile_remains_unavailable_to_chart_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    experiment, bundle = _candidate_experiment(tmp_path)
    monkeypatch.setattr("src.section_profile_evaluation._materialize_held_out_cache", _cache)
    report_path = tmp_path / "held-out.json"
    evaluate_section_candidate(
        bundle_root=bundle,
        task_view_path=_task_view(tmp_path),
        catalog_root=tmp_path / "private-catalog",
        output_path=report_path,
        instrument="guitar",
    )
    output = tmp_path / "evaluation-profile"
    result = package_section_evaluation_profile(
        experiment_dir=experiment,
        evaluation_path=report_path,
        output_dir=output,
        profile_id="section-guitar-held-out",
        instrument="guitar",
        minimum_accuracy=0.001,
        maximum_expected_calibration_error=1.0,
    )
    assert result["capability"] == CAPABILITY
    assert result["deployment_status"] == "evaluation_only_not_auto_chart_runnable"
    profile = load_section_classifier_evaluation_profile(
        load_model_bundle(output, check_files=True), "section-guitar-held-out"
    )
    assert profile.instrument == "guitar"
    plan = validate_inference_profile(
        output, profile_id="section-guitar-held-out", difficulty_policy="evaluation_only"
    )
    assert plan["capability"] == CAPABILITY
    assert not _chart_execution_available(
        capability=CAPABILITY, difficulty_policy="evaluation_only", instruments=("guitar",)
    )


def test_section_loader_rejects_pickle_like_or_incompatible_state(tmp_path: Path) -> None:
    _experiment, bundle = _candidate_experiment(tmp_path)
    checkpoint = bundle / "weights" / "guitar-section.pt"
    torch.save({"state_dict": {"wrong": torch.tensor([1.0])}}, checkpoint)
    manifest = json.loads((bundle / MANIFEST_FILENAME).read_text())
    component = manifest["components"]["section_classifier.guitar"]
    component["sha256"] = _sha256(checkpoint)
    component["byte_length"] = checkpoint.stat().st_size
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest))
    with pytest.raises(BundleValidationError, match="tensor state is incompatible"):
        load_section_classifier_candidate(load_model_bundle(bundle, check_files=True), "guitar")


def test_section_package_rejects_non_held_out_report(tmp_path: Path) -> None:
    experiment, bundle = _candidate_experiment(tmp_path)
    report = {
        "schema_version": 1,
        "format": EVALUATION_FORMAT,
        "model_id": "catalog-section-guitar",
        "bundle_manifest_sha256": _sha256(bundle / MANIFEST_FILENAME),
        "task_view_sha256": "a" * 64,
        "instrument": "guitar",
        "component_id": "section_classifier.guitar",
        "split": "val",
        "records_evaluated": 1,
    }
    path = tmp_path / "not-held-out.json"
    path.write_text(json.dumps(report))
    with pytest.raises(SectionProfileEvaluationError, match="verified held-out report"):
        package_section_evaluation_profile(
            experiment_dir=experiment,
            evaluation_path=path,
            output_dir=tmp_path / "output",
            profile_id="section-guitar-held-out",
            instrument="guitar",
            minimum_accuracy=0.5,
            maximum_expected_calibration_error=0.2,
        )
