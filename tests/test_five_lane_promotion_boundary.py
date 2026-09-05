"""Promotion rejects biased evaluation and mismatched training lineage."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("instrument", ["guitar", "bass", "keys"])
@pytest.mark.parametrize("override", [{"tolerance_ms": 1000}, {"limit_songs": 1}])
def test_evaluation_rejects_metric_and_subset_overrides(
    tmp_path: Path,
    instrument: str,
    override: dict,
) -> None:
    module = importlib.import_module(f"src.{instrument}_profile_packaging")
    evaluate = getattr(module, f"evaluate_{instrument}_candidate")
    error = getattr(module, f"{instrument.title()}ProfilePackagingError")
    with pytest.raises(error, match="tolerance|limit_songs"):
        evaluate(
            bundle_root=tmp_path / "bundle",
            task_view_path=tmp_path / "view.json",
            catalog_root=tmp_path / "catalog",
            output_path=tmp_path / "report.json",
            device="cpu",
            **override,
        )


@pytest.mark.parametrize("instrument", ["guitar", "bass", "keys"])
@pytest.mark.parametrize(
    "mutation",
    ["validation", "tolerance", "subset", "foreign_view", "overlap", "missing_admission"],
)
def test_package_rejects_biased_or_unbound_evidence(
    tmp_path: Path,
    instrument: str,
    mutation: str,
) -> None:
    fixture = importlib.import_module(f"tests.test_{instrument}_neural_profile")
    builder = getattr(
        fixture,
        {"guitar": "_worker_experiment", "bass": "_bass_experiment", "keys": "_keys_experiment"}[
            instrument
        ],
    )
    experiment, bundle = builder(tmp_path)
    report_path = fixture._evaluation(bundle, tmp_path / "evaluation.json")
    report = json.loads(report_path.read_text())
    experiment_path = experiment / "experiment.json"
    training = json.loads(experiment_path.read_text())
    if mutation == "validation":
        report["split"] = "val"
    elif mutation == "tolerance":
        report["alignment_tolerance_ms"] = 1000
    elif mutation == "subset":
        report["records_evaluated"] = 1
    elif mutation == "foreign_view":
        report["task_view_sha256"] = "b" * 64
    elif mutation == "overlap":
        training["task_view"]["source_inputs"][-1]["source_id"] = "train-0"
    else:
        training["task_view"].pop("profile_grade_admission")
    report_path.write_text(json.dumps(report))
    experiment_path.write_text(json.dumps(training))
    module = importlib.import_module(f"src.{instrument}_profile_packaging")
    error = getattr(module, f"{instrument.title()}ProfilePackagingError")
    with pytest.raises(error, match="evaluation|report|gate"):
        getattr(module, f"package_{instrument}_profile")(
            experiment_dir=experiment,
            evaluation_path=report_path,
            output_dir=tmp_path / "profile",
            profile_id=f"{instrument}-expert",
        )
    assert not (tmp_path / "profile").exists()
