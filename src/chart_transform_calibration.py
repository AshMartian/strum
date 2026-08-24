"""Deterministic, STRUM-owned decoder calibration for five-lane transforms."""

from __future__ import annotations

import hashlib
import json
import math

import torch

CALIBRATION_FORMAT = "strum-chart-transform-decoder-calibration/v1"
CALIBRATION_POLICY_ID = "chart-transform-per-lane-f1-grid-v1"
# Fixed policy bytes: callers cannot provide or reorder candidates.
THRESHOLD_GRID = tuple(index / 20 for index in range(1, 20))


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def calibration_policy() -> dict[str, object]:
    return {
        "schema_version": 1,
        "format": "strum-chart-transform-decoder-calibration-policy/v1",
        "policy_id": CALIBRATION_POLICY_ID,
        "threshold_grid": list(THRESHOLD_GRID),
        "objective": "per_lane_f1_maximize_then_higher_threshold/v1",
    }


def calibration_policy_evidence() -> dict[str, object]:
    policy = calibration_policy()
    return {**policy, "policy_sha256": _canonical_sha256(policy)}


def metrics_from_probabilities(
    probabilities: torch.Tensor, targets: torch.Tensor, thresholds: tuple[float, ...]
) -> dict[str, float]:
    threshold_tensor = torch.tensor(
        thresholds, dtype=probabilities.dtype, device=probabilities.device
    )
    predictions = (probabilities >= threshold_tensor).to(torch.int64)
    expected = targets.to(torch.int64)
    true_positive = int(((predictions == 1) & (expected == 1)).sum())
    false_positive = int(((predictions == 1) & (expected == 0)).sum())
    false_negative = int(((predictions == 0) & (expected == 1)).sum())
    precision = (
        true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"lane_precision": precision, "lane_recall": recall, "lane_f1": f1}


def calibrate_thresholds(
    probabilities: torch.Tensor, targets: torch.Tensor
) -> tuple[tuple[float, ...], dict[str, float]]:
    """Choose each lane's F1 optimum; ties choose the higher threshold."""
    if (
        probabilities.ndim != 2
        or targets.shape != probabilities.shape
        or probabilities.shape[1] != 5
    ):
        raise ValueError("calibration requires five-lane probability and target tensors")
    thresholds: list[float] = []
    for lane in range(5):
        target = targets[:, lane].to(torch.int64)
        best: tuple[float, float] | None = None
        for threshold in THRESHOLD_GRID:
            prediction = (probabilities[:, lane] >= threshold).to(torch.int64)
            true_positive = int(((prediction == 1) & (target == 1)).sum())
            false_positive = int(((prediction == 1) & (target == 0)).sum())
            false_negative = int(((prediction == 0) & (target == 1)).sum())
            precision = (
                true_positive / (true_positive + false_positive)
                if true_positive + false_positive
                else 0.0
            )
            recall = (
                true_positive / (true_positive + false_negative)
                if true_positive + false_negative
                else 0.0
            )
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            candidate = (f1, threshold)
            if best is None or candidate > best:
                best = candidate
        assert best is not None
        thresholds.append(best[1])
    resolved = tuple(thresholds)
    return resolved, metrics_from_probabilities(probabilities, targets, resolved)


def calibration_evidence(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    component_sha256: str,
    calibration_song_ids: list[str],
    split_assignments_sha256: str,
) -> dict[str, object]:
    thresholds, metrics = calibrate_thresholds(probabilities, targets)
    return {
        "schema_version": 1,
        "format": CALIBRATION_FORMAT,
        "policy": calibration_policy_evidence(),
        "component_sha256": component_sha256,
        "split_assignments_sha256": split_assignments_sha256,
        "calibration_song_ids": sorted(calibration_song_ids),
        "thresholds": list(thresholds),
        "metrics": metrics,
        "event_examples": int(targets.shape[0]),
    }


def validate_calibration_evidence(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    thresholds = value.get("thresholds")
    metrics = value.get("metrics")
    return (
        set(value)
        == {
            "schema_version",
            "format",
            "policy",
            "component_sha256",
            "split_assignments_sha256",
            "calibration_song_ids",
            "thresholds",
            "metrics",
            "event_examples",
        }
        and value.get("schema_version") == 1
        and value.get("format") == CALIBRATION_FORMAT
        and value.get("policy") == calibration_policy_evidence()
        and isinstance(value.get("component_sha256"), str)
        and len(value["component_sha256"]) == 64
        and isinstance(value.get("split_assignments_sha256"), str)
        and len(value["split_assignments_sha256"]) == 64
        and isinstance(value.get("calibration_song_ids"), list)
        and value["calibration_song_ids"] == sorted(value["calibration_song_ids"])
        and all(isinstance(item, str) and item for item in value["calibration_song_ids"])
        and isinstance(thresholds, list)
        and len(thresholds) == 5
        and all(isinstance(item, float) and item in THRESHOLD_GRID for item in thresholds)
        and isinstance(metrics, dict)
        and set(metrics) == {"lane_precision", "lane_recall", "lane_f1"}
        and all(
            isinstance(item, float) and math.isfinite(item) and 0.0 <= item <= 1.0
            for item in metrics.values()
        )
        and isinstance(value.get("event_examples"), int)
        and not isinstance(value.get("event_examples"), bool)
        and value["event_examples"] > 0
    )
