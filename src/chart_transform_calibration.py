"""Deterministic, STRUM-owned decoder calibration for five-lane transforms."""

from __future__ import annotations

import hashlib
import json
import math

import torch

CALIBRATION_FORMAT = "strum-chart-transform-decoder-calibration/v1"
CALIBRATION_POLICY_ID = "chart-transform-per-lane-f1-grid-v1"
CHECKPOINT_SELECTION_FORMAT = "strum-chart-transform-calibration-checkpoint-selection/v1"
CHECKPOINT_SELECTION_POLICY_ID = "chart-transform-calibration-best-checkpoint/v1"
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


def checkpoint_selection_policy() -> dict[str, object]:
    """Return the fixed V2-only calibration checkpoint selection policy."""
    return {
        "schema_version": 1,
        "format": "strum-chart-transform-calibration-checkpoint-selection-policy/v1",
        "policy_id": CHECKPOINT_SELECTION_POLICY_ID,
        "data_split": "calibration",
        "selection_key": [
            "lane_f1_desc",
            "lane_precision_desc",
            "lane_recall_desc",
            "epoch_asc",
        ],
    }


def checkpoint_selection_policy_evidence() -> dict[str, object]:
    policy = checkpoint_selection_policy()
    return {**policy, "policy_sha256": _canonical_sha256(policy)}


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash exact tensor values without serializing a device-dependent pickle."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint state must be a tensor dictionary")
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


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


def calibration_checkpoint_is_better(
    candidate: dict[str, object], best: dict[str, object] | None
) -> bool:
    """Apply the immutable calibration-only checkpoint ordering."""

    def key(value: dict[str, object]) -> tuple[float, float, float, int]:
        epoch = value.get("epoch")
        metrics = value.get("metrics")
        if (
            not isinstance(epoch, int)
            or isinstance(epoch, bool)
            or epoch < 1
            or not isinstance(metrics, dict)
            or any(
                not isinstance(metrics.get(name), float)
                or not math.isfinite(metrics[name])
                or not 0.0 <= metrics[name] <= 1.0
                for name in ("lane_f1", "lane_precision", "lane_recall")
            )
        ):
            raise ValueError("calibration checkpoint candidate is invalid")
        return (
            metrics["lane_f1"],
            metrics["lane_precision"],
            metrics["lane_recall"],
            -epoch,
        )

    candidate_key = key(candidate)
    return best is None or candidate_key > key(best)


def checkpoint_selection_evidence(
    *,
    epoch: int,
    epochs_evaluated: int,
    selected_state_sha256: str,
    thresholds: tuple[float, ...],
    metrics: dict[str, float],
    calibration_trace: list[dict[str, object]],
) -> dict[str, object]:
    value = {
        "schema_version": 1,
        "format": CHECKPOINT_SELECTION_FORMAT,
        "policy": checkpoint_selection_policy_evidence(),
        "selected_epoch": epoch,
        "epochs_evaluated": epochs_evaluated,
        "selected_state_sha256": selected_state_sha256,
        "thresholds": list(thresholds),
        "metrics": metrics,
        "calibration_trace": calibration_trace,
        "calibration_trace_sha256": _canonical_sha256(calibration_trace),
    }
    if not validate_checkpoint_selection_evidence(value):
        raise ValueError("checkpoint selection evidence is invalid")
    return value


def calibration_evidence(
    *,
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    component_sha256: str,
    calibration_song_ids: list[str],
    split_assignments_sha256: str,
    checkpoint_selection: dict[str, object],
) -> dict[str, object]:
    thresholds, metrics = calibrate_thresholds(probabilities, targets)
    if (
        not validate_checkpoint_selection_evidence(checkpoint_selection)
        or checkpoint_selection["thresholds"] != list(thresholds)
        or checkpoint_selection["metrics"] != metrics
    ):
        raise ValueError("checkpoint selection does not match calibration metrics")
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
        "checkpoint_selection": checkpoint_selection,
    }


def validate_checkpoint_selection_evidence(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    thresholds = value.get("thresholds")
    metrics = value.get("metrics")
    epoch = value.get("selected_epoch")
    epochs_evaluated = value.get("epochs_evaluated")
    trace = value.get("calibration_trace")
    if (
        not isinstance(trace, list)
        or not isinstance(epochs_evaluated, int)
        or isinstance(epochs_evaluated, bool)
        or epochs_evaluated < 1
        or not isinstance(epoch, int)
        or isinstance(epoch, bool)
        or len(trace) != epochs_evaluated
    ):
        return False

    def valid_trace_entry(entry: object, expected_epoch: int) -> bool:
        if not isinstance(entry, dict):
            return False
        entry_thresholds = entry.get("thresholds")
        entry_metrics = entry.get("metrics")
        return (
            set(entry) == {"epoch", "state_sha256", "thresholds", "metrics"}
            and entry.get("epoch") == expected_epoch
            and isinstance(entry.get("state_sha256"), str)
            and len(entry["state_sha256"]) == 64
            and all(character in "0123456789abcdef" for character in entry["state_sha256"])
            and isinstance(entry_thresholds, list)
            and len(entry_thresholds) == 5
            and all(isinstance(item, float) and item in THRESHOLD_GRID for item in entry_thresholds)
            and isinstance(entry_metrics, dict)
            and set(entry_metrics) == {"lane_precision", "lane_recall", "lane_f1"}
            and all(
                isinstance(item, float) and math.isfinite(item) and 0.0 <= item <= 1.0
                for item in entry_metrics.values()
            )
        )

    if not all(valid_trace_entry(entry, index) for index, entry in enumerate(trace, start=1)):
        return False
    selected = trace[epoch - 1] if 1 <= epoch <= len(trace) else None
    winning = max(
        trace,
        key=lambda entry: (
            entry["metrics"]["lane_f1"],
            entry["metrics"]["lane_precision"],
            entry["metrics"]["lane_recall"],
            -entry["epoch"],
        ),
    )
    return (
        set(value)
        == {
            "schema_version",
            "format",
            "policy",
            "selected_epoch",
            "epochs_evaluated",
            "selected_state_sha256",
            "thresholds",
            "metrics",
            "calibration_trace",
            "calibration_trace_sha256",
        }
        and value.get("schema_version") == 1
        and value.get("format") == CHECKPOINT_SELECTION_FORMAT
        and value.get("policy") == checkpoint_selection_policy_evidence()
        and isinstance(epoch, int)
        and not isinstance(epoch, bool)
        and 1 <= epoch <= epochs_evaluated
        and isinstance(value.get("selected_state_sha256"), str)
        and len(value["selected_state_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["selected_state_sha256"])
        and isinstance(thresholds, list)
        and len(thresholds) == 5
        and all(isinstance(item, float) and item in THRESHOLD_GRID for item in thresholds)
        and isinstance(metrics, dict)
        and set(metrics) == {"lane_precision", "lane_recall", "lane_f1"}
        and all(
            isinstance(item, float) and math.isfinite(item) and 0.0 <= item <= 1.0
            for item in metrics.values()
        )
        and value.get("calibration_trace_sha256") == _canonical_sha256(trace)
        and selected is not None
        and selected["state_sha256"] == value["selected_state_sha256"]
        and selected["thresholds"] == thresholds
        and selected["metrics"] == metrics
        and winning == selected
    )


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
            "checkpoint_selection",
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
        and validate_checkpoint_selection_evidence(value.get("checkpoint_selection"))
        and value["checkpoint_selection"].get("thresholds") == thresholds
        and value["checkpoint_selection"].get("metrics") == metrics
    )
