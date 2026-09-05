"""Canonical, host-independent quality policy for Expert profile promotion."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from src.five_lane_runtime_admission import PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT

PROFILE_POLICY_ID: Final = "strum-five-lane-expert-profile/v2"
_POLICY_JSON: Final = json.dumps(
    {
        "policy_id": PROFILE_POLICY_ID,
        "evaluation_split": "test",
        "alignment_tolerance_ms": 50.0,
        "minimum_test_sources": PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT["test"],
        "minimum_onset_f1": 0.5,
        "minimum_fret_f1": 0.5,
        "onset_threshold": 0.4,
        "fret_thresholds": [0.5, 0.5, 0.5, 0.5, 0.5],
        "note_duration_ms": 100.0,
    },
    sort_keys=True,
    separators=(",", ":"),
)


def profile_quality_policy() -> dict[str, object]:
    return json.loads(_POLICY_JSON)


def profile_quality_policy_sha256() -> str:
    return hashlib.sha256(_POLICY_JSON.encode("utf-8")).hexdigest()


def evaluation_matches_experiment(report: dict[str, object], experiment: dict[str, object]) -> bool:
    """Require complete, source-disjoint test evidence from the training view."""
    from src.five_lane_runtime_admission import (
        PROFILE_GRADE_ADMISSION,
        PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT,
    )

    task = experiment.get("task_view")
    if not isinstance(task, dict) or task.get("profile_grade_admission") != PROFILE_GRADE_ADMISSION:
        return False
    if report.get("task_view_sha256") != task.get("sha256"):
        return False
    inputs = task.get("source_inputs")
    if not isinstance(inputs, list):
        return False
    partitions = {split: set() for split in PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT}
    test_records = 0
    for item in inputs:
        if (
            not isinstance(item, dict)
            or item.get("split") not in partitions
            or not isinstance(item.get("source_id"), str)
        ):
            return False
        partitions[item["split"]].add(item["source_id"])
        test_records += item["split"] == "test"
    if any(
        len(partitions[split]) < minimum
        for split, minimum in PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT.items()
    ):
        return False
    if any(
        partitions[left] & partitions[right]
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        return False
    return report.get("records_evaluated") == test_records
