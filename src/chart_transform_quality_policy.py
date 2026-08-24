"""Immutable STRUM-owned admission policy for five-lane transform profiles.

The policy deliberately lives outside worker request schemas.  A renderer may
discover it, but cannot choose or weaken its thresholds.  Evaluation records
the exact policy bytes and its decision; packaging and preflight independently
recompute that decision before admitting a profile.
"""

from __future__ import annotations

import hashlib
import json
import math

QUALITY_POLICY_FORMAT = "strum-chart-transform-promotion-quality-policy/v1"
QUALITY_POLICY_ID = "chart-transform-five-lane-held-out-v1"

# A five-lane transform must beat a weak event-copy baseline while retaining
# both coverage and precision.  These initial bounds intentionally leave the
# observed 0.3297 F1 local baseline ineligible for deployment.  They apply to
# the candidate's song-disjoint validation split, never an arbitrary UI value.
QUALITY_POLICY_METRIC_MINIMA = (
    ("lane_f1", 0.50),
    ("lane_precision", 0.45),
    ("lane_recall", 0.45),
)
QUALITY_POLICY_RATIONALE = (
    "Five-lane transform promotion requires useful held-out lane agreement and "
    "guards against a precision-only or recall-only result. The minima apply "
    "only to the immutable song-disjoint validation split."
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def quality_policy_definition() -> dict[str, object]:
    """Return the versioned policy bytes that determine a promotion decision."""
    return {
        "schema_version": 1,
        "format": QUALITY_POLICY_FORMAT,
        "policy_id": QUALITY_POLICY_ID,
        "metric_minima": dict(QUALITY_POLICY_METRIC_MINIMA),
        "rationale": QUALITY_POLICY_RATIONALE,
    }


def quality_policy_evidence() -> dict[str, object]:
    """Return the complete policy identity persisted with evaluation evidence."""
    policy = quality_policy_definition()
    return {**policy, "policy_sha256": _canonical_sha256(policy)}


def evaluate_quality_policy(metrics: object) -> dict[str, object]:
    """Recompute the non-configurable held-out quality decision."""
    values = metrics if isinstance(metrics, dict) else {}
    failed_metrics = [
        name
        for name, minimum in QUALITY_POLICY_METRIC_MINIMA
        if not isinstance(values.get(name), (int, float))
        or isinstance(values.get(name), bool)
        or not math.isfinite(float(values[name]))
        or float(values[name]) < minimum
    ]
    return {
        "status": "passed" if not failed_metrics else "failed",
        "failed_metrics": failed_metrics,
    }


def validate_quality_policy_evidence(policy: object, gate: object, metrics: object) -> bool:
    """Return whether persisted policy/gate evidence is exactly STRUM's V1 contract."""
    return policy == quality_policy_evidence() and gate == evaluate_quality_policy(metrics)
