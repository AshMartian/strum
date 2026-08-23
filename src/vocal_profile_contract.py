"""Fail-closed contracts for a future composed Vocal chart profile.

This module deliberately implements *contract verification*, not a Vocal
trainer, evaluator, package writer, or chart runtime.  The existing bounded
lead-component experiments and the OCTAVE-produced Harmony source task are
not interchangeable with an executable profile.  When those future stages
arrive they must call these validators before accepting a held-out report or
producing a package.

All identities here are opaque SHA-256 values.  In particular, a Harmony
binding records the selected source-task view and the OCTAVE sidecar policy by
content hash; it never records a catalog path or private asset location.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence

from src.vocal_harmony_catalog import (
    HARMONY_SOURCE_POLICY_FORMAT,
    HARMONY_SOURCE_TASK_FORMAT,
    HARMONY_TRACK_ROLES,
)

VOCAL_PROFILE_QUALITY_POLICY_FORMAT = "strum-vocal-profile-quality-policy/v1"
VOCAL_HELD_OUT_REPORT_FORMAT = "strum-vocal-held-out-chart-evaluation-report/v1"
_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_HARMONY_TRACKS = tuple(HARMONY_TRACK_ROLES)
_REQUIRED_COMPONENTS = frozenset(
    {
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
        "vocals.harmony_model",
    }
)
_REQUIRED_TASK_VIEW_HASHES = frozenset(
    {
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
        "vocals_harmony_source_policy",
    }
)


class VocalProfileContractError(ValueError):
    """Raised when future Vocal profile evidence is incomplete or unsafe."""


# This is intentionally a STRUM-owned, versioned policy rather than a host
# option.  A profile package must pin the canonical content hash below and the
# evaluator must produce every outcome; callers cannot substitute a policy
# that drops inconvenient metrics after seeing the held-out split.
_VOCAL_PROFILE_QUALITY_POLICY: dict[str, object] = {
    "schema_version": 1,
    "format": VOCAL_PROFILE_QUALITY_POLICY_FORMAT,
    "policy_id": "vocal-chart-baseline-quality-v1",
    "metric_requirements": {
        "pitched_notes": {
            "onset_f1": {"operator": "gte", "threshold": 0.80},
            "offset_f1": {"operator": "gte", "threshold": 0.75},
            "pitch_accuracy": {"operator": "gte", "threshold": 0.85},
        },
        "phrases": {
            "start_f1": {"operator": "gte", "threshold": 0.80},
            "end_f1": {"operator": "gte", "threshold": 0.80},
        },
        "lyrics": {
            "token_error_rate": {"operator": "lte", "threshold": 0.25},
            "timestamp_alignment_error_ms": {"operator": "lte", "threshold": 100.0},
        },
        "talkies": {"span_f1": {"operator": "gte", "threshold": 0.75}},
        "harmony": {"track_specific_note_f1": {"operator": "gte", "threshold": 0.75}},
        "assembled_chart": {
            "valid_midi": {"operator": "equals", "threshold": True},
            "per_track_event_coverage": {"operator": "gte", "threshold": 0.95},
        },
    },
    "aggregation": {
        "per_metric": "all-required-metrics-pass/v1",
        "harmony": "all-selected-approved-tracks-pass/v1",
        "assembled_chart": "lead-and-every-selected-harmony-track-pass/v1",
        "minimum_selected_harmony_tracks": 1,
        "threshold_selection": "policy-pinned-before-held-out-evaluation/v1",
        "failed_or_missing_outcome": "package-rejected/v1",
    },
}


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


_VOCAL_PROFILE_QUALITY_POLICY_CANONICAL = _canonical(_VOCAL_PROFILE_QUALITY_POLICY)
VOCAL_PROFILE_QUALITY_POLICY_SHA256 = hashlib.sha256(
    _VOCAL_PROFILE_QUALITY_POLICY_CANONICAL.encode("utf-8")
).hexdigest()


def vocal_profile_quality_policy_definition() -> dict[str, object]:
    """Return a fresh copy of STRUM's canonical, hash-pinned policy.

    The evaluator reads this canonical representation rather than a mutable
    module dictionary.  That prevents an in-process caller from changing
    thresholds after the policy identity has been published to a host.
    """
    value = json.loads(_VOCAL_PROFILE_QUALITY_POLICY_CANONICAL)
    assert isinstance(value, dict)
    return value


def vocal_profile_quality_policy_identity() -> dict[str, object]:
    """Return the safe immutable identity hosts and reports must pin."""
    # Never read identity data from the mutable module-level source object.
    # The canonical JSON was fixed when this module loaded; decode it anew so
    # an in-process caller cannot make the published ID disagree with the
    # pinned policy hash.
    policy = vocal_profile_quality_policy_definition()
    policy_id = policy["policy_id"]
    assert isinstance(policy_id, str)
    return {
        "format": VOCAL_PROFILE_QUALITY_POLICY_FORMAT,
        "policy_id": policy_id,
        "sha256": VOCAL_PROFILE_QUALITY_POLICY_SHA256,
    }


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise VocalProfileContractError(f"{label} must be a SHA-256")
    return value


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise VocalProfileContractError(f"{label} must be a non-empty string")
    return value


def _require_exact_keys(raw: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise VocalProfileContractError(f"{label} contains unsupported or missing fields")
    return raw


def _require_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VocalProfileContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VocalProfileContractError(f"{label} must be a finite number")
    return result


def _require_quality_policy_identity(raw: object) -> dict[str, object]:
    identity = _require_exact_keys(raw, {"format", "policy_id", "sha256"}, "quality policy")
    expected = vocal_profile_quality_policy_identity()
    if dict(identity) != expected:
        raise VocalProfileContractError("quality policy is not the pinned STRUM policy")
    return expected


def validate_vocal_harmony_bindings(
    selected_tracks: Sequence[str], bindings: Sequence[Mapping[str, object]]
) -> dict[str, object]:
    """Validate a non-empty approved HARM subset and its source-task lineage.

    A profile can elect to emit HARM1 alone, HARM1/HARM3, or all three.  It may
    not silently invent an omitted HARM track, collapse selected tracks onto a
    common stem, or bind a HARM track to a task view/policy different from the
    selected approved Harmony task.
    """
    if isinstance(selected_tracks, (str, bytes)) or not isinstance(selected_tracks, Sequence):
        raise VocalProfileContractError("selected harmony tracks must be an array")
    tracks = tuple(selected_tracks)
    if (
        not tracks
        or len(set(tracks)) != len(tracks)
        or any(track not in _HARMONY_TRACKS for track in tracks)
    ):
        raise VocalProfileContractError(
            "selected harmony tracks must be a unique non-empty HARM subset"
        )
    if isinstance(bindings, (str, bytes)) or not isinstance(bindings, Sequence):
        raise VocalProfileContractError("harmony bindings must be an array")
    if len(bindings) != len(tracks):
        raise VocalProfileContractError(
            "harmony bindings must cover exactly the selected HARM subset"
        )

    seen_tracks: set[str] = set()
    shared_task_identity: tuple[str, str, str, str, tuple[str, ...]] | None = None
    normalized: list[dict[str, object]] = []
    for raw in bindings:
        binding = _require_exact_keys(
            raw,
            {"track_name", "audio_role", "source_task"},
            "harmony binding",
        )
        track = binding["track_name"]
        role = binding["audio_role"]
        if track not in tracks or track in seen_tracks:
            raise VocalProfileContractError(
                "harmony bindings must map each selected HARM track once"
            )
        if role != HARMONY_TRACK_ROLES[track]:
            raise VocalProfileContractError(
                "harmony binding audio role does not match its HARM track"
            )
        source_task = _require_exact_keys(
            binding["source_task"],
            {
                "format",
                "task_view_sha256",
                "source_policy_format",
                "source_policy_sha256",
                "catalog_control_sha256",
                "harmony_tracks",
            },
            "harmony source task identity",
        )
        if source_task["format"] != HARMONY_SOURCE_TASK_FORMAT:
            raise VocalProfileContractError(
                "harmony binding requires the approved STRUM source task"
            )
        if source_task["source_policy_format"] != HARMONY_SOURCE_POLICY_FORMAT:
            raise VocalProfileContractError(
                "harmony binding requires the approved OCTAVE source policy"
            )
        task_tracks_raw = source_task["harmony_tracks"]
        if (
            not isinstance(task_tracks_raw, list)
            or tuple(task_tracks_raw) != tracks
            or len(set(task_tracks_raw)) != len(task_tracks_raw)
        ):
            raise VocalProfileContractError(
                "harmony source task subset does not match selected HARM tracks"
            )
        task_identity = (
            _require_sha(source_task["task_view_sha256"], "harmony source task view"),
            _require_sha(source_task["source_policy_sha256"], "harmony source policy"),
            _require_sha(source_task["catalog_control_sha256"], "harmony catalog control"),
            _require_string(source_task["format"], "harmony source task format"),
            tuple(task_tracks_raw),
        )
        if shared_task_identity is None:
            shared_task_identity = task_identity
        elif shared_task_identity != task_identity:
            raise VocalProfileContractError(
                "all selected HARM tracks must bind the same approved source task identity"
            )
        seen_tracks.add(track)
        normalized.append(
            {
                "track_name": track,
                "audio_role": role,
                # Construct the exact canonical identity from validated
                # scalar values.  ``dict(source_task)`` shallow-copied its
                # nested ``harmony_tracks`` list, letting a caller mutate
                # already-validated evidence after this function returned.
                "source_task": {
                    "format": task_identity[3],
                    "task_view_sha256": task_identity[0],
                    "source_policy_format": HARMONY_SOURCE_POLICY_FORMAT,
                    "source_policy_sha256": task_identity[1],
                    "catalog_control_sha256": task_identity[2],
                    "harmony_tracks": list(task_identity[4]),
                },
            }
        )
    if seen_tracks != set(tracks):
        raise VocalProfileContractError("harmony bindings omit a selected HARM track")
    return {
        "selected_tracks": list(tracks),
        "bindings": sorted(normalized, key=lambda item: str(item["track_name"])),
    }


def _metric_outcome(
    value: object, requirement: Mapping[str, object], label: str
) -> dict[str, object]:
    operator = requirement["operator"]
    threshold = requirement["threshold"]
    if operator == "equals":
        if not isinstance(value, bool) or not isinstance(threshold, bool):
            raise VocalProfileContractError(f"{label} must be a boolean")
        passed = value is threshold
        observed: bool | float = value
    else:
        observed = _require_finite_number(value, label)
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise VocalProfileContractError(f"quality policy {label} threshold is invalid")
        if operator == "gte":
            passed = observed >= float(threshold)
        elif operator == "lte":
            passed = observed <= float(threshold)
        else:  # Static STRUM policy above is controlled; fail closed if edited incorrectly.
            raise VocalProfileContractError(f"quality policy {label} operator is invalid")
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "passed": passed,
    }


def _evaluate_metric_group(
    values: object, requirements: Mapping[str, object], label: str
) -> dict[str, dict[str, object]]:
    if not isinstance(values, Mapping) or set(values) != set(requirements):
        raise VocalProfileContractError(
            f"{label} metric evidence is incomplete or contains unsupported fields"
        )
    outcomes: dict[str, dict[str, object]] = {}
    for name, requirement in requirements.items():
        if not isinstance(requirement, Mapping):
            raise VocalProfileContractError(f"quality policy {label} requirement is invalid")
        outcomes[name] = _metric_outcome(values[name], requirement, f"{label}.{name}")
    return outcomes


def _require_report_evidence(raw: object) -> dict[str, object]:
    evidence = _require_exact_keys(
        raw,
        {
            "component_hashes",
            "component_configuration_hashes",
            "catalog_control_sha256",
            "task_view_hashes",
            "test_source_ids_sha256",
        },
        "Vocal held-out report evidence",
    )
    component_hashes = evidence["component_hashes"]
    configuration_hashes = evidence["component_configuration_hashes"]
    if (
        not isinstance(component_hashes, Mapping)
        or not isinstance(configuration_hashes, Mapping)
        or set(component_hashes) != _REQUIRED_COMPONENTS
        or set(configuration_hashes) != _REQUIRED_COMPONENTS
    ):
        raise VocalProfileContractError("Vocal report component evidence is incomplete")
    for component in _REQUIRED_COMPONENTS:
        _require_sha(component_hashes[component], f"component hash {component}")
        _require_sha(configuration_hashes[component], f"component configuration hash {component}")
    task_view_hashes = evidence["task_view_hashes"]
    if (
        not isinstance(task_view_hashes, Mapping)
        or set(task_view_hashes) != _REQUIRED_TASK_VIEW_HASHES
    ):
        raise VocalProfileContractError("Vocal report task-view evidence is incomplete")
    for task_kind in _REQUIRED_TASK_VIEW_HASHES:
        _require_sha(task_view_hashes[task_kind], f"task view hash {task_kind}")
    return {
        "component_hashes": dict(component_hashes),
        "component_configuration_hashes": dict(configuration_hashes),
        "catalog_control_sha256": _require_sha(
            evidence["catalog_control_sha256"], "Vocal report catalog control"
        ),
        "task_view_hashes": dict(task_view_hashes),
        "test_source_ids_sha256": _require_sha(
            evidence["test_source_ids_sha256"], "Vocal report test source IDs"
        ),
    }


def evaluate_vocal_profile_quality_report(report: Mapping[str, object]) -> dict[str, object]:
    """Recompute policy outcomes for a future STRUM-held-out Vocal report.

    This rejects partial evidence and recomputes `passed` rather than trusting
    a caller-supplied aggregate.  It intentionally has no filesystem or model
    runtime inputs: the future evaluator must materialize and hash its own
    private catalog assets before constructing this path-free report.
    """
    required = {
        "format",
        "quality_policy",
        "evidence",
        "harmony",
        "metrics",
        "assembled_chart",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise VocalProfileContractError(
            "Vocal held-out report contains unsupported or missing fields"
        )
    if report["format"] != VOCAL_HELD_OUT_REPORT_FORMAT:
        raise VocalProfileContractError("Vocal held-out report format is invalid")
    _require_quality_policy_identity(report["quality_policy"])
    report_evidence = _require_report_evidence(report["evidence"])

    harmony = _require_exact_keys(
        report["harmony"], {"selected_tracks", "bindings", "metrics"}, "harmony evidence"
    )
    bindings = validate_vocal_harmony_bindings(harmony["selected_tracks"], harmony["bindings"])
    selected_tracks = bindings["selected_tracks"]
    assert isinstance(selected_tracks, list)
    binding_source_task = bindings["bindings"][0]["source_task"]
    assert isinstance(binding_source_task, Mapping)
    if (
        binding_source_task["catalog_control_sha256"] != report_evidence["catalog_control_sha256"]
        or binding_source_task["task_view_sha256"]
        != report_evidence["task_view_hashes"]["vocals_harmony_source_policy"]
    ):
        raise VocalProfileContractError(
            "Harmony source task identity does not match report catalog/task-view evidence"
        )
    policy = vocal_profile_quality_policy_definition()
    policy_metrics = policy["metric_requirements"]
    assert isinstance(policy_metrics, Mapping)
    harmony_rows = harmony["metrics"]
    if not isinstance(harmony_rows, list) or len(harmony_rows) != len(selected_tracks):
        raise VocalProfileContractError(
            "Harmony per-track metric evidence must cover the selected subset"
        )
    binding_by_track = {
        str(item["track_name"]): item for item in bindings["bindings"] if isinstance(item, Mapping)
    }
    harmony_outcomes: dict[str, dict[str, object]] = {}
    for row in harmony_rows:
        item = _require_exact_keys(
            row, {"track_name", "audio_role", "source_task", "metrics"}, "Harmony metric evidence"
        )
        track = item["track_name"]
        if track not in binding_by_track or track in harmony_outcomes:
            raise VocalProfileContractError(
                "Harmony metric evidence must bind each selected HARM track once"
            )
        expected_binding = binding_by_track[str(track)]
        if (
            item["audio_role"] != expected_binding["audio_role"]
            or item["source_task"] != expected_binding["source_task"]
        ):
            raise VocalProfileContractError(
                "Harmony metric evidence does not match its approved track binding"
            )
        harmony_outcomes[str(track)] = _evaluate_metric_group(
            item["metrics"], policy_metrics["harmony"], f"harmony.{track}"
        )
    if set(harmony_outcomes) != set(selected_tracks):
        raise VocalProfileContractError("Harmony metric evidence omits a selected HARM track")

    metrics = report["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != {
        "pitched_notes",
        "phrases",
        "lyrics",
        "talkies",
    }:
        raise VocalProfileContractError(
            "lead Vocal metric evidence is incomplete or contains unsupported fields"
        )
    lead_outcomes = {
        group: _evaluate_metric_group(metrics[group], policy_metrics[group], group)
        for group in ("pitched_notes", "phrases", "lyrics", "talkies")
    }
    assembled = _require_exact_keys(
        report["assembled_chart"],
        {"valid_midi", "per_track_event_coverage"},
        "assembled chart evidence",
    )
    coverage = assembled["per_track_event_coverage"]
    expected_coverage_tracks = {"PART VOCALS", *selected_tracks}
    if not isinstance(coverage, Mapping) or set(coverage) != expected_coverage_tracks:
        raise VocalProfileContractError(
            "assembled chart coverage must bind lead and exactly selected HARM tracks"
        )
    assembled_outcomes = {
        "valid_midi": _metric_outcome(
            assembled["valid_midi"],
            policy_metrics["assembled_chart"]["valid_midi"],
            "assembled_chart.valid_midi",
        ),
        "per_track_event_coverage": {
            str(track): _metric_outcome(
                coverage[track],
                policy_metrics["assembled_chart"]["per_track_event_coverage"],
                f"assembled_chart.per_track_event_coverage.{track}",
            )
            for track in sorted(expected_coverage_tracks)
        },
    }
    passed = (
        all(item["passed"] for group in lead_outcomes.values() for item in group.values())
        and all(item["passed"] for group in harmony_outcomes.values() for item in group.values())
        and bool(assembled_outcomes["valid_midi"]["passed"])
        and all(item["passed"] for item in assembled_outcomes["per_track_event_coverage"].values())
    )
    return {
        "format": "strum-vocal-profile-quality-outcomes/v1",
        "quality_policy": vocal_profile_quality_policy_identity(),
        "evidence": report_evidence,
        "harmony": {
            "selected_tracks": selected_tracks,
            "bindings": bindings["bindings"],
            "per_track": harmony_outcomes,
            "aggregation": "all-selected-approved-tracks-pass/v1",
        },
        "lead": lead_outcomes,
        "assembled_chart": assembled_outcomes,
        "aggregation": {
            "rule": "all-required-metrics-pass/v1",
            "passed": passed,
        },
    }


def require_vocal_profile_package_evidence(report: Mapping[str, object]) -> dict[str, object]:
    """Fail closed for future Vocal package code until every policy outcome passes."""
    outcomes = evaluate_vocal_profile_quality_report(report)
    aggregation = outcomes["aggregation"]
    assert isinstance(aggregation, Mapping)
    if aggregation["passed"] is not True:
        raise VocalProfileContractError(
            "Vocal profile package requires a passing STRUM quality-policy report"
        )
    return outcomes
