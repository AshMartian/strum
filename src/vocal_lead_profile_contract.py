"""Fail-closed evidence contract for a future *lead-only* Vocal candidate.

This is intentionally smaller than :mod:`src.vocal_profile_contract`: it
describes ``PART VOCALS`` only and has no HARM-track inputs or outputs.  It is
not a chart runtime, model loader, event decoder, evaluator, package writer,
or deployable profile.  The four current catalog workers produce independent
experiments; a future implementation can use this module to prove that a
proposed lead composition has enough compatible, source-disjoint evidence
before it is even considered for a full Vocal profile.

The 3-song curated smoke views cannot meet this contract.  In particular,
they have no test split and the current CTC lyric component has no timestamp
decoder.  Keeping that fact executable in a path-free contract prevents a
host from treating a collection of checkpoint names as a playable chart.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_COMPONENT_IDS = frozenset(
    {
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
    }
)
_TASK_VIEW_KINDS = frozenset(
    {
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    }
)
_SPLITS = ("train", "val", "test")
_LABELS = ("pitched_note_events", "phrase_boundaries", "lyric_events", "talky_spans")


class VocalLeadCandidateContractError(ValueError):
    """Raised when future lead-only candidate evidence is incomplete or unsafe."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _policy_bytes() -> bytes:
    """Return the immutable, canonical lead-candidate quality policy bytes."""
    return (
        b'{"aggregation":{"failed_or_missing_outcome":"candidate-rejected/v1",'
        b'"per_metric":"all-required-metrics-pass/v1",'
        b'"threshold_selection":"policy-pinned-before-held-out-evaluation/v1"},'
        b'"format":"strum-vocal-lead-candidate-quality-policy/v1",'
        b'"metric_requirements":{"assembled_chart":{"part_vocals_event_coverage":'
        b'{"operator":"gte","threshold":0.95},"valid_midi":{"operator":"equals",'
        b'"threshold":true}},"lyrics":{"timestamp_alignment_error_ms":{"operator":"lte",'
        b'"threshold":100.0},"token_error_rate":{"operator":"lte","threshold":0.25}},'
        b'"phrases":{"end_f1":{"operator":"gte","threshold":0.8},'
        b'"start_f1":{"operator":"gte","threshold":0.8}},"pitched_notes":'
        b'{"offset_f1":{"operator":"gte","threshold":0.75},"onset_f1":'
        b'{"operator":"gte","threshold":0.8},"pitch_accuracy":{"operator":"gte",'
        b'"threshold":0.85}},"talkies":{"span_f1":{"operator":"gte",'
        b'"threshold":0.75}}},"policy_id":"vocal-lead-candidate-baseline-v1",'
        b'"schema_version":1}'
    )


def _data_gate_bytes() -> bytes:
    """Return immutable minimum data/coverage gates for lead composition."""
    return (
        b'{"format":"strum-vocal-lead-candidate-data-gate/v1",'
        b'"minimum_label_counts":{"test":{"lyric_events":500,"phrase_boundaries":50,'
        b'"pitched_note_events":500,"talky_spans":25},"train":{"lyric_events":2000,'
        b'"phrase_boundaries":200,"pitched_note_events":2000,"talky_spans":100},'
        b'"val":{"lyric_events":500,"phrase_boundaries":50,"pitched_note_events":500,'
        b'"talky_spans":25}},"minimum_source_counts":{"test":10,"train":40,"val":10},'
        b'"schema_version":1,"source_partition":"source-id-disjoint-train-val-test/v1"}'
    )


def _decode_canonical(payload: bytes, label: str) -> dict[str, object]:
    value = json.loads(payload.decode("utf-8"))
    if (
        not isinstance(value, dict) or _canonical(value) != payload
    ):  # pragma: no cover - static guard
        raise RuntimeError(f"STRUM {label} canonical bytes are invalid")
    return value


def vocal_lead_candidate_quality_policy_definition() -> dict[str, object]:
    """Return a fresh, immutable-content definition of the metric policy."""
    return _decode_canonical(_policy_bytes(), "Vocal lead policy")


def vocal_lead_candidate_quality_policy_identity() -> dict[str, object]:
    """Return the hash-pinned identity a held-out report must carry."""
    policy = vocal_lead_candidate_quality_policy_definition()
    return {
        "format": policy["format"],
        "policy_id": policy["policy_id"],
        "sha256": hashlib.sha256(_policy_bytes()).hexdigest(),
    }


def vocal_lead_candidate_data_gate_definition() -> dict[str, object]:
    """Return a fresh, immutable-content minimum data gate."""
    return _decode_canonical(_data_gate_bytes(), "Vocal lead data gate")


def vocal_lead_candidate_data_gate_identity() -> dict[str, object]:
    """Return the hash-pinned identity a report must carry for data admission."""
    gate = vocal_lead_candidate_data_gate_definition()
    return {
        "format": gate["format"],
        "schema_version": gate["schema_version"],
        "sha256": hashlib.sha256(_data_gate_bytes()).hexdigest(),
    }


def vocal_lead_candidate_contract_definition() -> dict[str, object]:
    """Describe the exact, planned lead-only candidate boundary.

    A successful quality result remains deliberately non-deployable.  The
    missing decoder/loader/evaluator stages must be implemented and registered
    separately; this contract is not permission to call the legacy charter.
    """
    return {
        "format": "strum-vocal-lead-candidate-contract/v1",
        "status": "planned_nondeployable",
        "scope": {"input_track": "PART VOCALS", "output_track": "PART VOCALS"},
        "components": [
            {
                "id": "vocals.frame_activity_pitch",
                "required_outputs": ["pitched_vocal_activity", "midi_pitch_36_84"],
            },
            {
                "id": "vocals.phrase_boundaries",
                "required_outputs": ["lead_phrase_start", "lead_phrase_end"],
            },
            {
                "id": "vocals.lyric_alignment",
                "required_outputs": [
                    "observed_lyric_character_tokens",
                    "observed_lyric_event_alignment",
                ],
            },
            {"id": "vocals.talky_activity", "required_outputs": ["pitchless_talky_activity"]},
        ],
        "shared_semantics": {
            "input_frontend": "logmel-16khz-160hop-80mel-500frame/v1",
            "clock": "same-master-timeline/v1",
            "component_task_lineage": "same-catalog-control-and-source-partition/v1",
            "source_label_track": "PART VOCALS",
        },
        "required_new_stages": [
            "tensor-only-vocal-component-loader/v1",
            "pitched-frame-to-note-span-decoder/v1",
            "phrase-boundary-peak-and-pair-decoder/v1",
            "ctc-lyric-timestamp-decoder/v1",
            "talky-frame-to-span-decoder/v1",
            "part-vocals-midi-assembler/v1",
            "strum-recomputed-lead-held-out-evaluator/v1",
        ],
        "catalog_admission": {
            "required_splits": list(_SPLITS),
            "data_gate": {
                **vocal_lead_candidate_data_gate_identity(),
                "definition": vocal_lead_candidate_data_gate_definition(),
            },
            "test_source_ids_forbidden_in_training_or_calibration": True,
        },
        "held_out_evaluation": {
            "format": "strum-vocal-lead-held-out-evaluation-report/v1",
            "recomputed_by": "strum",
            "quality_policy": {
                **vocal_lead_candidate_quality_policy_identity(),
                "definition": vocal_lead_candidate_quality_policy_definition(),
            },
            "required_metrics": {
                "pitched_notes": ["onset_f1", "offset_f1", "pitch_accuracy"],
                "phrases": ["start_f1", "end_f1"],
                "lyrics": ["token_error_rate", "timestamp_alignment_error_ms"],
                "talkies": ["span_f1"],
                "assembled_chart": ["valid_midi", "part_vocals_event_coverage"],
            },
        },
        "deployment": {
            "status": "not_deployable",
            "profile_package": "forbidden-until-full-vocal-profile-contract/v1",
            "chart_execution": "not_available",
            "fallback": "forbidden",
        },
        "forbidden_shortcuts": [
            "raw-component-as-profile",
            "component-substitution-by-name-or-filename",
            "external-lyrics-as-chart-labels",
            "legacy-vocals-charter-fallback",
            "implicit-harmony-output",
        ],
    }


def _require_exact_keys(raw: object, keys: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(raw, Mapping) or set(raw) != keys:
        raise VocalLeadCandidateContractError(f"{label} contains unsupported or missing fields")
    return raw


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise VocalLeadCandidateContractError(f"{label} must be a SHA-256")
    return value


def _require_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VocalLeadCandidateContractError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise VocalLeadCandidateContractError(f"{label} must be a finite number")
    return result


def _require_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise VocalLeadCandidateContractError(f"{label} must be a non-negative integer")
    return value


def _metric_outcome(value: object, rule: Mapping[str, object], label: str) -> dict[str, object]:
    operator, threshold = rule.get("operator"), rule.get("threshold")
    if operator == "equals":
        if not isinstance(threshold, bool):  # pragma: no cover - policy guard
            raise RuntimeError("STRUM Vocal lead equality policy is invalid")
        if not isinstance(value, bool):
            raise VocalLeadCandidateContractError(f"{label} must be a boolean")
        return {
            "observed": value,
            "operator": operator,
            "threshold": threshold,
            "passed": value == threshold,
        }
    observed = _require_number(value, label)
    numeric_threshold = _require_number(threshold, f"policy threshold {label}")
    if operator == "gte":
        passed = observed >= numeric_threshold
    elif operator == "lte":
        passed = observed <= numeric_threshold
    else:  # pragma: no cover - policy guard
        raise RuntimeError("STRUM Vocal lead metric operator is invalid")
    return {
        "observed": observed,
        "operator": operator,
        "threshold": numeric_threshold,
        "passed": passed,
    }


def _require_policy_identity(raw: object) -> None:
    identity = _require_exact_keys(raw, {"format", "policy_id", "sha256"}, "quality policy")
    if dict(identity) != vocal_lead_candidate_quality_policy_identity():
        raise VocalLeadCandidateContractError("quality policy is not the pinned STRUM lead policy")


def _require_data_gate_identity(raw: object) -> None:
    identity = _require_exact_keys(raw, {"format", "schema_version", "sha256"}, "data gate")
    if dict(identity) != vocal_lead_candidate_data_gate_identity():
        raise VocalLeadCandidateContractError("data gate is not the pinned STRUM lead gate")


def _require_component_hashes(raw: object, label: str) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != _COMPONENT_IDS:
        raise VocalLeadCandidateContractError(f"{label} is incomplete")
    return {
        component: _require_sha(raw[component], f"{label} {component}")
        for component in _COMPONENT_IDS
    }


def _require_task_hashes(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping) or set(raw) != _TASK_VIEW_KINDS:
        raise VocalLeadCandidateContractError("lead task-view evidence is incomplete")
    return {kind: _require_sha(raw[kind], f"task view hash {kind}") for kind in _TASK_VIEW_KINDS}


def _evaluate_data_coverage(raw: object) -> dict[str, object]:
    coverage = _require_exact_keys(raw, {"source_counts", "label_counts"}, "data coverage")
    source_counts = coverage["source_counts"]
    label_counts = coverage["label_counts"]
    if not isinstance(source_counts, Mapping) or set(source_counts) != set(_SPLITS):
        raise VocalLeadCandidateContractError("source coverage must name train, val, and test")
    if not isinstance(label_counts, Mapping) or set(label_counts) != set(_SPLITS):
        raise VocalLeadCandidateContractError("label coverage must name train, val, and test")
    gate = vocal_lead_candidate_data_gate_definition()
    minimum_sources = gate["minimum_source_counts"]
    minimum_labels = gate["minimum_label_counts"]
    assert isinstance(minimum_sources, Mapping) and isinstance(minimum_labels, Mapping)
    source_outcomes: dict[str, dict[str, object]] = {}
    label_outcomes: dict[str, dict[str, dict[str, object]]] = {}
    for split in _SPLITS:
        count = _require_count(source_counts[split], f"source count {split}")
        required_source_count = _require_count(
            minimum_sources[split], f"minimum source count {split}"
        )
        source_outcomes[split] = {
            "observed": count,
            "minimum": required_source_count,
            "passed": count >= required_source_count,
        }
        split_labels = label_counts[split]
        expected_labels = minimum_labels[split]
        if not isinstance(split_labels, Mapping) or set(split_labels) != set(_LABELS):
            raise VocalLeadCandidateContractError(f"label coverage {split} is incomplete")
        if not isinstance(expected_labels, Mapping):  # pragma: no cover - policy guard
            raise RuntimeError("STRUM Vocal lead data policy is invalid")
        label_outcomes[split] = {}
        for label in _LABELS:
            count = _require_count(split_labels[label], f"label count {split}.{label}")
            minimum = _require_count(expected_labels[label], f"minimum label count {split}.{label}")
            label_outcomes[split][label] = {
                "observed": count,
                "minimum": minimum,
                "passed": count >= minimum,
            }
    passed = all(item["passed"] for item in source_outcomes.values()) and all(
        item["passed"] for split in label_outcomes.values() for item in split.values()
    )
    return {"source_counts": source_outcomes, "label_counts": label_outcomes, "passed": passed}


def _evaluate_metrics(raw: object) -> dict[str, dict[str, dict[str, object]]]:
    metrics = _require_exact_keys(
        raw, {"pitched_notes", "phrases", "lyrics", "talkies"}, "lead metrics"
    )
    policy = vocal_lead_candidate_quality_policy_definition()["metric_requirements"]
    assert isinstance(policy, Mapping)
    outcomes: dict[str, dict[str, dict[str, object]]] = {}
    for group in ("pitched_notes", "phrases", "lyrics", "talkies"):
        values, rules = metrics[group], policy[group]
        if (
            not isinstance(values, Mapping)
            or not isinstance(rules, Mapping)
            or set(values) != set(rules)
        ):
            raise VocalLeadCandidateContractError(f"lead metric group {group} is incomplete")
        outcomes[group] = {
            key: _metric_outcome(values[key], rules[key], f"{group}.{key}") for key in rules
        }
    return outcomes


def evaluate_vocal_lead_candidate_report(report: Mapping[str, object]) -> dict[str, object]:
    """Recompute non-deployable lead-candidate admission and metric outcomes.

    The future evaluator must obtain all input data from revalidated catalog
    assets; this validator accepts no paths and never loads a checkpoint.
    Passing outcomes are evidence for a later implementation, *not* profile
    packaging or chart execution authority.
    """
    required = {
        "format",
        "quality_policy",
        "data_gate",
        "evidence",
        "data_coverage",
        "metrics",
        "assembled_chart",
    }
    if not isinstance(report, Mapping) or set(report) != required:
        raise VocalLeadCandidateContractError(
            "lead candidate report contains unsupported or missing fields"
        )
    if report["format"] != "strum-vocal-lead-held-out-evaluation-report/v1":
        raise VocalLeadCandidateContractError("lead candidate report format is invalid")
    _require_policy_identity(report["quality_policy"])
    _require_data_gate_identity(report["data_gate"])
    evidence = _require_exact_keys(
        report["evidence"],
        {
            "component_hashes",
            "component_configuration_hashes",
            "catalog_control_sha256",
            "task_view_hashes",
            "split_source_ids_sha256",
            "source_partition",
        },
        "lead evidence",
    )
    component_hashes = _require_component_hashes(evidence["component_hashes"], "component hashes")
    configuration_hashes = _require_component_hashes(
        evidence["component_configuration_hashes"], "component configuration hashes"
    )
    catalog_control = _require_sha(evidence["catalog_control_sha256"], "catalog control")
    task_view_hashes = _require_task_hashes(evidence["task_view_hashes"])
    split_hashes = evidence["split_source_ids_sha256"]
    if not isinstance(split_hashes, Mapping) or set(split_hashes) != set(_SPLITS):
        raise VocalLeadCandidateContractError("split source evidence is incomplete")
    split_source_hashes = {
        split: _require_sha(split_hashes[split], f"source IDs {split}") for split in _SPLITS
    }
    if len(set(split_source_hashes.values())) != len(_SPLITS):
        raise VocalLeadCandidateContractError("split source evidence must be distinct")
    if evidence["source_partition"] != "source-id-disjoint-train-val-test/v1":
        raise VocalLeadCandidateContractError("lead source partition is invalid")
    data_outcomes = _evaluate_data_coverage(report["data_coverage"])
    metric_outcomes = _evaluate_metrics(report["metrics"])
    assembled = _require_exact_keys(
        report["assembled_chart"],
        {"valid_midi", "part_vocals_event_coverage"},
        "assembled lead chart",
    )
    policy = vocal_lead_candidate_quality_policy_definition()["metric_requirements"]
    assert isinstance(policy, Mapping) and isinstance(policy["assembled_chart"], Mapping)
    assembled_outcomes = {
        key: _metric_outcome(
            assembled[key], policy["assembled_chart"][key], f"assembled_chart.{key}"
        )
        for key in policy["assembled_chart"]
    }
    quality_passed = all(
        item["passed"] for group in metric_outcomes.values() for item in group.values()
    ) and all(item["passed"] for item in assembled_outcomes.values())
    return {
        "format": "strum-vocal-lead-candidate-outcomes/v1",
        "quality_policy": vocal_lead_candidate_quality_policy_identity(),
        "data_gate": vocal_lead_candidate_data_gate_identity(),
        "evidence": {
            "component_hashes": component_hashes,
            "component_configuration_hashes": configuration_hashes,
            "catalog_control_sha256": catalog_control,
            "task_view_hashes": task_view_hashes,
            "split_source_ids_sha256": split_source_hashes,
        },
        "data_admission": data_outcomes,
        "quality": {
            "metrics": metric_outcomes,
            "assembled_chart": assembled_outcomes,
            "passed": quality_passed,
        },
        "candidate": {
            "status": "not_deployable",
            "profile_packaging": "forbidden-until-full-vocal-profile-contract/v1",
            "chart_execution": "not_available",
            "fallback": "forbidden",
        },
        "aggregation": {
            "rule": "data-gate-and-all-required-quality-metrics-pass/v1",
            "passed": bool(data_outcomes["passed"] and quality_passed),
        },
    }
