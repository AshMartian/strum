from __future__ import annotations

import copy

import pytest

import src.vocal_lead_profile_contract as contract
from src.vocal_lead_profile_contract import (
    VocalLeadCandidateContractError,
    evaluate_vocal_lead_candidate_report,
    vocal_lead_candidate_contract_definition,
    vocal_lead_candidate_data_gate_identity,
    vocal_lead_candidate_quality_policy_identity,
)


def _coverage() -> dict[str, object]:
    return {
        "source_counts": {"train": 40, "val": 10, "test": 10},
        "label_counts": {
            "train": {
                "pitched_note_events": 2000,
                "phrase_boundaries": 200,
                "lyric_events": 2000,
                "talky_spans": 100,
            },
            "val": {
                "pitched_note_events": 500,
                "phrase_boundaries": 50,
                "lyric_events": 500,
                "talky_spans": 25,
            },
            "test": {
                "pitched_note_events": 500,
                "phrase_boundaries": 50,
                "lyric_events": 500,
                "talky_spans": 25,
            },
        },
    }


def _passing_report() -> dict[str, object]:
    components = [
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
    ]
    tasks = [
        "vocals_activity",
        "vocals_phrase_boundaries",
        "vocals_lyric_alignment",
        "vocals_talky_activity",
    ]
    return {
        "format": "strum-vocal-lead-held-out-evaluation-report/v1",
        "quality_policy": vocal_lead_candidate_quality_policy_identity(),
        "data_gate": vocal_lead_candidate_data_gate_identity(),
        "evidence": {
            "component_hashes": {
                name: f"{index:x}" * 64 for index, name in enumerate(components, 1)
            },
            "component_configuration_hashes": {
                name: f"{index:x}" * 64 for index, name in enumerate(components, 5)
            },
            "catalog_control_sha256": "9" * 64,
            "task_view_hashes": {name: f"{index:x}" * 64 for index, name in enumerate(tasks, 1)},
            "split_source_ids_sha256": {"train": "a" * 64, "val": "b" * 64, "test": "c" * 64},
            "source_partition": "source-id-disjoint-train-val-test/v1",
        },
        "data_coverage": _coverage(),
        "metrics": {
            "pitched_notes": {"onset_f1": 0.8, "offset_f1": 0.75, "pitch_accuracy": 0.85},
            "phrases": {"start_f1": 0.8, "end_f1": 0.8},
            "lyrics": {"token_error_rate": 0.25, "timestamp_alignment_error_ms": 100.0},
            "talkies": {"span_f1": 0.75},
        },
        "assembled_chart": {"valid_midi": True, "part_vocals_event_coverage": 0.95},
    }


def test_contract_is_lead_only_planned_and_cannot_deploy() -> None:
    definition = vocal_lead_candidate_contract_definition()

    assert definition["status"] == "planned_nondeployable"
    assert definition["scope"] == {"input_track": "PART VOCALS", "output_track": "PART VOCALS"}
    assert [item["id"] for item in definition["components"]] == [
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
    ]
    assert "ctc-lyric-timestamp-decoder/v1" in definition["required_new_stages"]
    assert (
        "strum-owned-lead-catalog-task-admission-resolver/v1" in definition["required_new_stages"]
    )
    assert definition["catalog_admission"]["status"] == (
        "not_available_without-strum-catalog-task-revalidation/v1"
    )
    assert definition["deployment"] == {
        "status": "not_deployable",
        "profile_package": "forbidden-until-full-vocal-profile-contract/v1",
        "chart_execution": "not_available",
        "fallback": "forbidden",
    }
    assert all("harm" not in item["id"].lower() for item in definition["components"])
    assert "implicit-harmony-output" in definition["forbidden_shortcuts"]


def test_synthetic_passing_report_is_schema_only_and_never_admits() -> None:
    outcomes = evaluate_vocal_lead_candidate_report(_passing_report())

    assert outcomes["report_validation"]["status"] == "schema-only-untrusted-report/v1"
    assert outcomes["report_validation"]["reported_data_coverage"]["passed"] is True
    assert outcomes["data_admission"] == {
        "status": "unavailable-without-strum-catalog-task-revalidation/v1",
        "passed": False,
        "reason": "caller-claimed-counts-and-hashes-are-not-catalog-admission-evidence/v1",
    }
    assert outcomes["quality"]["reported_thresholds_met"] is True
    assert outcomes["aggregation"] == {
        "rule": "public-report-schema-validation-never-admits/v1",
        "passed": False,
        "reason": "strum-owned-catalog-task-admission-resolver-not-implemented/v1",
    }
    assert outcomes["candidate"] == {
        "status": "not_deployable",
        "profile_packaging": "forbidden-until-full-vocal-profile-contract/v1",
        "chart_execution": "not_available",
        "fallback": "forbidden",
    }


def test_three_song_smoke_style_claims_are_reported_but_never_admit() -> None:
    report = _passing_report()
    report["data_coverage"] = {
        "source_counts": {"train": 1, "val": 2, "test": 0},
        "label_counts": {
            split: {
                "pitched_note_events": 1,
                "phrase_boundaries": 1,
                "lyric_events": 1,
                "talky_spans": 1,
            }
            for split in ("train", "val", "test")
        },
    }

    outcomes = evaluate_vocal_lead_candidate_report(report)

    reported = outcomes["report_validation"]["reported_data_coverage"]
    assert reported["passed"] is False
    assert outcomes["aggregation"]["passed"] is False
    assert reported["source_counts"]["test"] == {
        "observed": 0,
        "minimum": 10,
        "passed": False,
    }


def test_distinct_synthetic_split_hashes_cannot_establish_source_disjoint_admission() -> None:
    report = _passing_report()
    report["evidence"]["split_source_ids_sha256"] = {
        "train": "d" * 64,
        "val": "e" * 64,
        "test": "f" * 64,
    }

    outcomes = evaluate_vocal_lead_candidate_report(report)

    assert outcomes["report_validation"]["reported_split_source_ids_sha256"] == {
        "train": "d" * 64,
        "val": "e" * 64,
        "test": "f" * 64,
    }
    assert outcomes["data_admission"]["passed"] is False
    assert outcomes["aggregation"]["passed"] is False


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda report: report["evidence"].__setitem__("source_partition", "by-window/v1"),
            "source partition",
        ),
        (
            lambda report: report["evidence"]["split_source_ids_sha256"].__setitem__(
                "test", "a" * 64
            ),
            "must be distinct",
        ),
        (
            lambda report: report["data_coverage"]["label_counts"]["test"].pop("talky_spans"),
            "label coverage test",
        ),
        (
            lambda report: report["metrics"]["lyrics"].__setitem__(
                "token_error_rate", float("nan")
            ),
            "finite number",
        ),
        (
            lambda report: report.__setitem__("harmony", {}),
            "unsupported or missing",
        ),
    ],
)
def test_report_rejects_forged_or_incomplete_evidence(mutate: object, message: str) -> None:
    report = copy.deepcopy(_passing_report())
    assert callable(mutate)
    mutate(report)

    with pytest.raises(VocalLeadCandidateContractError, match=message):
        evaluate_vocal_lead_candidate_report(report)


def test_policy_and_gate_are_immune_to_mutable_legacy_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_policy = vocal_lead_candidate_quality_policy_identity()
    expected_gate = vocal_lead_candidate_data_gate_identity()
    monkeypatch.setattr(contract, "VOCAL_LEAD_POLICY", {"sha256": "f" * 64}, raising=False)
    monkeypatch.setattr(contract, "VOCAL_LEAD_DATA_GATE", {"sha256": "e" * 64}, raising=False)

    outcomes = evaluate_vocal_lead_candidate_report(_passing_report())

    assert outcomes["quality_policy"] == expected_policy
    assert outcomes["data_gate"] == expected_gate
