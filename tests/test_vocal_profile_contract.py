from __future__ import annotations

import copy
import json

import pytest

import src.vocal_profile_contract as vocal_profile_contract
from src.vocal_profile_contract import (
    VOCAL_HELD_OUT_REPORT_FORMAT,
    VocalProfileContractError,
    evaluate_vocal_profile_quality_report,
    require_vocal_profile_package_evidence,
    validate_vocal_harmony_bindings,
    vocal_profile_quality_policy_definition,
    vocal_profile_quality_policy_identity,
)


def _source_task(tracks: list[str]) -> dict[str, object]:
    return {
        "format": "strum-vocal-harmony-source-task/v1",
        "task_view_sha256": "a" * 64,
        "source_policy_format": "octave-vocal-harmony-source-policy/v1",
        "source_policy_sha256": "b" * 64,
        "catalog_control_sha256": "c" * 64,
        "harmony_tracks": tracks,
    }


def _binding(track: str, tracks: list[str]) -> dict[str, object]:
    return {
        "track_name": track,
        "audio_role": {"HARM1": "harm1", "HARM2": "harm2", "HARM3": "harm3"}[track],
        "source_task": _source_task(tracks),
    }


def _passing_report(tracks: list[str] | None = None) -> dict[str, object]:
    selected = tracks or ["HARM1", "HARM3"]
    bindings = [_binding(track, selected) for track in selected]
    return {
        "format": VOCAL_HELD_OUT_REPORT_FORMAT,
        "quality_policy": vocal_profile_quality_policy_identity(),
        "evidence": {
            "component_hashes": {
                "vocals.frame_activity_pitch": "d" * 64,
                "vocals.phrase_boundaries": "e" * 64,
                "vocals.lyric_alignment": "f" * 64,
                "vocals.talky_activity": "1" * 64,
                "vocals.harmony_model": "2" * 64,
            },
            "component_configuration_hashes": {
                "vocals.frame_activity_pitch": "3" * 64,
                "vocals.phrase_boundaries": "4" * 64,
                "vocals.lyric_alignment": "5" * 64,
                "vocals.talky_activity": "6" * 64,
                "vocals.harmony_model": "7" * 64,
            },
            "catalog_control_sha256": "c" * 64,
            "task_view_hashes": {
                "vocals_activity": "8" * 64,
                "vocals_phrase_boundaries": "9" * 64,
                "vocals_lyric_alignment": "0" * 64,
                "vocals_talky_activity": "1" * 64,
                "vocals_harmony_source_policy": "a" * 64,
            },
            "test_source_ids_sha256": "b" * 64,
        },
        "harmony": {
            "selected_tracks": selected,
            "bindings": bindings,
            "metrics": [
                {
                    **_binding(track, selected),
                    "metrics": {"track_specific_note_f1": 0.80},
                }
                for track in selected
            ],
        },
        "metrics": {
            "pitched_notes": {"onset_f1": 0.90, "offset_f1": 0.90, "pitch_accuracy": 0.90},
            "phrases": {"start_f1": 0.90, "end_f1": 0.90},
            "lyrics": {"token_error_rate": 0.10, "timestamp_alignment_error_ms": 50.0},
            "talkies": {"span_f1": 0.90},
        },
        "assembled_chart": {
            "valid_midi": True,
            "per_track_event_coverage": {"PART VOCALS": 1.0, **dict.fromkeys(selected, 1.0)},
        },
    }


def test_harmony_contract_supports_approved_subsets_with_exact_track_role_bindings() -> None:
    tracks = ["HARM1", "HARM3"]
    result = validate_vocal_harmony_bindings(tracks, [_binding(track, tracks) for track in tracks])

    assert result["selected_tracks"] == tracks
    assert [item["audio_role"] for item in result["bindings"]] == ["harm1", "harm3"]
    rendered = json.dumps(result)
    assert "catalog_root" not in rendered
    assert "audio_path" not in rendered
    assert "midi_path" not in rendered


def test_harmony_contract_returns_canonical_evidence_immune_to_input_mutation() -> None:
    tracks = ["HARM1", "HARM3"]
    bindings = [_binding(track, tracks) for track in tracks]

    result = validate_vocal_harmony_bindings(tracks, bindings)
    bindings[0]["source_task"]["harmony_tracks"][0] = "HARM2"

    returned_task = result["bindings"][0]["source_task"]
    assert returned_task["harmony_tracks"] == ["HARM1", "HARM3"]
    assert returned_task is not bindings[0]["source_task"]
    assert returned_task["harmony_tracks"] is not bindings[0]["source_task"]["harmony_tracks"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda bindings: bindings.__setitem__(0, {**bindings[0], "audio_role": "vocals"}),
            "audio role does not match",
        ),
        (
            lambda bindings: bindings.__setitem__(1, {**bindings[1], "track_name": "HARM2"}),
            "must map each selected",
        ),
        (
            lambda bindings: bindings[1]["source_task"].__setitem__("harmony_tracks", ["HARM1"]),
            "subset does not match",
        ),
        (
            lambda bindings: bindings[1]["source_task"].__setitem__(
                "source_policy_sha256", "d" * 64
            ),
            "same approved source task identity",
        ),
    ],
)
def test_harmony_contract_rejects_invalid_mapping_subset_or_policy_lineage(
    mutate: object, message: str
) -> None:
    tracks = ["HARM1", "HARM3"]
    bindings = [_binding(track, tracks) for track in tracks]
    assert callable(mutate)
    mutate(bindings)

    with pytest.raises(VocalProfileContractError, match=message):
        validate_vocal_harmony_bindings(tracks, bindings)


def test_quality_report_recomputes_per_metric_and_per_track_outcomes() -> None:
    outcomes = evaluate_vocal_profile_quality_report(_passing_report())

    assert outcomes["aggregation"] == {"rule": "all-required-metrics-pass/v1", "passed": True}
    assert outcomes["harmony"]["aggregation"] == "all-selected-approved-tracks-pass/v1"
    assert outcomes["harmony"]["per_track"]["HARM1"]["track_specific_note_f1"] == {
        "observed": 0.8,
        "operator": "gte",
        "threshold": 0.75,
        "passed": True,
    }
    assert (
        require_vocal_profile_package_evidence(_passing_report())["aggregation"]["passed"] is True
    )


def test_quality_policy_definition_is_reconstructed_from_its_pinned_content() -> None:
    mutated_copy = vocal_profile_quality_policy_definition()
    mutated_copy["metric_requirements"]["pitched_notes"]["onset_f1"]["threshold"] = 0.0

    fresh_copy = vocal_profile_quality_policy_definition()
    assert fresh_copy["metric_requirements"]["pitched_notes"]["onset_f1"]["threshold"] == 0.80
    assert fresh_copy["format"] == vocal_profile_quality_policy_identity()["format"]


def test_quality_policy_identity_and_package_evidence_ignore_mutated_module_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = vocal_profile_quality_policy_identity()
    monkeypatch.setattr(
        vocal_profile_contract,
        "VOCAL_PROFILE_QUALITY_POLICY_FORMAT",
        "mutated-policy-format/v999",
        raising=False,
    )
    monkeypatch.setattr(
        vocal_profile_contract,
        "VOCAL_PROFILE_QUALITY_POLICY_SHA256",
        "f" * 64,
        raising=False,
    )
    monkeypatch.setattr(
        vocal_profile_contract,
        "_VOCAL_PROFILE_QUALITY_POLICY",
        {"format": "mutated-policy-format/v999", "policy_id": "mutated-policy-id"},
        raising=False,
    )

    assert vocal_profile_quality_policy_identity() == expected
    assert evaluate_vocal_profile_quality_report(_passing_report())["quality_policy"] == expected

    mutated_report = _passing_report()
    mutated_report["quality_policy"]["format"] = "mutated-policy-format/v999"
    with pytest.raises(VocalProfileContractError, match="pinned STRUM policy"):
        require_vocal_profile_package_evidence(mutated_report)


def test_quality_gate_rejects_missing_or_non_strum_policy_and_failed_evidence() -> None:
    missing_policy = _passing_report()
    del missing_policy["quality_policy"]
    with pytest.raises(VocalProfileContractError, match="unsupported or missing"):
        require_vocal_profile_package_evidence(missing_policy)

    unpinned_policy = _passing_report()
    unpinned_policy["quality_policy"] = {
        **vocal_profile_quality_policy_identity(),
        "sha256": "d" * 64,
    }
    with pytest.raises(VocalProfileContractError, match="pinned STRUM policy"):
        require_vocal_profile_package_evidence(unpinned_policy)

    failed = _passing_report()
    failed["metrics"]["pitched_notes"]["onset_f1"] = 0.79
    outcomes = evaluate_vocal_profile_quality_report(failed)
    assert outcomes["lead"]["pitched_notes"]["onset_f1"]["passed"] is False
    assert outcomes["aggregation"]["passed"] is False
    with pytest.raises(VocalProfileContractError, match="requires a passing"):
        require_vocal_profile_package_evidence(failed)

    missing_component_evidence = _passing_report()
    del missing_component_evidence["evidence"]["component_hashes"]["vocals.harmony_model"]
    with pytest.raises(VocalProfileContractError, match="component evidence is incomplete"):
        require_vocal_profile_package_evidence(missing_component_evidence)


def test_quality_gate_rejects_missing_per_track_evidence_and_shared_audio_fallback() -> None:
    missing_track = _passing_report()
    missing_track["harmony"]["metrics"].pop()
    with pytest.raises(VocalProfileContractError, match="must cover the selected subset"):
        require_vocal_profile_package_evidence(missing_track)

    fallback = _passing_report()
    fallback["harmony"]["bindings"][0]["audio_role"] = "mix"
    with pytest.raises(VocalProfileContractError, match="audio role does not match"):
        require_vocal_profile_package_evidence(fallback)

    mismatched_per_track = copy.deepcopy(_passing_report())
    mismatched_per_track["harmony"]["metrics"][0]["source_task"]["task_view_sha256"] = "d" * 64
    with pytest.raises(
        VocalProfileContractError, match="does not match its approved track binding"
    ):
        require_vocal_profile_package_evidence(mismatched_per_track)
