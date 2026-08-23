"""Selected-candidate contracts for raw catalog-backed Pro experiments.

The two Pro workers intentionally produce mutually exclusive research
artifacts.  This module is the single authority for their public descriptor
map *and* for validating a produced bundle.  A caller must not infer a
candidate's identity from a component name or accept a generic valid bundle:
doing so would allow known-event attributes and free-running proposals to be
combined or relabelled as one another.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.model_bundle import BundleValidationError, ModelBundle, load_model_bundle
from src.pro_event_proposal_preprocessing import (
    PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
    PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
)

KNOWN_EVENT_CANDIDATE_KIND = "known_event_attributes/v1"
FREE_RUNNING_PROPOSAL_CANDIDATE_KIND = "free_running_event_proposal/v1"
KNOWN_EVENT_PREPROCESSING_ID = "pro-logmel-event-windows/v1"
PRO_CANDIDATE_OUTPUT_CONTRACT_FORMAT = "strum-candidate-checkpoint-output-contracts/v1"

_PIPELINE_BY_TASK_KIND = {
    "pro_guitar": "strum.instrument-chart/pro-guitar/v1",
    "pro_bass": "strum.instrument-chart/pro-bass/v1",
    "pro_keys": "strum.instrument-chart/pro-keys/v1",
}


class ProCandidateContractError(BundleValidationError):
    """Raised when a raw Pro bundle does not prove its selected candidate kind."""


@dataclass(frozen=True)
class ProCandidateContract:
    """Immutable component and semantic contract for one Pro candidate kind."""

    task_kind: str
    candidate_kind: str
    pipeline_id: str
    component_id: str
    model_outputs: tuple[str, ...]
    preprocessing: dict[str, object]
    config_format: str
    model_implementation: str
    input_contract: dict[str, object]
    output_contract: dict[str, object]
    target_contract: dict[str, object] | None

    def as_descriptor_entry(self) -> dict[str, object]:
        """Return the host-visible, path-free selected-output contract."""
        return {
            "component_outputs": [self.component_id],
            "model_outputs": list(self.model_outputs),
            "preprocessing": dict(self.preprocessing),
            "candidate_bundle": {
                "config_format": self.config_format,
                "task_kind": self.task_kind,
                "pipeline_id": self.pipeline_id,
                "model_implementation": self.model_implementation,
                "input_contract": dict(self.input_contract),
                "output_contract": dict(self.output_contract),
                **(
                    {"target_contract": dict(self.target_contract)}
                    if self.target_contract is not None
                    else {}
                ),
                "component_set": [self.component_id],
                "profiles": "forbidden",
                "companions": "forbidden",
            },
            "deployment_scope": {
                "status": "raw_experiment_candidate_only",
                "profile": "not_available",
                "chart_execution": "not_available",
            },
        }


def _target_contract(task_kind: str) -> dict[str, object]:
    if task_kind in {"pro_guitar", "pro_bass"}:
        return {
            "kind": "pro_string_fret_technique/v1",
            "string_count": 6,
            "fret_range": [0, 22],
            "techniques": [
                "normal",
                "arpeggio_form",
                "bent",
                "muted",
                "tapped",
                "harmonic",
                "pinch_harmonic",
            ],
            "track_variant_head": ["standard", "22_fret"],
        }
    if task_kind == "pro_keys":
        return {
            "kind": "pro_keys_pitch_channel_range_shift/v1",
            "pitch_range": [48, 72],
            "channel_metadata": "retained_in_labels_not_predicted/v1",
            "range_state_head": ["none", "C", "D", "E", "F", "G", "A"],
        }
    raise ProCandidateContractError("unsupported Pro task kind")


def resolve_pro_candidate_contract(task_kind: str, candidate_kind: str) -> ProCandidateContract:
    """Resolve one supported task/candidate pair without trusting caller metadata."""
    pipeline_id = _PIPELINE_BY_TASK_KIND.get(task_kind)
    if pipeline_id is None:
        raise ProCandidateContractError("unsupported Pro task kind")
    instrument = task_kind.removeprefix("pro_")
    if candidate_kind == KNOWN_EVENT_CANDIDATE_KIND:
        model_outputs = (
            ("string_fret_technique", "track_variant")
            if instrument in {"guitar", "bass"}
            else ("chromatic_pitch_set", "range_shift_state")
        )
        input_contract = {
            "format": "strum-pro-known-reference-event-window/v1",
            "event_time_source": "held_out_catalog_label_only",
            "free_running_event_proposal": False,
            "sequence_decoding": False,
            "midi_emission": False,
        }
        output_contract = {
            "format": "strum-pro-known-event-attributes/v1",
            "outputs": list(model_outputs),
            "free_running_event_proposal": False,
            "sequence_decoding": False,
            "midi_emission": False,
        }
        return ProCandidateContract(
            task_kind=task_kind,
            candidate_kind=candidate_kind,
            pipeline_id=pipeline_id,
            component_id=f"pro.{instrument}.event_attributes",
            model_outputs=model_outputs,
            preprocessing={
                "id": KNOWN_EVENT_PREPROCESSING_ID,
                "input_contract": input_contract["format"],
            },
            config_format="strum-pro-event-attribute-candidate-config/v1",
            model_implementation="ProEventAttributeCNN/v1",
            input_contract=input_contract,
            output_contract=output_contract,
            target_contract=_target_contract(task_kind),
        )
    if candidate_kind == FREE_RUNNING_PROPOSAL_CANDIDATE_KIND:
        input_contract = {
            "format": "strum-pro-arbitrary-audio-window/v1",
            "requires_midi_at_inference": False,
            "offline_window_scoring": True,
            "free_running_event_proposal": True,
            "sequence_decoding": False,
            "midi_emission": False,
        }
        output_contract = {
            "format": "strum-pro-event-proposal-scores/v1",
            "event_attributes": False,
            "midi_emission": False,
        }
        return ProCandidateContract(
            task_kind=task_kind,
            candidate_kind=candidate_kind,
            pipeline_id=pipeline_id,
            component_id=f"pro.{instrument}.event_proposal",
            model_outputs=("audio_event_proposal_scores",),
            preprocessing={
                "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
                "input_contract": input_contract["format"],
                "negative_policy": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
            },
            config_format="strum-pro-event-proposal-candidate-config/v1",
            model_implementation="ProEventProposalCNN/v1",
            input_contract=input_contract,
            output_contract=output_contract,
            target_contract=None,
        )
    raise ProCandidateContractError("Pro candidate kind is invalid")


def pro_candidate_checkpoint_output_contracts(task_kind: str) -> dict[str, object]:
    """Return both mutually exclusive candidates for the discovery descriptor."""
    return {
        "format": PRO_CANDIDATE_OUTPUT_CONTRACT_FORMAT,
        "selector": {
            "training_option": "candidate_kind",
            "default": KNOWN_EVENT_CANDIDATE_KIND,
        },
        "by_candidate_kind": {
            candidate_kind: resolve_pro_candidate_contract(
                task_kind, candidate_kind
            ).as_descriptor_entry()
            for candidate_kind in (
                KNOWN_EVENT_CANDIDATE_KIND,
                FREE_RUNNING_PROPOSAL_CANDIDATE_KIND,
            )
        },
    }


def _require_equal(label: str, actual: object, expected: object) -> None:
    if actual != expected:
        raise ProCandidateContractError(
            f"Pro candidate bundle {label} disagrees with selected contract"
        )


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProCandidateContractError("Pro candidate bundle config is unreadable") from error
    if not isinstance(raw, dict):
        raise ProCandidateContractError("Pro candidate bundle config is invalid")
    return raw


def validate_pro_candidate_bundle(path: str | Path, contract: ProCandidateContract) -> ModelBundle:
    """Require a produced bundle to exactly match one selected raw candidate.

    This is intentionally narrower than generic portable-bundle preflight.
    It verifies all component/config bytes first, then proves that the bundle
    cannot mix candidate kinds or reinterpret one kind's checkpoint with the
    other's input/output semantics.  It never creates a profile or grants
    deployment authority.
    """
    bundle = load_model_bundle(path, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    if errors:
        raise ProCandidateContractError("; ".join(errors))
    _require_equal("model component set", set(bundle.components), {contract.component_id})
    _require_equal("profiles", set(bundle.profiles), set())
    _require_equal("companions", set(bundle.companions), set())
    component = bundle.component(contract.component_id)
    if component is None:  # covered above, retained for type narrowing
        raise ProCandidateContractError("Pro candidate bundle component is missing")
    if component.checkpoint is None or component.sha256 is None or component.byte_length is None:
        raise ProCandidateContractError("Pro candidate bundle checkpoint identity is incomplete")
    if (
        component.config is None
        or component.config_sha256 is None
        or component.config_byte_length is None
    ):
        raise ProCandidateContractError("Pro candidate bundle config identity is incomplete")
    _require_equal("manifest architecture", component.architecture, contract.model_implementation)
    _require_equal("manifest preprocessing", component.preprocessing, contract.preprocessing["id"])
    config = _load_config(component.config)
    _require_equal("config schema_version", config.get("schema_version"), 1)
    _require_equal("config format", config.get("format"), contract.config_format)
    _require_equal("config task_kind", config.get("task_kind"), contract.task_kind)
    _require_equal("config pipeline_id", config.get("pipeline_id"), contract.pipeline_id)
    _require_equal(
        "config model_implementation",
        config.get("model_implementation"),
        contract.model_implementation,
    )
    _require_equal("config input_contract", config.get("input_contract"), contract.input_contract)
    _require_equal(
        "config output_contract", config.get("output_contract"), contract.output_contract
    )
    if contract.target_contract is not None:
        _require_equal(
            "config target_contract", config.get("target_contract"), contract.target_contract
        )
        _require_equal(
            "config preprocessing", config.get("preprocessing"), contract.preprocessing["id"]
        )
    else:
        preprocessing = config.get("preprocessing")
        expected_preprocessing = {
            "id": contract.preprocessing["id"],
            "negative_policy": {"id": contract.preprocessing["negative_policy"]},
        }
        _require_equal("config preprocessing", preprocessing, expected_preprocessing)
    return bundle
