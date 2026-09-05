"""Versioned, machine-readable STRUM worker contract.

This entry point intentionally exposes discovery and safe preflight first. It
does not import the mutable auto-chart pipeline while OCTAVE is deciding which
runtime or model bundle to use. Training and chart execution are added as
pipeline handlers behind the same protocol rather than requiring callers to
know STRUM's source-tree scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.catalog_chart_pairs import CatalogChartPairOptions, prepare_catalog_chart_pairs
from src.catalog_drums_manifest import build_drums_manifest, write_drums_manifest
from src.catalog_guitar_manifest import (
    build_guitar_manifest,
    write_guitar_manifest,
)
from src.catalog_guitar_manifest import (
    deterministic_split as guitar_deterministic_split,
)
from src.catalog_task_manifest import (
    DEFAULT_AUDIO_ROLES as CATALOG_TASK_DEFAULT_AUDIO_ROLES,
)
from src.catalog_task_manifest import (
    PIPELINE_IDS as CATALOG_TASK_PIPELINES,
)
from src.catalog_task_manifest import (
    TASK_INSTRUMENTS as CATALOG_TASK_INSTRUMENTS,
)
from src.catalog_task_manifest import (
    TASK_LABEL_SCHEMAS as CATALOG_TASK_LABEL_SCHEMAS,
)
from src.catalog_task_manifest import (
    build_catalog_task_manifest,
    has_exact_lead_vocal_label_source,
    select_compatible_vocal_audio_role,
    write_catalog_task_manifest,
)
from src.catalog_task_manifest import (
    deterministic_split as catalog_task_deterministic_split,
)
from src.chart_transform_calibration import (
    calibration_policy_evidence,
    checkpoint_selection_policy_evidence,
)
from src.chart_transform_quality_policy import quality_policy_evidence
from src.five_lane_runtime_admission import (
    PROFILE_GRADE_ADMISSION,
    PROFILE_GRADE_ADMISSION_FORMAT,
    PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT,
    classify_five_lane_runtime_source,
    profile_grade_admission_is_met,
    validate_profile_grade_audio_selection,
)
from src.model_bundle import (
    MANIFEST_FILENAME,
    BundleValidationError,
    InferenceProfile,
    ModelBundle,
    load_model_bundle,
)
from src.pro_candidate_contract import (
    FREE_RUNNING_PROPOSAL_CANDIDATE_KIND,
    KNOWN_EVENT_CANDIDATE_KIND,
    ProCandidateContractError,
    pro_candidate_checkpoint_output_contracts,
    resolve_pro_candidate_contract,
    validate_pro_candidate_bundle,
)
from src.pro_event_proposal_preprocessing import (
    PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
    PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
)
from src.pro_target_manifest import (
    build_catalog_pro_target_manifest,
    write_catalog_pro_target_manifest,
)
from src.song_source_catalog import (
    AUDIO_ROLES,
    CATALOG_FILENAME,
    TRAINING_ALLOWED,
    CatalogAsset,
    CatalogValidationError,
    SongSourceCatalog,
    load_catalog,
    select_training_sources,
)
from src.source_provenance import source_revision_identity
from src.vocal_harmony_catalog import (
    build_vocal_harmony_source_task,
    inspect_vocal_harmony_source_catalog,
    write_vocal_harmony_source_task,
)
from src.vocal_lead_catalog_admission import (
    VocalLeadCatalogAdmissionError,
    resolve_vocal_lead_catalog_admission,
)
from src.vocal_lead_profile_contract import vocal_lead_candidate_contract_definition
from src.vocal_profile_contract import (
    vocal_profile_protocol_definition,
    vocal_profile_protocol_identity,
    vocal_profile_quality_policy_definition,
    vocal_profile_quality_policy_identity,
)

PROTOCOL_VERSION = 1
MODEL_BUNDLE_SCHEMA_VERSIONS = (1,)
MAX_ESTIMATED_STORAGE_BYTES = (1 << 63) - 1
CHART_PREFLIGHT_FORMAT = "strum-chart-preflight/v1"
CHART_RUN_FORMAT = "strum-chart-run/v1"
MODEL_BUNDLE_DISCOVERY_FORMAT = "strum-model-bundle-discovery/v1"
MODEL_BUNDLE_INSPECTION_FORMAT = "strum-model-bundle-inspection/v1"
# A selected model directory is an operator-controlled local location, not a
# general-purpose filesystem index.  These bounds make discovery predictable
# and avoid an accidental scan of a large source tree while still supporting
# normal ``models/<release>/<bundle>`` layouts.
MAX_MODEL_DISCOVERY_DEPTH = 8
MAX_DISCOVERED_MODEL_MANIFESTS = 256
MODEL_DISCOVERY_IGNORED_DIRECTORIES = frozenset({".git", ".venv", "venv", "__pycache__"})
CATALOG_STORAGE_ESTIMATE_SEMANTICS = (
    "sum of distinct catalog input assets selected by the declared policy; "
    "excludes generated task views, preprocessing caches, checkpoints, and existing catalog storage"
)


@dataclass(frozen=True)
class PipelineDescriptor:
    """A STRUM-owned pipeline capability visible to host applications."""

    id: str
    display_name: str
    kind: str
    version: int
    catalog_requirements: dict[str, object]
    prepare_schema: dict[str, object]
    train_schema: dict[str, object] | None
    checkpoint_outputs: tuple[str, ...]
    inference_capability: str | None
    status: str
    preparation_status: str
    training_status: str
    # Some catalog-ready workers expose mutually exclusive candidate kinds.
    # ``checkpoint_outputs`` is deliberately empty for those pipelines: a
    # host must not assume that every candidate writes every component.  This
    # typed selector map declares the component(s) produced by each train
    # option, the preprocessing contract that produced them, and their
    # deliberately non-deployable scope.
    checkpoint_output_contracts: dict[str, object] | None = None
    # These names are safe to reveal to a host renderer, but their values are
    # always resolved and injected by the host's main process.  Keeping them
    # in STRUM's descriptor avoids a growing OCTAVE-side list of pipeline IDs.
    private_request_fields: tuple[str, ...] = ()
    catalog_inspection_option_keys: tuple[str, ...] = ()
    # Stable capability identifiers explaining a runtime prerequisite or the
    # next STRUM-owned contract. Hosts can display these without treating a
    # task view or raw experiment as a deployment claim.
    training_requirements: tuple[str, ...] = ()
    # Optional structured requirements for an intentionally planned training
    # path.  This lets a host explain why a catalog-ready task cannot train or
    # chart yet without guessing from a prose status string.
    training_contract: dict[str, object] | None = None
    # Explicit post-training operations are part of the worker protocol, not
    # an OCTAVE-side table of legacy subcommands.  Their private inputs are
    # resolved by the host main process; renderer schemas contain options only.
    promotion_jobs: tuple[PromotionJobDescriptor, ...] = ()

    def as_json(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_outputs"] = list(self.checkpoint_outputs)
        if data["checkpoint_output_contracts"] is None:
            data.pop("checkpoint_output_contracts")
        data["private_request_fields"] = list(self.private_request_fields)
        data["catalog_inspection_option_keys"] = list(self.catalog_inspection_option_keys)
        data["training_requirements"] = list(self.training_requirements)
        if data["training_contract"] is None:
            data.pop("training_contract")
        elif self.id == "strum.instrument-chart/vocals/v1":
            # The policy identity must be re-derived from the canonical Vocal
            # policy bytes at output time.  Do not leak a process-mutable
            # global alias into an OCTAVE-visible descriptor.
            data["training_contract"] = _vocal_training_contract_for_output(
                data["training_contract"]
            )
        data["promotion_jobs"] = [job.as_json() for job in self.promotion_jobs]
        return data


@dataclass(frozen=True)
class PromotionJobDescriptor:
    """One typed, worker-owned post-training evaluation or package action."""

    id: str
    display_name: str
    kind: str
    status: str
    options_schema: dict[str, object]
    private_request_fields: tuple[str, ...]
    output_kind: str
    deployment_scope: str
    optional_private_request_fields: tuple[str, ...] = ()
    quality_policy: dict[str, object] | None = None
    calibration_policy: dict[str, object] | None = None
    checkpoint_selection_policy: dict[str, object] | None = None

    def as_json(self) -> dict[str, object]:
        data: dict[str, object] = {
            "id": self.id,
            "display_name": self.display_name,
            "kind": self.kind,
            "status": self.status,
            "options_schema": self.options_schema,
            "private_request_fields": list(self.private_request_fields),
            "optional_private_request_fields": list(self.optional_private_request_fields),
            "output_kind": self.output_kind,
            "deployment_scope": self.deployment_scope,
        }
        if self.quality_policy is not None:
            # Re-derive canonical bytes for the only current policy-bearing
            # jobs. A caller that mutates a previously rendered descriptor
            # must not influence later discovery output.
            data["quality_policy"] = (
                quality_policy_evidence()
                if self.id.startswith("chart-transform.")
                else self.quality_policy
            )
        if self.calibration_policy is not None:
            data["calibration_policy"] = (
                calibration_policy_evidence()
                if self.id.startswith("chart-transform.")
                else self.calibration_policy
            )
        if self.checkpoint_selection_policy is not None:
            data["checkpoint_selection_policy"] = (
                checkpoint_selection_policy_evidence()
                if self.id.startswith("chart-transform.")
                else self.checkpoint_selection_policy
            )
        return data


def _object_schema(
    properties: dict[str, object], *, required: tuple[str, ...] = ()
) -> dict[str, object]:
    """Return the small JSON-Schema subset exposed to the OCTAVE renderer."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(required),
    }


CATALOG_AUDIO_OPTIONS = {
    "audio_role": {"type": "string"},
    "fallback_audio_role": {"type": ["string", "null"]},
    "required_difficulty": {"type": "string", "default": "expert"},
}
FIVE_LANE_PROFILE_AUDIO_OPTIONS = {
    **CATALOG_AUDIO_OPTIONS,
    "profile_grade": {
        "type": "boolean",
        "default": False,
        "description": "Require dedicated audio, runtime admission, and source-disjoint coverage.",
    },
}
CHART_TRANSFORM_PREPARE_SCHEMA = _object_schema(
    {
        "instrument": {"type": "string", "enum": ["guitar", "bass", "keys", "drums"]},
        "target_difficulty": {"type": "string", "enum": ["Hard", "Medium", "Easy"]},
        "split_seed": {"type": "integer", "default": 20260814},
        "calibration_fraction": {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 1,
            "default": 0.1,
        },
        "test_fraction": {"type": "number", "exclusiveMinimum": 0, "maximum": 1, "default": 0.1},
        "dataset_id": {"type": "string"},
        "overwrite": {"type": "boolean", "default": False},
        "audio_feature_mode": {"type": "string", "enum": ["none", "rms_onset_v1"]},
        "audio_role": {"type": ["string", "null"]},
        "fallback_audio_role": {"type": ["string", "null"]},
    },
    required=("instrument", "target_difficulty"),
)
CHART_TRANSFORM_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "checkpoint_mode": {
            "type": "string",
            "enum": ["fresh", "fine_tune"],
            "default": "fresh",
        },
        # OCTAVE renders an opaque artifact selector, then resolves it in its
        # main process to the private top-level ``parent_bundle`` request
        # field. The renderer never receives a filesystem location.
        "parent_artifact_id": {
            "type": "string",
            "format": "strum-model-bundle-artifact-id",
        },
        "seed": {"type": "integer", "default": 20260813},
        "lane_count": {"type": "integer", "minimum": 1, "default": 5},
        "alignment_tolerance_ms": {"type": "number", "minimum": 0, "default": 50},
        "hidden_dim": {"type": "integer", "minimum": 1, "default": 32},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "epochs": {"type": "integer", "minimum": 1, "default": 20},
        "device": {"type": "string", "default": "auto"},
        "audio_sample_rate": {"type": "integer", "minimum": 1, "default": 16000},
        "audio_window_ms": {"type": "number", "exclusiveMinimum": 0, "default": 50},
        "audio_max_duration_seconds": {"type": "number", "exclusiveMinimum": 0, "default": 900},
        "strum_revision": {"type": "string"},
    },
    required=("model_id",),
)
GUITAR_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 128},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
    },
    required=("model_id",),
)
BASS_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 128},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
    },
    required=("model_id",),
)
KEYS_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 128},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
    },
    required=("model_id",),
)
FRET_MAPPER_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 30},
        "batch_size": {"type": "integer", "minimum": 1, "default": 4096},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "hidden": {"type": "integer", "minimum": 1, "default": 256},
        "dropout": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.2},
        "pos_weight_cap": {"type": "number", "exclusiveMinimum": 0, "default": 5.0},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "max_songs": {"type": "integer", "minimum": 0, "default": 0},
        "workers": {"type": "integer", "minimum": 1, "default": 2},
        "onset_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
        "frame_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.3},
        "min_note_length": {"type": "integer", "minimum": 1, "default": 11},
        "seed": {"type": "integer", "minimum": 0, "default": 42},
    },
    required=("model_id",),
)
SECTION_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 15},
        "batch_size": {"type": "integer", "minimum": 1, "default": 256},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "num_workers": {"type": "integer", "minimum": 0, "default": 0},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
    },
    required=("model_id",),
)
PLANNED_TRAINING_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    # ``strum.instrument-chart/{bass,keys}/v1`` are durable catalog label
    # contracts for a future, potentially different architecture.  They are
    # deliberately not aliases for the narrower V1 five-lane workers below:
    # a host must not send their generic task view to a V1 trainer or assume a
    # generic task view has a chart handler.  The concrete V1 paths are
    # advertised separately in ``INSTRUMENT_CHART_TRAINING_CONTRACTS``.
    "bass": (
        "generic_bass_chart_model_trainer/v1",
        "generic_bass_held_out_chart_evaluation/v1",
        "generic_bass_profile_package/v1",
        "generic_bass_chart_execution/v1",
    ),
    "keys": (
        "generic_keys_chart_model_trainer/v1",
        "generic_keys_held_out_chart_evaluation/v1",
        "generic_keys_profile_package/v1",
        "generic_keys_chart_execution/v1",
    ),
    "vocals": (
        "vocal_activity_pitch_component/v1",
        "vocal_phrase_boundary_component/v1",
        "vocal_lyric_ctc_component/v1",
        "vocal_talky_activity_component/v1",
        "vocal_harmony_source_policy/v1",
        "vocal_chart_composition_contract/v1",
        "vocal_held_out_chart_evaluation/v1",
        "vocal_profile_package/v1",
        "vocal_chart_execution/v1",
    ),
    "pro_guitar": (
        "pro_guitar_free_running_event_proposal/v1",
        "pro_guitar_variant_aware_sequence_decoder/v1",
        "pro_guitar_held_out_evaluation/v1",
        "pro_guitar_profile_package/v1",
        "pro_guitar_chart_execution/v1",
    ),
    "pro_bass": (
        "pro_bass_free_running_event_proposal/v1",
        "pro_bass_variant_aware_sequence_decoder/v1",
        "pro_bass_held_out_evaluation/v1",
        "pro_bass_profile_package/v1",
        "pro_bass_chart_execution/v1",
    ),
    "pro_keys": (
        "pro_keys_free_running_event_proposal/v1",
        "pro_keys_chromatic_sequence_decoder/v1",
        "pro_keys_held_out_evaluation/v1",
        "pro_keys_profile_package/v1",
        "pro_keys_chart_execution/v1",
    ),
    "section_guitar": (
        "section_router_profile_loader_tensor_only",
        "held_out_section_calibration_evaluation",
        "held_out_chart_impact_ablation",
        "composed_guitar_chart_profile_contract",
    ),
    "section_bass": (
        "section_router_profile_loader_tensor_only",
        "held_out_section_calibration_evaluation",
        "held_out_chart_impact_ablation",
        "composed_bass_chart_profile_contract",
    ),
}


INSTRUMENT_CHART_TRAINING_CONTRACTS: dict[str, dict[str, object]] = {
    "bass": {
        "format": "strum-planned-training-contract/v1",
        "training_status": "planned",
        "label_source": {
            "schema_id": "five-lane-midi/v2",
            "selection": "exact-five-lane-track/v1",
            "tracks": ["PART BASS"],
            "required_difficulty": "expert",
            "target_semantics": [
                "five_lane_note_timing_and_duration",
                "expert_lane_notes_96_100",
            ],
        },
        # A generic descriptor is a durable source contract, not a promise
        # that any existing runtime can execute it.  This explicit bridge
        # lets OCTAVE present the available narrow V1 implementation without
        # treating the two task views, component identities, or profiles as
        # interchangeable.
        "available_concrete_paths": [
            {
                "pipeline_id": "bass.onset-fret/v1",
                "task_kind": "bass_onset_fret",
                "label_source": "exact-part-bass-five-lane/v1",
                "components": ["bass.onset", "bass.fret"],
                "profile_capability": "bass.neural-v1-expert/v1",
                "deployment_status": "requires_held_out_evaluation_and_profile_packaging",
                "difficulty_policy": "expert_only",
                "execution": "available_after_profile_validation",
            }
        ],
        "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["bass"]),
        "execution": {"status": "not_available", "inference_capability": None},
    },
    "keys": {
        "format": "strum-planned-training-contract/v1",
        "training_status": "planned",
        "label_source": {
            "schema_id": "five-lane-midi/v2",
            "selection": "exact-five-lane-track/v1",
            "tracks": ["PART KEYS"],
            "required_difficulty": "expert",
            "target_semantics": [
                "five_lane_note_timing_and_duration",
                "expert_lane_notes_96_100",
            ],
        },
        "available_concrete_paths": [
            {
                "pipeline_id": "keys.onset-fret/v1",
                "task_kind": "keys_onset_fret",
                "label_source": "exact-part-keys-five-lane/v1",
                "components": ["keys.onset", "keys.fret"],
                "profile_capability": "keys.neural-v1-expert/v1",
                "deployment_status": "requires_held_out_evaluation_and_profile_packaging",
                "difficulty_policy": "expert_only",
                "execution": "available_after_profile_validation",
            }
        ],
        "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["keys"]),
        "execution": {"status": "not_available", "inference_capability": None},
    },
}


VOCALS_TRAINING_CONTRACT: dict[str, object] = {
    "format": "strum-planned-training-contract/v1",
    "training_status": "planned",
    "label_source": {
        "schema_id": "vocals-pitch-phrase-lyrics-midi/v1",
        "selection": "exact-lead-vocal-track/v1",
        "tracks": ["PART VOCALS"],
        "required_difficulty": "expert",
        "target_semantics": [
            "pitched_note_timing_duration_midi_36_84",
            "phrase_boundary_markers_midi_105_106_or_105_span",
            "lyric_and_text_meta_events",
            "pitchless_talky_marker_midi_96",
        ],
        "excluded_source_tracks": ["HARM1", "HARM2", "HARM3"],
    },
    "available_experiment_components": [
        "vocals.frame_activity_pitch",
        "vocals.phrase_boundaries",
        "vocals.lyric_alignment",
        "vocals.talky_activity",
    ],
    "available_source_policies": {
        "harmony": {
            "pipeline_id": "vocals.harmony-source-policy/v1",
            "task_format": "strum-vocal-harmony-source-task/v1",
            "status": "prepare_only",
            "shared_vocal_or_mix_fallback": False,
        }
    },
    # These are requirements for a future composed profile, not a description
    # of any current raw experiment.  Keeping source/event semantics here is
    # intentional: OCTAVE must not invent a Vocal graph from four arbitrary
    # checkpoints just because their component names happen to look related.
    "catalog_admission": {
        "lead": {
            "track": "PART VOCALS",
            "difficulty": "expert",
            "admission_resolver": {
                "id": "strum-owned-lead-catalog-task-admission-resolver/v1",
                "status": "available",
                "scope": "catalog-data-admission-only/v1",
                "required_task_views": [
                    "vocals_activity",
                    "vocals_phrase_boundaries",
                    "vocals_lyric_alignment",
                    "vocals_talky_activity",
                ],
                "report_format": "strum-vocal-lead-catalog-task-admission/v1",
            },
            "required_labels": [
                "pitched_note_timing_duration_midi_36_84",
                "phrase_boundary_markers_midi_105_106_or_105_span",
                "lyric_and_text_meta_events",
                "pitchless_talky_marker_midi_96",
            ],
            "audio": {
                "selection": "catalog-recorded-vocal-or-mix-identity/v1",
                "preferred_role": "vocals",
                "fallback_role": "mix",
                "required_timeline": "same-master-timeline/v1",
            },
        },
        "harmony": {
            "tracks": ["HARM1", "HARM2", "HARM3"],
            "selection": "exact-approved-harmony-subset/v1",
            "source_task_format": "strum-vocal-harmony-source-task/v1",
            "required_audio_policy": "isolated-harmony-stem-only/v1",
            "forbidden_audio_roles": ["vocals", "mix"],
            "required_provenance": [
                "isolated_source_stem/v1",
                "isolated_separation_output/v1",
            ],
            "subset": {
                "minimum_tracks": 1,
                "all_tracks_required": False,
                "must_be_declared_by_selected_source_task": True,
            },
            "required_per_track_binding": {
                "track_name": "HARM1|HARM2|HARM3",
                "audio_role": "matching-harm1-harm2-harm3-role/v1",
                "source_task": {
                    "format": "strum-vocal-harmony-source-task/v1",
                    "task_view_sha256": "sha256",
                    "source_policy_format": "octave-vocal-harmony-source-policy/v1",
                    "source_policy_sha256": "sha256",
                    "catalog_control_sha256": "sha256",
                    "harmony_tracks": "exact-selected-subset/v1",
                },
            },
        },
        "partition": {
            "unit": "source_id",
            "required_splits": ["train", "val", "test"],
            "cross_component_split_assignment": "identical-by-source-id/v1",
            "test_source_ids_forbidden_in_training_or_calibration": True,
        },
    },
    "composition_contract": {
        "format": "strum-vocal-chart-composition-contract/v1",
        "status": "not_available",
        "profile_graph_format": "strum-profile-composition/v1",
        "required_components": [
            {
                "id": "vocals.frame_activity_pitch",
                "producer_pipeline": "vocals.note-activity/v1",
                "required_outputs": ["pitched_vocal_activity", "midi_pitch_36_84"],
            },
            {
                "id": "vocals.phrase_boundaries",
                "producer_pipeline": "vocals.phrase-boundaries/v1",
                "required_outputs": ["lead_phrase_start", "lead_phrase_end"],
            },
            {
                "id": "vocals.lyric_alignment",
                "producer_pipeline": "vocals.lyric-alignment/v1",
                "required_outputs": [
                    "observed_lyric_character_tokens",
                    "observed_lyric_event_alignment",
                ],
            },
            {
                "id": "vocals.talky_activity",
                "producer_pipeline": "vocals.talky-activity/v1",
                "required_outputs": ["pitchless_talky_activity"],
            },
            {
                "id": "vocals.harmony_model",
                "producer_pipeline": "not_implemented",
                "required_outputs": "approved-nonempty-HARM-subset/v1",
                "source_policy": "vocals.harmony-source-policy/v1",
                "per_track_binding": "strum-vocal-harmony-track-binding/v1",
                "same_source_task_for_selected_tracks": True,
            },
        ],
        "compatibility": {
            "component_task_lineage": "same-catalog-control-and-source-partition/v1",
            "audio_binding": "same-catalog-audio-identity-or-pinned-alignment/v1",
            "clock": "same-master-timeline/v1",
            "lead_track": "PART VOCALS",
            "harmony_tracks_must_remain_distinct": True,
            "no_implicit_lyrics_or_harmony": True,
        },
        "chart_outputs": {
            "lead": {
                "track": "PART VOCALS",
                "events": [
                    "pitched_notes_midi_36_84",
                    "pitchless_talky_note_96",
                    "phrase_markers_105_106_or_105_span",
                    "lyrics_or_text_meta_events",
                ],
            },
            "harmony": {
                "tracks": ["HARM1", "HARM2", "HARM3"],
                "selected_subset": "approved-nonempty-subset/v1",
                "only_from_provenance_approved_harmony_components": True,
                "per_track_output_binding_required": True,
            },
        },
        "forbidden_shortcuts": [
            "raw_component_as_profile",
            "shared_vocals_or_mix_as_harmony_supervision",
            "external_lyrics_as_chart_labels",
            "legacy_vocals_charter_fallback",
        ],
    },
    "held_out_evaluation_contract": {
        "format": "strum-vocal-held-out-chart-evaluation-contract/v1",
        "status": "not_available",
        "split": "test",
        "source_partition": "source-id-disjoint-from-train-and-val/v1",
        "required_reference_labels": {
            "lead": [
                "pitched_notes_midi_36_84",
                "phrase_boundaries",
                "lyrics_or_text_meta_events",
                "pitchless_talky_note_96",
            ],
            "harmony": [
                "each-selected-HARM-track-with-matching-harm-role-and-approved-source-task"
            ],
        },
        "required_metrics": {
            "pitched_notes": ["onset_f1", "offset_f1", "pitch_accuracy"],
            "phrases": ["start_f1", "end_f1"],
            "lyrics": ["token_error_rate", "timestamp_alignment_error_ms"],
            "talkies": ["span_f1"],
            "harmony": ["per-track-track_specific_note_f1"],
            "assembled_chart": ["valid_midi", "per_track_event_coverage"],
        },
        "evidence": {
            "recomputed_by": "strum",
            "binds": [
                "component_hashes",
                "component_configuration_hashes",
                "catalog_control_sha256",
                "task_view_hashes",
                "test_source_ids",
                "per-track-harmony-source-task-view-sha256",
                "per-track-harmony-source-policy-sha256",
                "per-track-harmony-catalog-control-sha256",
                "strum-quality-policy-sha256",
            ],
            "per_track_harmony_evidence": "required-for-each-selected-HARM-track/v1",
            "quality_policy": {
                **vocal_profile_quality_policy_identity(),
                "outcomes": "strum-vocal-profile-quality-outcomes/v1",
                "aggregation": "all-required-metrics-pass/v1",
            },
        },
    },
    "packaging_contract": {
        "format": "strum-vocal-profile-package-contract/v1",
        "status": "not_available",
        "requires": [
            "completed-strum-recomputed-held-out-report",
            "pinned-strum-vocal-profile-quality-policy-v1",
            "all-required-strum-quality-policy-outcomes-pass",
            "per-selected-HARM-track-source-task-policy-and-metric-evidence",
            "hash-verified-required-components-and-configurations",
            "validated-strum-profile-composition-v1-graph",
            "registered-vocal-chart-execution-handler",
        ],
        "quality_policy": {
            **vocal_profile_quality_policy_identity(),
            "failure_behavior": "reject-package/v1",
            "verification": "recompute-outcomes-do-not-trust-reported-pass/v1",
        },
        # Expose the complete pinned requirement rather than asking OCTAVE to
        # duplicate thresholds.  This is still a planned package contract;
        # no package writer or Vocal chart handler is registered here.
        "quality_policy_definition": vocal_profile_quality_policy_definition(),
        "quality_policy_sha256": vocal_profile_quality_policy_identity()["sha256"],
        "raw_component_bundle_deployment_status": "not_deployable",
    },
    "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["vocals"]),
    "execution": {
        "status": "not_available",
        "inference_capability": None,
        "required_handler": "vocal_chart_profile_handler/v1",
        "fallback": "forbidden",
    },
}


def _vocal_training_contract_for_output(template: object) -> dict[str, object]:
    """Return the planned Vocal descriptor with canonical STRUM identities.

    Pipeline descriptors are long-lived module objects, while OCTAVE-visible
    output is serialized later.  Rebuild every policy-bearing output field at
    serialization time so it is derived by the shared identity helper rather
    than any process-mutable module alias or stale import-time digest.
    """
    if not isinstance(template, dict):  # pragma: no cover - static descriptor guard
        raise RuntimeError("Vocal training contract template must be an object")

    # ``asdict`` has already recursively copied the descriptor, but copy the
    # JSON-shaped template here too so callers of this helper cannot mutate a
    # descriptor's static data through a returned object.
    contract = json.loads(json.dumps(template))
    quality_identity = vocal_profile_quality_policy_identity()
    protocol = vocal_profile_protocol_definition()
    protocol_identity = vocal_profile_protocol_identity()
    report_format = protocol["held_out_report_format"]
    source_policy_format = protocol["harmony_source_policy_format"]
    source_task_format = protocol["harmony_source_task_format"]
    harmony_track_roles = protocol["harmony_track_roles"]
    if (
        not isinstance(report_format, str)
        or not isinstance(source_policy_format, str)
        or not isinstance(source_task_format, str)
        or not isinstance(harmony_track_roles, dict)
        or not all(
            isinstance(track, str) and isinstance(role, str)
            for track, role in harmony_track_roles.items()
        )
    ):  # pragma: no cover - static contract guard
        raise RuntimeError("Vocal protocol definition is invalid")

    admission = contract["catalog_admission"]
    composition = contract["composition_contract"]
    evaluation = contract["held_out_evaluation_contract"]
    packaging = contract["packaging_contract"]
    if (
        not isinstance(admission, dict)
        or not isinstance(composition, dict)
        or not isinstance(evaluation, dict)
        or not isinstance(packaging, dict)
    ):  # pragma: no cover
        raise RuntimeError("Vocal training contract template is invalid")
    admission_harmony = admission["harmony"]
    composition_outputs = composition["chart_outputs"]
    if (
        not isinstance(admission_harmony, dict)
        or not isinstance(composition_outputs, dict)
        or not isinstance(composition_outputs.get("harmony"), dict)
    ):  # pragma: no cover
        raise RuntimeError("Vocal Harmony contract template is invalid")
    evaluation_evidence = evaluation["evidence"]
    if not isinstance(evaluation_evidence, dict):  # pragma: no cover
        raise RuntimeError("Vocal evaluation contract template is invalid")

    # Reconstruct every exact HARM/source/report identity at serialization
    # time.  The static descriptor is documentation-oriented; the emitted
    # OCTAVE contract must remain immutable even if a caller mutates legacy
    # module aliases such as ``HARMONY_TRACK_ROLES`` or the report format.
    contract["vocal_profile_protocol"] = {
        **protocol,
        "identity": protocol_identity,
    }
    available_source_policies = contract["available_source_policies"]
    if not isinstance(available_source_policies, dict) or not isinstance(
        available_source_policies.get("harmony"), dict
    ):  # pragma: no cover
        raise RuntimeError("Vocal Harmony source policy template is invalid")
    available_source_policies["harmony"]["task_format"] = source_task_format
    available_source_policies["harmony"]["source_policy_format"] = source_policy_format
    admission_harmony["tracks"] = list(harmony_track_roles)
    admission_harmony["source_task_format"] = source_task_format
    admission_harmony["source_policy_format"] = source_policy_format
    admission_harmony["track_audio_roles"] = dict(harmony_track_roles)
    required_binding = admission_harmony["required_per_track_binding"]
    if not isinstance(required_binding, dict) or not isinstance(
        required_binding.get("source_task"), dict
    ):  # pragma: no cover
        raise RuntimeError("Vocal Harmony binding template is invalid")
    required_binding["track_name"] = "|".join(harmony_track_roles)
    required_binding["audio_role"] = "exact-canonical-HARM-to-harm-role/v1"
    source_task = required_binding["source_task"]
    source_task["format"] = source_task_format
    source_task["source_policy_format"] = source_policy_format
    composition_outputs["harmony"]["tracks"] = list(harmony_track_roles)
    evaluation["held_out_report_format"] = report_format
    evaluation["harmony_protocol"] = {
        "source_task_format": source_task_format,
        "source_policy_format": source_policy_format,
        "track_audio_roles": dict(harmony_track_roles),
        "identity": protocol_identity,
    }
    evaluation_evidence["quality_policy"] = {
        **quality_identity,
        "outcomes": "strum-vocal-profile-quality-outcomes/v1",
        "aggregation": "all-required-metrics-pass/v1",
    }
    packaging["quality_policy"] = {
        **quality_identity,
        "failure_behavior": "reject-package/v1",
        "verification": "recompute-outcomes-do-not-trust-reported-pass/v1",
    }
    packaging["quality_policy_definition"] = vocal_profile_quality_policy_definition()
    packaging["quality_policy_sha256"] = quality_identity["sha256"]
    # Lead-only is a useful research boundary while OCTAVE has no isolated
    # Harmony material.  It is deliberately a separate candidate/evidence
    # contract, never a weakened substitute for the full Vocal profile.
    contract["lead_only_candidate_contract"] = vocal_lead_candidate_contract_definition()
    return contract


def _pro_candidate_checkpoint_output_contracts(task_kind: str) -> dict[str, object]:
    """Return the selector-bound, non-deployable Pro artifact contract.

    Pro's two train options do not produce interchangeable components.  Keep
    that mapping alongside discovery rather than overloading the descriptor's
    static ``checkpoint_outputs`` tuple, which means "all required outputs"
    for ordinary single-output workers.
    """
    return pro_candidate_checkpoint_output_contracts(task_kind)


def _pro_task_kind_for_pipeline(pipeline_id: str) -> str:
    """Resolve the catalog task identity before selecting an output contract."""
    task_kind = {
        "strum.instrument-chart/pro-guitar/v1": "pro_guitar",
        "strum.instrument-chart/pro-bass/v1": "pro_bass",
        "strum.instrument-chart/pro-keys/v1": "pro_keys",
    }.get(pipeline_id)
    if task_kind is None:  # pragma: no cover - protected by caller dispatch
        raise WorkerRequestError("unsupported Pro event candidate pipeline")
    return task_kind


PRO_TRAINING_CONTRACTS: dict[str, dict[str, object]] = {
    "pro_guitar": {
        "format": "strum-planned-training-contract/v1",
        "training_status": "experiment_only",
        "label_source": {
            "schema_id": "pro-string-fret-midi/v1",
            "selection": "exact-real-track-identities/v1",
            "tracks": ["PART REAL_GUITAR", "PART REAL_GUITAR_22"],
            "required_difficulty": "expert",
            # The encoder must retain this source distinction.  A generic
            # five-lane output or a merged 17/22-fret target is invalid.
            "target_semantics": [
                "note_timing_and_duration",
                "string_fret_events",
                "track_variant",
                "pro_technique_events",
            ],
        },
        "prepared_target_encoding": "strum-pro-midi-target-decoder/v1",
        "available_preprocessing": {
            "id": "pro-logmel-event-windows/v1",
            "target_binding": "exact-real-track-event-windows/v1",
            "deployment_status": "raw_experiment_candidates_only",
        },
        "proposal_preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "audio_features": {
                "source": "task_view.audio_preprocessing",
                "base_id": "pro-logmel-event-windows/v1",
                "default_window_before_ms": 100,
                "default_window_after_ms": 400,
            },
            "negative_policy": {
                "id": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
                "rule": "negative-feature-window-must-not-contain-real-event-onset/v1",
                "options": {
                    "negative_ratio": {"minimum": 1, "maximum": 32, "default": 4},
                    "negative_exclusion_ms": {"minimum": 0, "maximum": 5000, "default": 80},
                    "negative_seed": {"source": "training_options.seed", "default": 20260822},
                },
            },
            "deployment_status": "raw_experiment_candidates_only",
        },
        "available_experiment_stages": [
            {
                "id": "pro_guitar_known_event_attribute_candidate/v1",
                "input_contract": "strum-pro-known-reference-event-window/v1",
                "outputs": ["string_fret_technique", "track_variant"],
                "free_running_event_proposal": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
            {
                "id": "pro_guitar_free_running_event_proposal_candidate/v1",
                "input_contract": "strum-pro-arbitrary-audio-window/v1",
                "outputs": ["audio_event_proposal_scores"],
                "negative_event_coverage": "deterministic_catalog_audio_windows/v1",
                "requires_midi_at_inference": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
        ],
        "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["pro_guitar"]),
        "execution": {"status": "not_available", "inference_capability": None},
    },
    "pro_bass": {
        "format": "strum-planned-training-contract/v1",
        "training_status": "experiment_only",
        "label_source": {
            "schema_id": "pro-string-fret-midi/v1",
            "selection": "exact-real-track-identities/v1",
            "tracks": ["PART REAL_BASS", "PART REAL_BASS_22"],
            "required_difficulty": "expert",
            "target_semantics": [
                "note_timing_and_duration",
                "string_fret_events",
                "track_variant",
                "pro_technique_events",
            ],
        },
        "prepared_target_encoding": "strum-pro-midi-target-decoder/v1",
        "available_preprocessing": {
            "id": "pro-logmel-event-windows/v1",
            "target_binding": "exact-real-track-event-windows/v1",
            "deployment_status": "raw_experiment_candidates_only",
        },
        "proposal_preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "audio_features": {
                "source": "task_view.audio_preprocessing",
                "base_id": "pro-logmel-event-windows/v1",
                "default_window_before_ms": 100,
                "default_window_after_ms": 400,
            },
            "negative_policy": {
                "id": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
                "rule": "negative-feature-window-must-not-contain-real-event-onset/v1",
                "options": {
                    "negative_ratio": {"minimum": 1, "maximum": 32, "default": 4},
                    "negative_exclusion_ms": {"minimum": 0, "maximum": 5000, "default": 80},
                    "negative_seed": {"source": "training_options.seed", "default": 20260822},
                },
            },
            "deployment_status": "raw_experiment_candidates_only",
        },
        "available_experiment_stages": [
            {
                "id": "pro_bass_known_event_attribute_candidate/v1",
                "input_contract": "strum-pro-known-reference-event-window/v1",
                "outputs": ["string_fret_technique", "track_variant"],
                "free_running_event_proposal": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
            {
                "id": "pro_bass_free_running_event_proposal_candidate/v1",
                "input_contract": "strum-pro-arbitrary-audio-window/v1",
                "outputs": ["audio_event_proposal_scores"],
                "negative_event_coverage": "deterministic_catalog_audio_windows/v1",
                "requires_midi_at_inference": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
        ],
        "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["pro_bass"]),
        "execution": {"status": "not_available", "inference_capability": None},
    },
    "pro_keys": {
        "format": "strum-planned-training-contract/v1",
        "training_status": "experiment_only",
        "label_source": {
            "schema_id": "pro-keys-pitch-midi/v1",
            "selection": "exact-real-track-identities/v1",
            "tracks": ["PART REAL_KEYS_X"],
            "required_difficulty": "expert",
            # Lower-difficulty REAL_KEYS tracks are not inputs to this Expert
            # label path.  STRUM's learned difficulty stage owns that later.
            "target_semantics": [
                "note_timing_and_duration",
                "midi_pitch_events",
                "expert_difficulty_track",
            ],
        },
        "prepared_target_encoding": "strum-pro-midi-target-decoder/v1",
        "available_preprocessing": {
            "id": "pro-logmel-event-windows/v1",
            "target_binding": "exact-real-track-event-windows/v1",
            "deployment_status": "raw_experiment_candidates_only",
        },
        "proposal_preprocessing": {
            "id": PRO_EVENT_PROPOSAL_PREPROCESSING_ID,
            "audio_features": {
                "source": "task_view.audio_preprocessing",
                "base_id": "pro-logmel-event-windows/v1",
                "default_window_before_ms": 100,
                "default_window_after_ms": 400,
            },
            "negative_policy": {
                "id": PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
                "rule": "negative-feature-window-must-not-contain-real-event-onset/v1",
                "options": {
                    "negative_ratio": {"minimum": 1, "maximum": 32, "default": 4},
                    "negative_exclusion_ms": {"minimum": 0, "maximum": 5000, "default": 80},
                    "negative_seed": {"source": "training_options.seed", "default": 20260822},
                },
            },
            "deployment_status": "raw_experiment_candidates_only",
        },
        "available_experiment_stages": [
            {
                "id": "pro_keys_known_event_attribute_candidate/v1",
                "input_contract": "strum-pro-known-reference-event-window/v1",
                "outputs": ["chromatic_pitch_set", "range_shift_state"],
                "free_running_event_proposal": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
            {
                "id": "pro_keys_free_running_event_proposal_candidate/v1",
                "input_contract": "strum-pro-arbitrary-audio-window/v1",
                "outputs": ["audio_event_proposal_scores"],
                "negative_event_coverage": "deterministic_catalog_audio_windows/v1",
                "requires_midi_at_inference": False,
                "sequence_decoding": False,
                "chart_execution": False,
            },
        ],
        "required_stages": list(PLANNED_TRAINING_REQUIREMENTS["pro_keys"]),
        "execution": {"status": "not_available", "inference_capability": None},
    },
}
DRUMS_ONSET_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "profile": {
            "type": "string",
            "enum": ["onset_classifier_v2"],
            "default": "onset_classifier_v2",
        },
        "seed": {"type": "integer", "default": 20260813},
        "batch_size": {"type": "integer", "minimum": 1, "default": 256},
        "epochs": {"type": "integer", "minimum": 1, "default": 100},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.001},
        "max_train_batches": {"type": "integer", "minimum": 1, "default": 2000},
        "max_test_batches": {"type": "integer", "minimum": 1, "default": 500},
        "num_workers": {"type": "integer", "minimum": 0, "default": 0},
        "strum_revision": {"type": "string"},
    },
    required=("model_id",),
)
VOCALS_ACTIVITY_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 16},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.0003},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
        "max_train_batches": {"type": "integer", "minimum": 0, "default": 0},
        "max_val_batches": {"type": "integer", "minimum": 0, "default": 0},
        "seed": {"type": "integer", "minimum": 0, "default": 20260822},
    },
    required=("model_id",),
)
PRO_EVENT_CANDIDATE_TRAIN_SCHEMA = _object_schema(
    {
        "model_id": {"type": "string"},
        "candidate_kind": {
            "type": "string",
            "enum": ["known_event_attributes/v1", "free_running_event_proposal/v1"],
            "default": "known_event_attributes/v1",
            # The selected value resolves a single entry in the descriptor's
            # path-free checkpoint-output contract.  This is intentionally a
            # selector, not permission to combine raw candidate artifacts.
            "x-strum-checkpoint-output-contract-selector": (
                "checkpoint_output_contracts.by_candidate_kind"
            ),
        },
        "epochs": {"type": "integer", "minimum": 1, "default": 25},
        "batch_size": {"type": "integer", "minimum": 1, "default": 32},
        "learning_rate": {"type": "number", "exclusiveMinimum": 0, "default": 0.0003},
        "device": {"type": "string", "enum": ["auto", "cuda", "mps", "cpu"], "default": "auto"},
        "limit_songs": {"type": "integer", "minimum": 0, "default": 0},
        "max_train_batches": {"type": "integer", "minimum": 0, "default": 0},
        "max_val_batches": {"type": "integer", "minimum": 0, "default": 0},
        "seed": {"type": "integer", "minimum": 0, "default": 20260822},
        "channels": {"type": "integer", "minimum": 1, "default": 48},
        "negative_ratio": {"type": "integer", "minimum": 1, "maximum": 32, "default": 4},
        "negative_exclusion_ms": {"type": "integer", "minimum": 0, "maximum": 5000, "default": 80},
    },
    required=("model_id",),
)
VOCAL_HARMONY_SOURCE_PREPARE_SCHEMA = _object_schema(
    {
        "harmony_tracks": {
            "type": "array",
            "items": {"type": "string", "enum": ["HARM1", "HARM2", "HARM3"]},
            "minItems": 1,
            "uniqueItems": True,
        },
        "split_ratios": {"type": "array", "items": {"type": "integer"}},
        "split_seed": {"type": "string", "default": "catalog-source-id/v1"},
    }
)
VOCALS_ACTIVITY_PREPARE_SCHEMA = _object_schema(
    {
        **CATALOG_AUDIO_OPTIONS,
        "split_ratios": {"type": "array", "items": {"type": "integer"}},
        "split_seed": {"type": "string", "default": "catalog-source-id/v1"},
    }
)

PROFILE_EVALUATE_OPTIONS_SCHEMA = _object_schema(
    {
        "device": {"type": "string", "enum": ["cpu", "cuda", "mps"], "default": "cpu"},
        "tolerance_ms": {"type": "number", "enum": [50], "default": 50},
        "limit_songs": {"type": "integer", "enum": [0], "default": 0},
    }
)
PROFILE_PACKAGE_OPTIONS_SCHEMA = _object_schema(
    {
        "profile_id": {"type": "string"},
    },
    required=("profile_id",),
)

TRANSFORM_EVALUATE_OPTIONS_SCHEMA = _object_schema(
    {"device": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu"}}
)
TRANSFORM_PACKAGE_OPTIONS_SCHEMA = _object_schema(
    {
        "profile_id": {"type": "string"},
        "device": {"type": "string", "enum": ["cpu", "cuda"], "default": "cpu"},
    },
    required=("profile_id",),
)
SECTION_EVALUATE_OPTIONS_SCHEMA = _object_schema(
    {"device": {"type": "string", "enum": ["cpu", "cuda", "mps"], "default": "cpu"}}
)
SECTION_PACKAGE_OPTIONS_SCHEMA = _object_schema(
    {
        "profile_id": {"type": "string"},
        "minimum_accuracy": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
        "maximum_expected_calibration_error": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
    required=("profile_id", "minimum_accuracy", "maximum_expected_calibration_error"),
)
EMPTY_PROMOTION_OPTIONS_SCHEMA = _object_schema({})


def _profile_promotion_jobs(instrument: str) -> tuple[PromotionJobDescriptor, ...]:
    """Return the identical strict V1 evaluation/package surface per instrument."""
    label = instrument.title()
    return (
        PromotionJobDescriptor(
            id=f"{instrument}.profile-evaluate/v1",
            display_name=f"Evaluate {label} candidate",
            kind="evaluation",
            status="available",
            options_schema=PROFILE_EVALUATE_OPTIONS_SCHEMA,
            private_request_fields=("bundle_root", "task_view", "catalog_root", "output"),
            output_kind=f"{instrument}_held_out_evaluation_report",
            deployment_scope="evaluation_evidence_only",
        ),
        PromotionJobDescriptor(
            id=f"{instrument}.profile-package/v1",
            display_name=f"Package {label} Expert profile",
            kind="package",
            status="available",
            options_schema=PROFILE_PACKAGE_OPTIONS_SCHEMA,
            private_request_fields=("experiment", "evaluation", "output"),
            output_kind=f"{instrument}_expert_profile_bundle",
            deployment_scope="deployable_after_profile_validation",
        ),
    )


TRANSFORM_PROMOTION_JOBS = (
    PromotionJobDescriptor(
        id="chart-transform.profile-evaluate/v1",
        display_name="Evaluate learned difficulty transform",
        kind="evaluation",
        status="available",
        options_schema=TRANSFORM_EVALUATE_OPTIONS_SCHEMA,
        private_request_fields=("bundle_root", "dataset_manifest", "output"),
        optional_private_request_fields=("catalog_root",),
        output_kind="chart_transform_held_out_evaluation_report",
        deployment_scope="evaluation_evidence_only",
        quality_policy=quality_policy_evidence(),
        calibration_policy=calibration_policy_evidence(),
        checkpoint_selection_policy=checkpoint_selection_policy_evidence(),
    ),
    PromotionJobDescriptor(
        id="chart-transform.profile-package/v1",
        display_name="Package learned difficulty transform",
        kind="package",
        status="available",
        options_schema=TRANSFORM_PACKAGE_OPTIONS_SCHEMA,
        private_request_fields=("experiment", "evaluation", "dataset_manifest", "output"),
        optional_private_request_fields=("catalog_root",),
        output_kind="learned_difficulty_transform_profile_bundle",
        deployment_scope="deployable_after_profile_validation",
        quality_policy=quality_policy_evidence(),
        calibration_policy=calibration_policy_evidence(),
        checkpoint_selection_policy=checkpoint_selection_policy_evidence(),
    ),
)


def _section_promotion_jobs(instrument: str) -> tuple[PromotionJobDescriptor, ...]:
    label = instrument.title()
    return (
        PromotionJobDescriptor(
            id=f"section.{instrument}.profile-evaluate/v1",
            display_name=f"Evaluate {label} section classifier",
            kind="evaluation",
            status="available",
            options_schema=SECTION_EVALUATE_OPTIONS_SCHEMA,
            private_request_fields=("bundle_root", "task_view", "catalog_root", "output"),
            output_kind=f"section_{instrument}_held_out_evaluation_report",
            deployment_scope="evaluation_evidence_only",
        ),
        PromotionJobDescriptor(
            id=f"section.{instrument}.profile-package/v1",
            display_name=f"Package {label} section evaluator",
            kind="package",
            status="available",
            options_schema=SECTION_PACKAGE_OPTIONS_SCHEMA,
            private_request_fields=("experiment", "evaluation", "output"),
            output_kind=f"section_{instrument}_evaluation_profile_bundle",
            deployment_scope="evaluation_only_not_auto_chart_runnable",
        ),
    )


DRUMS_PROMOTION_JOBS = (
    PromotionJobDescriptor(
        id="drums.onset-classifier.package-evaluation/v1",
        display_name="Package Drums onset evaluation artifact",
        kind="package",
        status="available",
        options_schema=EMPTY_PROMOTION_OPTIONS_SCHEMA,
        private_request_fields=("experiment_root", "output"),
        output_kind="drums_onset_classifier_evaluation_bundle",
        deployment_scope="evaluation_only_not_auto_chart_runnable",
    ),
)

PIPELINES = (
    PipelineDescriptor(
        id="guitar.onset-fret/v1",
        display_name="Guitar onset + fret",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "guitar",
            "difficulties": ["expert"],
            "audio_roles": ["guitar", "mix"],
            "audio_policy": "prefer:guitar,fallback:mix",
            "profile_grade_admission": PROFILE_GRADE_ADMISSION,
        },
        prepare_schema=_object_schema(FIVE_LANE_PROFILE_AUDIO_OPTIONS),
        train_schema=GUITAR_TRAIN_SCHEMA,
        checkpoint_outputs=("guitar.onset", "guitar.fret"),
        inference_capability="guitar.neural-v1-expert/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
            "profile_grade",
        ),
        training_requirements=("profile_evaluation", "profile_packaging"),
        promotion_jobs=_profile_promotion_jobs("guitar"),
    ),
    PipelineDescriptor(
        id="bass.onset-fret/v1",
        display_name="Bass onset + fret",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "bass",
            "difficulties": ["expert"],
            "audio_roles": ["bass", "mix"],
            "audio_policy": "prefer:bass,fallback:mix",
            "label_schema": CATALOG_TASK_LABEL_SCHEMAS["bass_onset_fret"]["id"],
            "label_tracks": CATALOG_TASK_LABEL_SCHEMAS["bass_onset_fret"]["track_names"],
            "profile_grade_admission": PROFILE_GRADE_ADMISSION,
        },
        prepare_schema=_object_schema(FIVE_LANE_PROFILE_AUDIO_OPTIONS),
        train_schema=BASS_TRAIN_SCHEMA,
        checkpoint_outputs=("bass.onset", "bass.fret"),
        # Raw experiments still require a held-out Bass evaluation and an
        # immutable Bass-only profile.  The capability names the profile type,
        # not an implicit promotion of every worker checkpoint.
        inference_capability="bass.neural-v1-expert/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
            "profile_grade",
        ),
        training_requirements=("bass_profile_evaluation", "bass_profile_packaging"),
        promotion_jobs=_profile_promotion_jobs("bass"),
    ),
    PipelineDescriptor(
        id="keys.onset-fret/v1",
        display_name="Keys onset + fret",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "keys",
            "difficulties": ["expert"],
            "audio_roles": ["keys", "mix"],
            "audio_policy": "prefer:keys,fallback:mix",
            "label_schema": CATALOG_TASK_LABEL_SCHEMAS["keys_onset_fret"]["id"],
            "label_tracks": CATALOG_TASK_LABEL_SCHEMAS["keys_onset_fret"]["track_names"],
            "profile_grade_admission": PROFILE_GRADE_ADMISSION,
        },
        prepare_schema=_object_schema(FIVE_LANE_PROFILE_AUDIO_OPTIONS),
        train_schema=KEYS_TRAIN_SCHEMA,
        checkpoint_outputs=("keys.onset", "keys.fret"),
        # This names only an evaluated, immutable Keys profile.  It does not
        # promote arbitrary training checkpoints to an auto-chart runtime.
        inference_capability="keys.neural-v1-expert/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
            "profile_grade",
        ),
        training_requirements=("keys_profile_evaluation", "keys_profile_packaging"),
        promotion_jobs=_profile_promotion_jobs("keys"),
    ),
    PipelineDescriptor(
        id="chart_transform.five_lane/v1",
        display_name="Learned five-lane difficulty transform",
        kind="chart_to_chart",
        version=1,
        catalog_requirements={
            "instruments": ["guitar", "bass", "keys", "drums"],
            "source_difficulty": "expert",
            "target_difficulties": ["hard", "medium", "easy"],
        },
        prepare_schema=CHART_TRANSFORM_PREPARE_SCHEMA,
        train_schema=CHART_TRANSFORM_TRAIN_SCHEMA,
        checkpoint_outputs=("chart_transform",),
        inference_capability="difficulty.transform/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        # Audio-conditioned transforms resolve catalog assets only in the
        # worker.  A fine-tune parent is likewise a main-process-only bundle.
        private_request_fields=("catalog_root", "parent_bundle"),
        catalog_inspection_option_keys=(
            "instrument",
            "target_difficulty",
            "audio_feature_mode",
            "audio_role",
            "fallback_audio_role",
        ),
        promotion_jobs=TRANSFORM_PROMOTION_JOBS,
        training_requirements=(
            "source_disjoint_train_calibration_test/v2",
            "strum_owned_decoder_calibration/v1",
            "strum_owned_calibration_checkpoint_selection/v1",
            "test_only_transform_promotion/v1",
        ),
    ),
    PipelineDescriptor(
        id="vocals.harmony-source-policy/v1",
        display_name="Vocal harmony isolated-source policy",
        kind="dataset_policy",
        version=1,
        catalog_requirements={
            "instrument": "vocals",
            "difficulties": ["expert"],
            "label_tracks": ["HARM1", "HARM2", "HARM3"],
            "audio_roles": ["harm1", "harm2", "harm3"],
            "audio_policy": "isolated_harmony_only:no_fallback_to_vocals_or_mix",
            "source_policy_format": "octave-vocal-harmony-source-policy/v1",
            "source_provenance": [
                "isolated_source_stem/v1",
                "isolated_separation_output/v1",
            ],
        },
        prepare_schema=VOCAL_HARMONY_SOURCE_PREPARE_SCHEMA,
        train_schema=None,
        checkpoint_outputs=(),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="not_available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=("harmony_tracks",),
        training_requirements=(
            "vocal_harmony_model_component/v1",
            "vocal_chart_composition_contract/v1",
            "vocal_held_out_chart_evaluation/v1",
            "vocal_profile_package/v1",
            "vocal_chart_execution/v1",
        ),
    ),
    PipelineDescriptor(
        id="vocals.note-activity/v1",
        display_name="Vocals activity + pitch",
        kind="audio_to_vocal_labels",
        version=1,
        catalog_requirements={
            "instrument": "vocals",
            "difficulties": ["expert"],
            "audio_roles": ["vocals", "mix"],
            "audio_policy": "prefer:vocals,fallback:mix",
            "label_tracks": ["PART VOCALS"],
            "label_outputs": ["pitched_vocal_activity", "midi_pitch_36_84"],
        },
        prepare_schema=VOCALS_ACTIVITY_PREPARE_SCHEMA,
        train_schema=VOCALS_ACTIVITY_TRAIN_SCHEMA,
        checkpoint_outputs=("vocals.frame_activity_pitch",),
        # A playable chart also needs phrase/lyric/talky stages and an
        # instrument-specific evaluation/profile contract.
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        ),
        training_requirements=(
            "vocal_phrase_boundary_component/v1",
            "vocal_lyric_tokenizer_and_alignment/v1",
            "vocal_talky_activity_component/v1",
            "vocal_harmony_source_policy/v1",
            "vocal_chart_composition_contract/v1",
            "vocal_held_out_chart_evaluation/v1",
            "vocal_profile_package/v1",
        ),
    ),
    PipelineDescriptor(
        id="vocals.phrase-boundaries/v1",
        display_name="Vocals phrase boundaries",
        kind="audio_to_vocal_labels",
        version=1,
        catalog_requirements={
            "instrument": "vocals",
            "difficulties": ["expert"],
            "audio_roles": ["vocals", "mix"],
            "audio_policy": "prefer:vocals,fallback:mix",
            "label_tracks": ["PART VOCALS"],
            "label_outputs": ["lead_phrase_start", "lead_phrase_end"],
            "label_conventions": [
                "midi_105_start_marker",
                "midi_106_end_marker",
                "midi_105_sustained_span_end",
            ],
        },
        prepare_schema=VOCALS_ACTIVITY_PREPARE_SCHEMA,
        train_schema=VOCALS_ACTIVITY_TRAIN_SCHEMA,
        checkpoint_outputs=("vocals.phrase_boundaries",),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        ),
        training_requirements=(
            "vocal_activity_pitch_component/v1",
            "vocal_lyric_tokenizer_and_alignment/v1",
            "vocal_talky_activity_component/v1",
            "vocal_harmony_source_policy/v1",
            "vocal_chart_composition_contract/v1",
            "vocal_held_out_chart_evaluation/v1",
            "vocal_profile_package/v1",
        ),
    ),
    PipelineDescriptor(
        id="vocals.lyric-alignment/v1",
        display_name="Vocals lyric alignment",
        kind="audio_to_vocal_labels",
        version=1,
        catalog_requirements={
            "instrument": "vocals",
            "difficulties": ["expert"],
            "audio_roles": ["vocals", "mix"],
            "audio_policy": "prefer:vocals,fallback:mix",
            "label_tracks": ["PART VOCALS"],
            "label_outputs": ["observed_lyric_character_tokens", "observed_lyric_event_alignment"],
            "label_conventions": ["part-vocals-lyrics-or-text-meta-events/v1"],
        },
        prepare_schema=VOCALS_ACTIVITY_PREPARE_SCHEMA,
        train_schema=VOCALS_ACTIVITY_TRAIN_SCHEMA,
        checkpoint_outputs=("vocals.lyric_alignment",),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        ),
        training_requirements=(
            "vocal_activity_pitch_component/v1",
            "vocal_phrase_boundary_component/v1",
            "vocal_talky_activity_component/v1",
            "vocal_harmony_source_policy/v1",
            "vocal_chart_composition_contract/v1",
            "vocal_held_out_chart_evaluation/v1",
            "vocal_profile_package/v1",
        ),
    ),
    PipelineDescriptor(
        id="vocals.talky-activity/v1",
        display_name="Vocals pitchless/talky activity",
        kind="audio_to_vocal_labels",
        version=1,
        catalog_requirements={
            "instrument": "vocals",
            "difficulties": ["expert"],
            "audio_roles": ["vocals", "mix"],
            "audio_policy": "prefer:vocals,fallback:mix",
            "label_tracks": ["PART VOCALS"],
            "label_outputs": ["pitchless_talky_activity"],
            "label_conventions": ["part-vocals-midi-note-96-span/v1"],
        },
        prepare_schema=VOCALS_ACTIVITY_PREPARE_SCHEMA,
        train_schema=VOCALS_ACTIVITY_TRAIN_SCHEMA,
        checkpoint_outputs=("vocals.talky_activity",),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        ),
        training_requirements=(
            "vocal_activity_pitch_component/v1",
            "vocal_phrase_boundary_component/v1",
            "vocal_lyric_tokenizer_and_alignment/v1",
            "vocal_harmony_source_policy/v1",
            "vocal_chart_composition_contract/v1",
            "vocal_held_out_chart_evaluation/v1",
            "vocal_profile_package/v1",
        ),
    ),
    PipelineDescriptor(
        id="drums.onset-classifier/v1",
        display_name="Drums onset + velocity",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "drums",
            "difficulties": ["expert"],
            "audio_roles": ["drums", "mix"],
            "audio_policy": "prefer:drums,fallback:mix",
        },
        prepare_schema=_object_schema(CATALOG_AUDIO_OPTIONS),
        train_schema=DRUMS_ONSET_TRAIN_SCHEMA,
        checkpoint_outputs=("drums_onset_classifier",),
        inference_capability=None,
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
        private_request_fields=("catalog_root",),
        catalog_inspection_option_keys=(
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        ),
        training_requirements=("drums_chart_execution_profile",),
        promotion_jobs=DRUMS_PROMOTION_JOBS,
    ),
    *(
        PipelineDescriptor(
            id=pipeline_id,
            display_name=task_kind.replace("_", " ").title(),
            kind="derived_labels"
            if task_kind.startswith(("fret_mapper", "section_"))
            else "audio_to_chart",
            version=1,
            catalog_requirements={
                "instrument": task_kind.replace("fret_mapper_", "").replace("section_", ""),
                "difficulties": ["expert"],
                "audio_policy": "task-specific managed role with mix fallback",
                # Bass and Keys generic source contracts must not make OCTAVE
                # reconstruct their audio selection from an implementation
                # detail.  The concrete five-lane worker has the same
                # preferred/fallback contract, but remains a separate path.
                **(
                    {
                        "audio_roles": list(CATALOG_TASK_DEFAULT_AUDIO_ROLES[task_kind]),
                        "audio_policy": "prefer:"
                        + CATALOG_TASK_DEFAULT_AUDIO_ROLES[task_kind][0]
                        + ",fallback:"
                        + CATALOG_TASK_DEFAULT_AUDIO_ROLES[task_kind][1],
                    }
                    if task_kind in INSTRUMENT_CHART_TRAINING_CONTRACTS
                    else {}
                ),
                **(
                    {
                        "label_schema": CATALOG_TASK_LABEL_SCHEMAS[task_kind]["id"],
                        "label_tracks": CATALOG_TASK_LABEL_SCHEMAS[task_kind].get(
                            "track_names",
                            CATALOG_TASK_LABEL_SCHEMAS[task_kind].get("track_prefixes", []),
                        ),
                        **(
                            {
                                "prepared_task_view_format": "strum-pro-target-task-manifest/v1",
                                "prepared_target_encoding": "strum-pro-midi-target-decoder/v1",
                                "prepared_audio_preprocessing": "pro-logmel-event-windows/v1",
                            }
                            if task_kind in PRO_TRAINING_CONTRACTS
                            else {}
                        ),
                    }
                    if task_kind
                    in {*PRO_TRAINING_CONTRACTS, *INSTRUMENT_CHART_TRAINING_CONTRACTS, "vocals"}
                    else {}
                ),
            },
            prepare_schema=_object_schema(
                {
                    **CATALOG_AUDIO_OPTIONS,
                    "disable_fallback": {"type": "boolean", "default": False},
                    "split_ratios": {"type": "array", "items": {"type": "integer"}},
                    "split_seed": {"type": "string", "default": "catalog-source-id/v1"},
                    "preprocessing": {"type": "object", "default": {}},
                }
            ),
            train_schema=(
                PRO_EVENT_CANDIDATE_TRAIN_SCHEMA
                if task_kind in PRO_TRAINING_CONTRACTS
                else FRET_MAPPER_TRAIN_SCHEMA
                if task_kind.startswith("fret_mapper_")
                else SECTION_TRAIN_SCHEMA
                if task_kind.startswith("section_")
                else None
            ),
            checkpoint_outputs=(
                (f"fret_mapper.{task_kind.removeprefix('fret_mapper_')}",)
                if task_kind.startswith("fret_mapper_")
                else (f"section_classifier.{task_kind.removeprefix('section_')}",)
                if task_kind.startswith("section_")
                # A planned generic instrument-chart descriptor does not have
                # a stable component identity.  Its concrete implementation
                # path declares the actual component pair in the training
                # contract instead of misleading a host with ``bass``/``keys``
                # pseudo-component names.  The planned Vocal descriptor has
                # the same boundary: its narrow lead-component workers expose
                # their own outputs, while the generic chart contract cannot
                # claim a fictitious ``vocals`` checkpoint before a composed
                # trainer and handler exist.
                else ()
                if task_kind in {*INSTRUMENT_CHART_TRAINING_CONTRACTS, "vocals"}
                # Pro has two selectable candidate kinds whose components are
                # intentionally not interchangeable.  Their exact output
                # identities live in ``checkpoint_output_contracts`` below;
                # exposing one static component here would misdescribe the
                # free-running proposal candidate.
                else ()
                if task_kind in PRO_TRAINING_CONTRACTS
                else (task_kind,)
            ),
            inference_capability=None,
            status="catalog_ready",
            preparation_status="available",
            training_status=(
                "available"
                if task_kind in PRO_TRAINING_CONTRACTS
                or task_kind.startswith(("fret_mapper_", "section_"))
                else "planned"
            ),
            checkpoint_output_contracts=(
                _pro_candidate_checkpoint_output_contracts(task_kind)
                if task_kind in PRO_TRAINING_CONTRACTS
                else None
            ),
            # Every catalog task view requires a private catalog root during
            # preparation.  Training-only private fields must never be
            # inferred from source layout by an OCTAVE renderer.
            private_request_fields=("catalog_root",),
            catalog_inspection_option_keys=(
                "audio_role",
                "fallback_audio_role",
                "disable_fallback",
                "required_difficulty",
            ),
            training_requirements=(
                (
                    "strum_pitch_extra",
                    "instrument_specific_profile_evaluation",
                    "instrument_specific_profile_packaging",
                )
                if task_kind.startswith("fret_mapper_")
                else (*PLANNED_TRAINING_REQUIREMENTS[task_kind],)
                if task_kind.startswith("section_")
                else PLANNED_TRAINING_REQUIREMENTS.get(task_kind, ())
            ),
            training_contract=(
                VOCALS_TRAINING_CONTRACT
                if task_kind == "vocals"
                else INSTRUMENT_CHART_TRAINING_CONTRACTS.get(
                    task_kind, PRO_TRAINING_CONTRACTS.get(task_kind)
                )
            ),
            promotion_jobs=(
                _section_promotion_jobs(task_kind.removeprefix("section_"))
                if task_kind.startswith("section_")
                else ()
            ),
        )
        for task_kind, pipeline_id in sorted(CATALOG_TASK_PIPELINES.items())
        if task_kind
        not in {
            "bass_onset_fret",
            "keys_onset_fret",
            "vocals_activity",
            "vocals_phrase_boundaries",
            "vocals_lyric_alignment",
            "vocals_talky_activity",
        }
    ),
)


class WorkerRequestError(ValueError):
    """Raised for an invalid host-to-worker request before any task is written."""


def _revision() -> tuple[str | None, bool | None]:
    """Return path-safe source identity and a conservative dirty-state value.

    A configured revision is only trusted when it is a canonical Git object
    identity.  It carries no trustworthy dirty state unless the caller also
    supplies the explicit ``STRUM_SOURCE_DIRTY=0`` or ``=1`` attestation.
    For a Git checkout, tracked changes anywhere and untracked executable
    source below ``src/`` or ``scripts/`` make the state dirty.  A failed status
    check produces ``None`` rather than claiming a clean source tree.
    """
    configured_raw = os.environ.get("STRUM_SOURCE_REVISION")
    if configured_raw is not None:
        configured = source_revision_identity(configured_raw)
        if configured is None:
            # Do not return hostile or incomplete configuration in a portable
            # runtime or bundle payload.  An explicitly empty value is an
            # invalid attestation too: falling back to Git would make it
            # inconsistent with ModelBundle's fail-closed pin validation.
            return None, None
        configured_dirty = os.environ.get("STRUM_SOURCE_DIRTY")
        if configured_dirty == "1":
            return configured, True
        if configured_dirty == "0":
            return configured, False
        return configured, None
    revision: str | None = None
    try:
        revision = source_revision_identity(
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        if revision is None:
            return None, None
        tracked_dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        executable_untracked_dirty = bool(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(PROJECT_ROOT),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                    "--",
                    "src",
                    "scripts",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, tracked_dirty or executable_untracked_dirty
    except (OSError, subprocess.CalledProcessError):
        # ``rev-parse`` and ``status`` have different evidentiary value.  A
        # revision obtained before status fails is still useful, but it must
        # never be represented as a clean source tree.
        return revision, None


def _runtime_payload() -> dict[str, object]:
    revision, dirty = _revision()
    capabilities = [
        "pipeline_discovery",
        "catalog_inspect",
        "dataset_prepare",
        "training_start",
        "post_train_job_discovery",
        "post_train_job_start",
        "chart_preflight",
        "chart_run",
        "typed_chart_results",
        "model_bundle_preflight",
        "checkpoint_discovery",
        "checkpoint_inspect",
        "checkpoint_package",
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "runtime": {
            "id": f"strum-{__version__}" + (f"+git.{revision[:12]}" if revision else ""),
            "version": __version__,
            "source_revision": revision,
            "source_dirty": dirty,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "python_requires": ">=3.11",
        "device_support": ["cuda", "mps", "cpu"],
        "capabilities": capabilities,
        "model_bundle_schema_versions": list(MODEL_BUNDLE_SCHEMA_VERSIONS),
        "chart_result_schema_versions": [1],
        "chart_result_formats": [CHART_PREFLIGHT_FORMAT, CHART_RUN_FORMAT],
        "optional_dependencies": {
            "basic_pitch": {
                "available": importlib.util.find_spec("basic_pitch") is not None,
                "required_by": [
                    "guitar.hybrid-v2-rule/v1",
                    "strum.fret-mapper/guitar/v1",
                    "strum.fret-mapper/bass/v1",
                ],
            }
        },
        "pipelines": [
            pipeline.id for pipeline in PIPELINES if pipeline.preparation_status == "available"
        ],
    }


def _manifest_sha256(bundle: ModelBundle) -> str | None:
    if bundle.manifest_path is None:
        return None
    return hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preflight_bundle(
    path: str | Path,
    *,
    required_components: Sequence[str] = (),
) -> dict[str, object]:
    """Validate one portable bundle without deserializing any model weights.

    Explicitly selected bundles require an on-disk checkpoint, SHA-256, and
    byte length for every required component. This is deliberately stricter
    than the legacy fallback layout, which is not deployable through this API.
    """
    bundle = load_model_bundle(path, check_files=True)
    errors = bundle.validate(check_files=True, verify_hashes=True)
    selected = tuple(required_components) or tuple(
        name for name, component in bundle.components.items() if component.required
    )
    components: list[dict[str, object]] = []
    for name in selected:
        component = bundle.component(name)
        if component is None:
            errors.append(f"required component is not declared: {name}")
            continue
        if component.checkpoint is None:
            errors.append(f"{name}: required component has no checkpoint")
            continue
        if component.sha256 is None:
            errors.append(f"{name}: deployable component requires sha256")
        if component.byte_length is None:
            errors.append(f"{name}: deployable component requires byte_length")
        if component.config is not None and component.config_sha256 is None:
            errors.append(f"{name}: deployable component config requires sha256")
        if component.config is not None and component.config_byte_length is None:
            errors.append(f"{name}: deployable component config requires byte_length")
        components.append(
            {
                "id": name,
                "sha256": component.sha256,
                "byte_length": component.byte_length,
                "architecture": component.architecture,
                "preprocessing": component.preprocessing,
                "config_sha256": component.config_sha256,
                "config_byte_length": component.config_byte_length,
            }
        )
    if errors:
        raise BundleValidationError("; ".join(errors))
    return {
        "status": "ready",
        "model_id": bundle.model_id,
        "manifest_sha256": _manifest_sha256(bundle),
        "components": components,
        "compatibility": bundle.compatibility,
    }


def validate_inference_profile(
    path: str | Path,
    *,
    profile_id: str,
    difficulty_policy: str,
) -> dict[str, object]:
    """Resolve a declared inference profile and verify all required model files."""
    bundle = load_model_bundle(path, check_files=True)
    profile: InferenceProfile | None = bundle.profile(profile_id)
    if profile is None:
        raise BundleValidationError(f"inference profile is not declared: {profile_id}")
    if difficulty_policy not in profile.difficulty_policies:
        raise BundleValidationError(
            f"profile {profile_id} does not support difficulty policy {difficulty_policy}"
        )
    plan = preflight_bundle(path, required_components=profile.required_components)
    if profile.capability == "difficulty.transform/v1":
        # Generic manifest hashes are not an admission proof for a transform.
        # Its promoted configuration must bind an immutable held-out report
        # before either profile validation or chart preflight can proceed.
        from src.chart_transform_profile import (  # noqa: PLC0415
            validate_promoted_chart_transform_profile,
        )

        validate_promoted_chart_transform_profile(bundle, profile.profile_id)
    return {
        **plan,
        "profile_id": profile.profile_id,
        "capability": profile.capability,
        "instruments": list(profile.instruments),
        "difficulty_policy": difficulty_policy,
        "required_companions": [
            bundle.companions[companion].as_json() for companion in profile.required_companions
        ],
        "profile_configuration_sha256": profile.configuration_sha256,
        "profile_configuration_byte_length": profile.configuration_byte_length,
        **({"composition": profile.graph.as_json()} if profile.graph is not None else {}),
    }


def _model_bundle_artifact_id(bundle: ModelBundle) -> str:
    """Return a stable opaque identity for a discovered bundle.

    The manifest digest lets a host keep its local model-root lookup private.
    A model name is useful display metadata, but it must never be used as a
    filesystem location or as the selected-bundle authority.
    """
    manifest_sha256 = _manifest_sha256(bundle)
    if manifest_sha256 is None:
        raise BundleValidationError("model bundle has no manifest identity")
    return f"strum-model-bundle/{manifest_sha256}"


def _verified_chart_run_profile(bundle: ModelBundle, plan: dict[str, object]) -> tuple[str, str]:
    """Revalidate the immutable preflight profile tuple before execution.

    A direct chart run has to reopen its bundle before deserializing weights.
    Its preflight request is writable, however, so it must not become a second
    source of profile authority.  Bind that reopened bundle to the preflight
    plan's manifest and profile tuple, and return the *plan* profile ID for
    every capability-specific loader below.
    """
    expected_manifest = plan.get("manifest_sha256")
    actual_manifest = _manifest_sha256(bundle)
    if not isinstance(expected_manifest, str) or not isinstance(actual_manifest, str):
        raise WorkerRequestError("chart profile bundle has no manifest identity")
    if actual_manifest != expected_manifest:
        raise WorkerRequestError("chart profile bundle identity does not match preflight")

    profile_id = plan.get("profile_id")
    capability = plan.get("capability")
    difficulty_policy = plan.get("difficulty_policy")
    instruments = plan.get("instruments")
    configuration_sha256 = plan.get("profile_configuration_sha256")
    if (
        not isinstance(profile_id, str)
        or not isinstance(capability, str)
        or not isinstance(difficulty_policy, str)
        or not isinstance(configuration_sha256, str)
        or not isinstance(instruments, list)
        or not instruments
        or not all(isinstance(instrument, str) and instrument for instrument in instruments)
    ):
        raise WorkerRequestError("chart preflight plan has invalid profile identity")

    refreshed = validate_inference_profile(
        bundle.root,
        profile_id=profile_id,
        difficulty_policy=difficulty_policy,
    )
    if (
        refreshed.get("profile_id") != profile_id
        or refreshed.get("capability") != capability
        or refreshed.get("difficulty_policy") != difficulty_policy
        or refreshed.get("profile_configuration_sha256") != configuration_sha256
    ):
        raise WorkerRequestError("chart profile identity does not match preflight")
    available_instruments = refreshed.get("instruments")
    if (
        not isinstance(available_instruments, list)
        or not all(isinstance(instrument, str) for instrument in available_instruments)
        or not set(instruments) <= set(available_instruments)
    ):
        raise WorkerRequestError("chart profile instruments do not match preflight")
    return profile_id, actual_manifest


def _safe_compatibility_summary(bundle: ModelBundle) -> dict[str, object]:
    """Return the compatibility keys defined by the portable bundle schema.

    Manifests may contain producer-specific compatibility annotations.  They
    remain private to the manifest: discovery exposes only the runtime gates
    understood by STRUM, never an arbitrary value that could be a local path.
    """
    return {
        key: bundle.compatibility[key]
        for key in ("manifest_schema", "strum_version", "strum_revision", "strum_source_dirty")
        if key in bundle.compatibility
    }


def _profile_discovery_record(
    bundle: ModelBundle, profile: InferenceProfile
) -> dict[str, object] | None:
    """Return one fully hash-verified profile record, or omit an invalid one.

    Discovering a bundle must not turn a declared profile into a deployment
    candidate merely because its manifest parsed.  This validates exactly the
    profile's companion components without loading tensor data.  A valid
    profile may still be non-executable when STRUM has not declared a chart
    handler for its capability; OCTAVE can show that distinction but cannot
    select it as an auto-chart default.
    """
    try:
        preflight_bundle(bundle.root, required_components=profile.required_components)
    except BundleValidationError:
        return None
    executable_policies = [
        policy
        for policy in profile.difficulty_policies
        if _chart_execution_available(
            capability=profile.capability,
            difficulty_policy=policy,
            instruments=profile.instruments,
        )
    ]
    if executable_policies and not _executable_profile_contract_is_valid(bundle, profile):
        executable_policies = []
    return {
        **bundle.profile_summary(profile),
        "profile_configuration_sha256": profile.configuration_sha256,
        "profile_configuration_byte_length": profile.configuration_byte_length,
        "execution": {
            "status": "available" if executable_policies else "not_available",
            "difficulty_policies": executable_policies,
        },
    }


def inspect_model_bundle(path: str | Path) -> dict[str, object]:
    """Inspect one selected bundle with no model-root or checkpoint paths.

    Unlike the older metadata-only inspection command, this checks manifest
    file hashes before exposing a candidate.  It deliberately keeps valid but
    profile-less experiment bundles visible as ``not_deployable`` so a host
    can explain why a training result cannot be selected for auto-charting.
    """
    bundle = load_model_bundle(path, check_files=True)
    preflight = preflight_bundle(bundle.root)
    profiles = [
        record
        for profile in sorted(bundle.profiles.values(), key=lambda item: item.profile_id)
        if (record := _profile_discovery_record(bundle, profile)) is not None
    ]
    return {
        "format": MODEL_BUNDLE_INSPECTION_FORMAT,
        "status": "ready",
        "artifact_id": _model_bundle_artifact_id(bundle),
        "model_id": bundle.model_id,
        "manifest_sha256": preflight["manifest_sha256"],
        "schema_version": bundle.schema_version,
        "compatibility": _safe_compatibility_summary(bundle),
        "components": [
            {
                "id": component["id"],
                "sha256": component["sha256"],
                "byte_length": component["byte_length"],
            }
            for component in preflight["components"]
            if isinstance(component, dict)
        ],
        "profiles": profiles,
        "rejected_profile_count": len(bundle.profiles) - len(profiles),
        "deployment_status": (
            "ready"
            if any(
                isinstance(profile.get("execution"), dict)
                and profile["execution"].get("status") == "available"
                for profile in profiles
            )
            else "not_deployable"
        ),
    }


def _discover_model_manifest_paths(path: str | Path) -> tuple[list[Path], bool]:
    """Find bounded manifests below one host-selected private model root."""
    root = Path(path).expanduser().resolve()
    if root.is_file():
        if root.name != MANIFEST_FILENAME:
            raise BundleValidationError("model discovery requires a bundle directory or manifest")
        return [root], False
    if not root.is_dir():
        raise BundleValidationError("model discovery root is not a directory")

    manifests: list[Path] = []
    seen: set[Path] = set()
    truncated = False
    for current, directories, files in os.walk(root, followlinks=False):
        current_path = Path(current)
        try:
            relative = current_path.relative_to(root)
        except ValueError:
            directories[:] = []
            continue
        directories[:] = [
            name
            for name in directories
            if name not in MODEL_DISCOVERY_IGNORED_DIRECTORIES
            and not (current_path / name).is_symlink()
        ]
        if len(relative.parts) >= MAX_MODEL_DISCOVERY_DEPTH:
            directories[:] = []
        if MANIFEST_FILENAME not in files:
            continue
        manifest = (current_path / MANIFEST_FILENAME).resolve()
        try:
            manifest.relative_to(root)
        except ValueError:
            continue
        if manifest in seen:
            continue
        seen.add(manifest)
        manifests.append(manifest)
        if len(manifests) >= MAX_DISCOVERED_MODEL_MANIFESTS:
            truncated = True
            break
    return sorted(manifests), truncated


def discover_model_bundles(path: str | Path) -> dict[str, object]:
    """Discover local bundle candidates without revealing their locations.

    The caller owns the private ``path`` to a user-selected checkpoint folder.
    It maps returned ``artifact_id`` values back to the selected roots in its
    main process before requesting a later preflight/chart job.  Invalid
    manifests are counted but not described, which avoids leaking filenames,
    parser errors, or component paths to renderer clients.
    """
    manifests, truncated = _discover_model_manifest_paths(path)
    candidates: list[dict[str, object]] = []
    rejected_bundle_count = 0
    for manifest in manifests:
        try:
            candidates.append(inspect_model_bundle(manifest))
        except (BundleValidationError, OSError):
            rejected_bundle_count += 1
    profile_count = sum(
        len(candidate["profiles"])
        for candidate in candidates
        if isinstance(candidate.get("profiles"), list)
    )
    return {
        "format": MODEL_BUNDLE_DISCOVERY_FORMAT,
        "status": "ready",
        "candidate_count": len(candidates),
        "profile_count": profile_count,
        "rejected_bundle_count": rejected_bundle_count,
        "truncated": truncated,
        "candidates": candidates,
    }


def _stage(
    *,
    status: str,
    required: bool,
    component_ids: Sequence[str] = (),
    difficulty: str | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    """Build one safe, typed chart-stage description.

    Stage records deliberately identify declared model components and output
    difficulty, but never their checkpoint locations or the caller's input or
    output locations.  The same shape is used for a preflight plan and its
    resulting chart-run manifest so OCTAVE can render partial work honestly.
    """
    stage: dict[str, object] = {
        "status": status,
        "required": required,
        "component_ids": list(component_ids),
    }
    if difficulty is not None:
        stage["difficulty"] = difficulty
    if reason is not None:
        stage["reason"] = reason
    return stage


def _copy_chart_contract(value: dict[str, object]) -> dict[str, object]:
    """Return a JSON-shaped copy before a run updates a preflight stage state."""
    return json.loads(json.dumps(value))


def _chart_transform_metadata(
    bundle: ModelBundle,
    component_id: str,
    *,
    requested_instruments: Sequence[str],
) -> tuple[str, str]:
    """Read the two safe transform identities required by the stage contract."""
    component = bundle.component(component_id)
    if component is None or component.config is None:
        raise WorkerRequestError("difficulty transform component is incomplete")
    try:
        config = json.loads(component.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("difficulty transform configuration is unreadable") from error
    if not isinstance(config, dict):
        raise WorkerRequestError("difficulty transform configuration must be an object")
    instrument = config.get("instrument")
    target_difficulty = config.get("target_difficulty")
    if (
        not isinstance(instrument, str)
        or not isinstance(target_difficulty, str)
        or not target_difficulty
        or tuple(requested_instruments) != (instrument,)
    ):
        raise WorkerRequestError("difficulty transform profile does not match its component")
    return instrument, target_difficulty


def _chart_result_contract(
    plan: dict[str, object],
    *,
    execution: str,
    transform_instrument: str | None = None,
    transform_target_difficulty: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """Describe every stage an executable profile will perform or intentionally omit.

    ``expert_only`` profiles always expose the omitted lower-difficulty stage.
    Learned transforms declare that Expert input is supplied by an earlier
    stage, then name the resulting target difficulty.  This prevents a caller
    from mistaking either narrow profile for a complete multi-instrument chart
    assembly pipeline.
    """
    instruments = plan["instruments"]
    capability = plan["capability"]
    policy = plan["difficulty_policy"]
    components = plan["components"]
    if not isinstance(instruments, list) or not all(isinstance(item, str) for item in instruments):
        raise AssertionError("validated chart plan must contain instruments")
    if not isinstance(capability, str) or not isinstance(policy, str):
        raise AssertionError("validated chart plan must contain capability and policy")
    component_ids = [
        component["id"]
        for component in components
        if isinstance(component, dict) and isinstance(component.get("id"), str)
    ]

    unavailable_reason = "execution_handler_not_declared"
    instrument_results: dict[str, object] = {}
    if capability == "difficulty.transform/v1":
        if transform_instrument is None or transform_target_difficulty is None:
            raise AssertionError("difficulty transform contract requires component metadata")
        stage_status = "ready" if execution == "available" else "unavailable"
        instrument_results[transform_instrument] = {
            "status": "ready" if execution == "available" else "not_available",
            "stages": {
                "expert_chart": _stage(
                    status="provided",
                    required=True,
                    difficulty="Expert",
                    reason="source_midi_required",
                ),
                "difficulty_transform": _stage(
                    status=stage_status,
                    required=True,
                    component_ids=component_ids,
                    difficulty=transform_target_difficulty,
                    reason=None if execution == "available" else unavailable_reason,
                ),
            },
        }
        difficulty = {
            "policy": policy,
            "status": "ready" if execution == "available" else "unavailable",
            "source_difficulty": "Expert",
            "target_difficulty": transform_target_difficulty,
        }
        return instrument_results, difficulty

    stage_name_by_capability = {
        "guitar.hybrid-v2-rule/v1": "expert_chart",
        "drums.v14-expert/v1": "expert_chart",
    }
    stage_name = stage_name_by_capability.get(capability, "expert_chart")
    stage_status = "ready" if execution == "available" else "unavailable"
    for instrument in instruments:
        instrument_results[instrument] = {
            "status": "ready" if execution == "available" else "not_available",
            "stages": {
                stage_name: _stage(
                    status=stage_status,
                    required=True,
                    component_ids=component_ids,
                    difficulty="Expert",
                    reason=None if execution == "available" else unavailable_reason,
                ),
                "difficulty_transform": _stage(
                    status="not_requested",
                    required=False,
                    difficulty="Expert",
                    reason="difficulty_policy_expert_only"
                    if policy == "expert_only"
                    else "profile_does_not_declare_transform",
                ),
            },
        }
    return instrument_results, {
        "policy": policy,
        "status": "expert_only" if policy == "expert_only" else "not_available",
        "source_difficulty": None,
        "target_difficulty": "Expert",
    }


def _resolved_profile_composition(
    plan: dict[str, object], *, execution: str
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Resolve a declarative profile graph without inventing an executor.

    The model-bundle graph is sufficient to validate every required component,
    dependency, and declared runtime companion at profile-preflight time.  It
    deliberately does not turn a graph into a chart handler: unless STRUM
    registers a handler for the exact profile capability, required stages stay
    ``unavailable`` and ``chart run`` remains fail-closed.
    """
    composition = plan.get("composition")
    if not isinstance(composition, dict):
        raise AssertionError("composed profile is missing its composition")
    stages = composition.get("stages")
    outputs = composition.get("outputs")
    policy = plan.get("difficulty_policy")
    requested_instruments = plan.get("instruments")
    if (
        not isinstance(stages, list)
        or not isinstance(outputs, list)
        or not isinstance(policy, str)
        or not isinstance(requested_instruments, list)
    ):
        raise AssertionError("validated profile composition has an invalid shape")

    resolved_stages: list[dict[str, object]] = []
    stage_statuses: dict[str, str] = {}
    for source_stage in stages:
        if not isinstance(source_stage, dict):
            raise AssertionError("validated profile composition has a non-object stage")
        stage = _copy_chart_contract(source_stage)
        stage_id = stage.get("id")
        required = stage.get("required")
        policies = stage.get("difficulty_policies", [])
        if (
            not isinstance(stage_id, str)
            or not isinstance(required, bool)
            or not isinstance(policies, list)
        ):
            raise AssertionError("validated profile composition has an invalid stage")
        selected = not policies or policy in policies
        if not selected:
            status = "not_requested"
            reason = "difficulty_policy_not_selected"
        elif not required:
            status = "not_requested"
            reason = "optional_stage_not_selected"
        elif execution == "available":
            status = "ready"
            reason = None
        else:
            status = "unavailable"
            reason = "execution_handler_not_declared"
        stage["status"] = status
        if reason is not None:
            stage["reason"] = reason
        stage_statuses[stage_id] = status
        resolved_stages.append(stage)

    resolved_outputs: list[dict[str, object]] = []
    terminal_stage_for_instrument: dict[str, str] = {}
    terminal_difficulty: dict[str, str] = {}
    for source_output in outputs:
        if not isinstance(source_output, dict):
            raise AssertionError("validated profile composition has a non-object output")
        output = _copy_chart_contract(source_output)
        instrument = output.get("instrument")
        stage_id = output.get("stage_id")
        difficulty = output.get("difficulty")
        if not all(isinstance(value, str) for value in (instrument, stage_id, difficulty)):
            raise AssertionError("validated profile composition has an invalid output")
        output["status"] = stage_statuses[stage_id]
        resolved_outputs.append(output)
        terminal_stage_for_instrument[instrument] = stage_id
        terminal_difficulty[instrument] = difficulty

    instrument_results: dict[str, object] = {}
    for instrument in requested_instruments:
        if not isinstance(instrument, str):
            raise AssertionError("validated chart plan has an invalid requested instrument")
        instrument_stages = {
            stage["id"]: {
                key: value
                for key, value in stage.items()
                if key
                in {
                    "status",
                    "required",
                    "component_ids",
                    "companion_ids",
                    "depends_on",
                    "difficulty",
                    "reason",
                }
            }
            for stage in resolved_stages
            if stage.get("instrument") == instrument
        }
        terminal_status = stage_statuses[terminal_stage_for_instrument[instrument]]
        instrument_results[instrument] = {
            "status": "ready" if terminal_status == "ready" else "not_available",
            "stages": instrument_stages,
        }

    target_difficulties = {
        terminal_difficulty[instrument]
        for instrument in requested_instruments
        if isinstance(instrument, str) and instrument in terminal_difficulty
    }
    target_difficulty = target_difficulties.pop() if len(target_difficulties) == 1 else None
    difficulty = {
        "policy": policy,
        "status": "ready" if execution == "available" else "not_available",
        "source_difficulty": None,
        "target_difficulty": target_difficulty,
    }
    return (
        {
            "format": composition["format"],
            "stages": resolved_stages,
            "outputs": resolved_outputs,
            "required_components": [component["id"] for component in plan["components"]],
            "required_companions": plan["required_companions"],
        },
        instrument_results,
        difficulty,
    )


def _chart_execution_available(
    *, capability: str, difficulty_policy: str, instruments: Sequence[str]
) -> bool:
    """Return true only for a worker handler with matching single-stage semantics."""
    expected = {
        "guitar.hybrid-v2-rule/v1": ("guitar", "expert_only"),
        "guitar.neural-v1-expert/v1": ("guitar", "expert_only"),
        "bass.neural-v1-expert/v1": ("bass", "expert_only"),
        "keys.neural-v1-expert/v1": ("keys", "expert_only"),
        "drums.v14-expert/v1": ("drums", "expert_only"),
    }
    if capability == "difficulty.transform/v1":
        return len(instruments) == 1 and difficulty_policy.startswith("learned:")
    required = expected.get(capability)
    return (
        required is not None
        and tuple(instruments) == (required[0],)
        and difficulty_policy == required[1]
    )


def _executable_profile_contract_is_valid(bundle: ModelBundle, profile: InferenceProfile) -> bool:
    """Verify the capability-specific config contract without deserializing tensors.

    A hash-valid generic profile is not necessarily an executable STRUM
    profile.  The Guitar, Bass, Keys, and Drums handlers each have a stricter
    configuration/companion contract, while a learned transform must bind its
    lone component and declared instrument.  Discovery uses this same narrow
    check before advertising an executable option.
    """
    try:
        if profile.capability == "guitar.hybrid-v2-rule/v1":
            from src.inference.guitar_hybrid_profile import (  # noqa: PLC0415
                load_guitar_hybrid_rule_profile,
            )

            load_guitar_hybrid_rule_profile(bundle, profile.profile_id)
        elif profile.capability == "guitar.neural-v1-expert/v1":
            from src.inference.guitar_neural_profile import (  # noqa: PLC0415
                load_guitar_neural_expert_profile,
            )

            load_guitar_neural_expert_profile(bundle, profile.profile_id)
        elif profile.capability == "bass.neural-v1-expert/v1":
            from src.inference.bass_neural_profile import (  # noqa: PLC0415
                load_bass_neural_expert_profile,
            )

            load_bass_neural_expert_profile(bundle, profile.profile_id)
        elif profile.capability == "keys.neural-v1-expert/v1":
            from src.inference.keys_neural_profile import (  # noqa: PLC0415
                load_keys_neural_expert_profile,
            )

            load_keys_neural_expert_profile(bundle, profile.profile_id)
        elif profile.capability == "drums.v14-expert/v1":
            from src.inference.drums_v14_profile import (  # noqa: PLC0415
                load_drums_v14_expert_profile,
            )

            load_drums_v14_expert_profile(bundle, profile.profile_id)
        elif profile.capability == "difficulty.transform/v1":
            if len(profile.required_components) != 1:
                return False
            component = bundle.component(profile.required_components[0])
            if component is None or component.architecture != "EventTransformMLP/v1":
                return False
            if profile.difficulty_policies != (f"learned:{component.name}",):
                return False
            from src.chart_transform_profile import (  # noqa: PLC0415
                validate_promoted_chart_transform_profile,
            )

            validate_promoted_chart_transform_profile(bundle, profile.profile_id)
            _chart_transform_metadata(
                bundle,
                component.name,
                requested_instruments=profile.instruments,
            )
        else:
            return False
    except (BundleValidationError, WorkerRequestError, OSError, ValueError):
        return False
    return True


def _complete_chart_stage(
    instrument_results: dict[str, object],
    difficulty: dict[str, object],
    *,
    instrument: str,
    stage_name: str,
    artifact_ids: Sequence[str],
) -> None:
    """Update one preflight stage for a completed run without adding locations."""
    instrument_result = instrument_results.get(instrument)
    if not isinstance(instrument_result, dict):
        raise AssertionError("chart preflight is missing the executed instrument")
    stages = instrument_result.get("stages")
    if not isinstance(stages, dict):
        raise AssertionError("chart preflight has invalid instrument stages")
    stage = stages.get(stage_name)
    if not isinstance(stage, dict) or stage.get("status") != "ready":
        raise AssertionError("chart preflight stage is not runnable")
    stage["status"] = "succeeded"
    stage["artifact_ids"] = list(artifact_ids)
    instrument_result["status"] = "succeeded"
    if stage_name == "difficulty_transform":
        difficulty["status"] = "succeeded"


def preflight_chart_request(request_path: Path) -> dict[str, object]:
    """Resolve a profile for a future chart job without loading model weights.

    This is intentionally a preflight-only boundary until each production
    auto-chart backend consumes declared bundle components rather than legacy
    working-directory defaults. A caller must not treat preflight success as an
    authorization to execute the legacy pipeline with arbitrary checkpoints.
    """
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart request is unreadable or not valid JSON") from error
    expected = {"model_root", "profile_id", "difficulty_policy", "instruments", "device"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WorkerRequestError("chart preflight request has unsupported fields")
    if not all(
        isinstance(raw[key], str) and raw[key]
        for key in ("model_root", "profile_id", "difficulty_policy", "device")
    ):
        raise WorkerRequestError("chart preflight identity fields must be non-empty strings")
    instruments = raw["instruments"]
    if (
        not isinstance(instruments, list)
        or not instruments
        or not all(isinstance(instrument, str) and instrument for instrument in instruments)
        or len(set(instruments)) != len(instruments)
    ):
        raise WorkerRequestError(
            "chart preflight instruments must be a unique non-empty string list"
        )
    plan = validate_inference_profile(
        raw["model_root"],
        profile_id=raw["profile_id"],
        difficulty_policy=raw["difficulty_policy"],
    )
    if not set(instruments) <= set(plan["instruments"]):
        raise WorkerRequestError("profile does not cover requested instruments")
    profile_configuration_sha256 = plan["profile_configuration_sha256"]
    transform_instrument = None
    transform_target_difficulty = None
    if plan["capability"] == "guitar.hybrid-v2-rule/v1":
        from src.inference.guitar_hybrid_profile import (
            load_guitar_hybrid_rule_profile,  # noqa: PLC0415
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_guitar_hybrid_rule_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "guitar.neural-v1-expert/v1":
        from src.inference.guitar_neural_profile import (  # noqa: PLC0415
            load_guitar_neural_expert_profile,
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_guitar_neural_expert_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "bass.neural-v1-expert/v1":
        from src.inference.bass_neural_profile import (  # noqa: PLC0415
            load_bass_neural_expert_profile,
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_bass_neural_expert_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "keys.neural-v1-expert/v1":
        from src.inference.keys_neural_profile import (  # noqa: PLC0415
            load_keys_neural_expert_profile,
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_keys_neural_expert_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "drums.v14-expert/v1":
        from src.inference.drums_v14_profile import (  # noqa: PLC0415
            load_drums_v14_expert_profile,
        )

        bundle = load_model_bundle(raw["model_root"], check_files=True)
        typed = load_drums_v14_expert_profile(bundle, raw["profile_id"])
        profile_configuration_sha256 = typed.configuration_sha256
    elif plan["capability"] == "difficulty.transform/v1":
        if (
            len(plan["components"]) != 1
            or plan["components"][0]["architecture"] != "EventTransformMLP/v1"
        ):
            raise WorkerRequestError(
                "difficulty transform requires exactly one EventTransformMLP/v1"
            )
        component_id = plan["components"][0]["id"]
        if plan["difficulty_policy"] != f"learned:{component_id}":
            raise WorkerRequestError("difficulty transform policy must name its declared component")
        bundle = load_model_bundle(raw["model_root"], check_files=True)
        transform_instrument, transform_target_difficulty = _chart_transform_metadata(
            bundle,
            component_id,
            requested_instruments=instruments,
        )
    execution = (
        "available"
        if _chart_execution_available(
            capability=plan["capability"],
            difficulty_policy=plan["difficulty_policy"],
            instruments=instruments,
        )
        else "not_available"
    )
    if "composition" in plan:
        composition, instrument_results, difficulty = _resolved_profile_composition(
            plan, execution=execution
        )
    else:
        composition = None
        instrument_results, difficulty = _chart_result_contract(
            plan,
            execution=execution,
            transform_instrument=transform_instrument,
            transform_target_difficulty=transform_target_difficulty,
        )
    response: dict[str, object] = {
        "schema_version": 1,
        "format": CHART_PREFLIGHT_FORMAT,
        "status": "ready",
        "execution": execution,
        "model_id": plan["model_id"],
        "profile_id": plan["profile_id"],
        "capability": plan["capability"],
        "difficulty_policy": plan["difficulty_policy"],
        "instruments": instruments,
        "device": raw["device"],
        "manifest_sha256": plan["manifest_sha256"],
        "components": plan["components"],
        "required_companions": plan["required_companions"],
        "profile_configuration_sha256": profile_configuration_sha256,
        "profile_configuration_byte_length": plan["profile_configuration_byte_length"],
        "instrument_results": instrument_results,
        "difficulty": difficulty,
    }
    if composition is not None:
        response["composition"] = composition
    return response


def _read_chart_run_request(request_path: Path, capability: str) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart run request is unreadable or not valid JSON") from error
    if capability in {
        "guitar.hybrid-v2-rule/v1",
        "guitar.neural-v1-expert/v1",
        "bass.neural-v1-expert/v1",
        "keys.neural-v1-expert/v1",
        "drums.v14-expert/v1",
    }:
        expected = {"preflight_request", "audio_path", "output_dir"}
    elif capability == "difficulty.transform/v1":
        expected = {"preflight_request", "source_midi_path", "song_path", "output_dir"}
    else:
        expected = {"preflight_request", "source_midi_path", "song_path", "output_dir", "threshold"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WorkerRequestError("chart run request has unsupported fields")
    required_locations = expected - {"song_path", "threshold"}
    if not all(isinstance(raw[key], str) and raw[key] for key in required_locations):
        raise WorkerRequestError("chart run request locations must be non-empty strings")
    if capability == "difficulty.transform/v1":
        if raw["song_path"] is not None and (
            not isinstance(raw["song_path"], str) or not raw["song_path"]
        ):
            raise WorkerRequestError("difficulty transform song_path must be a location or null")
    return raw


def _write_expert_five_lane_midi(chart: Any, output_path: Path, *, track_name: str) -> None:
    """Write one explicit Expert five-lane track with no difficulty fallback."""
    import mido  # noqa: PLC0415

    ticks_per_beat = 480
    tempo = mido.bpm2tempo(float(chart.tempo_bpm))
    messages: list[tuple[int, bool, int]] = []
    for note in chart.notes:
        start = round(float(note.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        end = round(
            (float(note.time_ms) + float(note.duration_ms))
            / 1000
            * ticks_per_beat
            * 1_000_000
            / tempo
        )
        messages.extend(
            ((start, True, 96 + int(note.fret)), (max(end, start + 1), False, 96 + int(note.fret)))
        )
    for chord in chart.chords:
        start = round(float(chord.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        end = round(
            (float(chord.time_ms) + float(chord.duration_ms))
            / 1000
            * ticks_per_beat
            * 1_000_000
            / tempo
        )
        for fret in chord.frets:
            messages.extend(
                ((start, True, 96 + int(fret)), (max(end, start + 1), False, 96 + int(fret)))
            )
    messages.sort(key=lambda event: (event[0], event[1]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name=track_name, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous = 0
    for tick, is_on, midi_note in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=midi_note,
                velocity=100 if is_on else 0,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _write_expert_guitar_midi(chart: Any, output_path: Path) -> None:
    """Write only Expert Guitar—lower difficulties need an explicit STRUM policy."""
    _write_expert_five_lane_midi(chart, output_path, track_name="PART GUITAR")


def _write_expert_bass_midi(chart: Any, output_path: Path) -> None:
    """Write only Expert Bass—lower difficulties need an explicit STRUM policy."""
    _write_expert_five_lane_midi(chart, output_path, track_name="PART BASS")


def _write_expert_keys_midi(chart: Any, output_path: Path) -> None:
    """Write only Expert Keys—lower difficulties require a STRUM policy."""
    _write_expert_five_lane_midi(chart, output_path, track_name="PART KEYS")


def _write_five_lane_midi(
    events: Sequence[dict[str, object]], *, instrument: str, difficulty: str, output_path: Path
) -> None:
    """Write one learned five-lane difficulty track with no implicit companion data."""
    import mido  # noqa: PLC0415

    from scripts.prepare_guitar_chart_pairs import (  # noqa: PLC0415
        DIFFICULTY_BASE_NOTES,
        FIVE_LANE_INSTRUMENT_TRACKS,
    )

    if instrument not in FIVE_LANE_INSTRUMENT_TRACKS or difficulty not in DIFFICULTY_BASE_NOTES:
        raise WorkerRequestError(
            "difficulty transform has unsupported instrument or target difficulty"
        )
    tempo, ticks_per_beat = 500_000, 480
    messages: list[tuple[int, bool, int]] = []
    for event in events:
        time_ms, lanes = event["time_ms"], event["lanes"]
        if not isinstance(time_ms, (int, float)) or not isinstance(lanes, list):
            raise WorkerRequestError("difficulty transform produced invalid event data")
        tick = round(float(time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        for lane in lanes:
            if not isinstance(lane, int) or not 0 <= lane < 5:
                raise WorkerRequestError("difficulty transform produced invalid lane data")
            note = DIFFICULTY_BASE_NOTES[difficulty] + lane
            messages.extend(((tick, True, note), (tick + 120, False, note)))
    messages.sort(key=lambda event: (event[0], event[1]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(
        mido.MetaMessage("track_name", name=FIVE_LANE_INSTRUMENT_TRACKS[instrument], time=0)
    )
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    previous = 0
    for tick, is_on, note in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=100 if is_on else 0,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _write_expert_drums_midi(events: Sequence[Any], output_path: Path) -> None:
    """Write only direct Expert Drums events from the typed V14 profile."""
    import mido  # noqa: PLC0415

    tempo, ticks_per_beat = 500_000, 480
    messages: list[tuple[int, bool, int, int]] = []
    for event in events:
        tick = round(float(event.time_ms) / 1000 * ticks_per_beat * 1_000_000 / tempo)
        messages.extend(
            ((tick, True, event.midi_note, event.velocity), (tick + 120, False, event.midi_note, 0))
        )
    messages.sort(key=lambda item: (item[0], item[1], item[2]))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    track = mido.MidiTrack(
        [
            mido.MetaMessage("track_name", name="PART DRUMS", time=0),
            mido.MetaMessage("set_tempo", tempo=tempo, time=0),
        ]
    )
    midi.tracks.append(track)
    previous = 0
    for tick, is_on, note, velocity in messages:
        track.append(
            mido.Message(
                "note_on" if is_on else "note_off",
                note=note,
                velocity=velocity,
                time=tick - previous,
            )
        )
        previous = tick
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(output_path)


def _run_without_legacy_output(callback: Any) -> Any:
    """Run a legacy inference callable without leaking private paths to stdout."""
    # Some optional inference dependencies keep logging handlers bound to the
    # process file descriptors, bypassing ``redirect_stdout``. Redirect both
    # descriptors for the short, synchronous call so the worker always emits
    # exactly one JSON response after it has completed.
    output_fds = {1, 2}
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(OSError, io.UnsupportedOperation):
            output_fds.add(stream.fileno())
    saved_fds = {fd: os.dup(fd) for fd in output_fds}
    try:
        with open(os.devnull, "w", encoding="utf-8") as sink:
            sys.stdout.flush()
            sys.stderr.flush()
            for fd in output_fds:
                os.dup2(sink.fileno(), fd)
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                return callback()
    finally:
        for fd, saved_fd in saved_fds.items():
            os.dup2(saved_fd, fd)
            os.close(saved_fd)


def run_chart_request(request_path: Path) -> dict[str, object]:
    """Execute a declared, bundle-backed chart profile with no legacy fallbacks."""
    try:
        raw_request = json.loads(request_path.read_text(encoding="utf-8"))
        preflight_request = (
            raw_request.get("preflight_request") if isinstance(raw_request, dict) else None
        )
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart run request is unreadable or not valid JSON") from error
    if not isinstance(preflight_request, str) or not preflight_request:
        raise WorkerRequestError("chart run request requires a preflight request")
    plan = preflight_chart_request(Path(preflight_request))
    request = _read_chart_run_request(request_path, plan["capability"])
    if plan["execution"] != "available":
        raise WorkerRequestError("profile has no worker chart execution handler")
    instrument_results = _copy_chart_contract(plan["instrument_results"])
    difficulty = _copy_chart_contract(plan["difficulty"])
    try:
        preflight_raw = json.loads(Path(request["preflight_request"]).read_text(encoding="utf-8"))
        bundle = load_model_bundle(preflight_raw["model_root"], check_files=True)
        errors = bundle.validate(check_files=True, verify_hashes=True)
        if errors:
            raise BundleValidationError("; ".join(errors))
        profile_id, manifest_sha256 = _verified_chart_run_profile(bundle, plan)
        output_dir = Path(request["output_dir"])
        if plan["capability"] == "guitar.hybrid-v2-rule/v1":
            audio = Path(request["audio_path"])
            if not audio.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from src.inference.guitar_hybrid_profile import (  # noqa: PLC0415
                load_guitar_hybrid_rule_profile,
            )
            from src.inference.guitar_hybrid_v2 import transcribe_guitar_hybrid  # noqa: PLC0415

            profile = load_guitar_hybrid_rule_profile(bundle, profile_id)
            chart = _run_without_legacy_output(
                lambda: transcribe_guitar_hybrid(
                    audio,
                    device=plan["device"],
                    execution_profile=profile,
                    model_bundle=bundle,
                )
            )
            midi_path = output_dir / "notes.mid"
            _write_expert_guitar_midi(chart, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {
                "guitar": {
                    "status": "succeeded",
                    "expert_event_count": len(chart.notes) + len(chart.chords),
                }
            }
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument="guitar",
                stage_name="expert_chart",
                artifact_ids=("notes_midi",),
            )
            response = {
                "output_name": midi_path.name,
                "expert_event_count": len(chart.notes) + len(chart.chords),
            }
        elif plan["capability"] == "guitar.neural-v1-expert/v1":
            audio_path = Path(request["audio_path"])
            if not audio_path.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from scripts.preprocess_guitar_windows import load_audio_mono_22050  # noqa: PLC0415
            from src.inference.guitar_bass import (  # noqa: PLC0415
                GuitarChart,
                GuitarChord,
                GuitarNote,
            )
            from src.inference.guitar_neural import GuitarNeuralCharter  # noqa: PLC0415
            from src.inference.guitar_neural_profile import (  # noqa: PLC0415
                load_guitar_neural_expert_profile,
            )

            audio = load_audio_mono_22050(audio_path)
            if audio is None:
                raise WorkerRequestError("chart input audio is unreadable")
            profile = load_guitar_neural_expert_profile(bundle, profile_id)
            events = _run_without_legacy_output(
                lambda: GuitarNeuralCharter.from_bundle_profile(
                    bundle, profile, device=plan["device"]
                ).transcribe(
                    audio,
                    onset_threshold=profile.onset_threshold,
                    min_distance_frames=profile.peak_min_distance_frames,
                    fret_thresholds_per_bit=profile.fret_thresholds,
                )
            )
            chart = GuitarChart(tempo_bpm=120.0, instrument="guitar")
            for event in events:
                if len(event.frets) >= 2:
                    chart.chords.append(
                        GuitarChord(
                            time_ms=event.time_sec * 1000.0,
                            frets=list(event.frets),
                            duration_ms=profile.note_duration_ms,
                        )
                    )
                elif event.frets:
                    chart.notes.append(
                        GuitarNote(
                            time_ms=event.time_sec * 1000.0,
                            fret=event.frets[0],
                            duration_ms=profile.note_duration_ms,
                        )
                    )
            midi_path = output_dir / "notes.mid"
            _write_expert_guitar_midi(chart, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {
                "guitar_neural": {
                    "status": "succeeded",
                    "expert_event_count": len(events),
                    "evaluation_sha256": profile.evaluation_sha256,
                }
            }
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument="guitar",
                stage_name="expert_chart",
                artifact_ids=("notes_midi",),
            )
            response = {"output_name": midi_path.name, "expert_event_count": len(events)}
        elif plan["capability"] == "bass.neural-v1-expert/v1":
            audio_path = Path(request["audio_path"])
            if not audio_path.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from scripts.preprocess_guitar_windows import load_audio_mono_22050  # noqa: PLC0415
            from src.inference.bass_neural_profile import (  # noqa: PLC0415
                BassNeuralCharter,
                load_bass_neural_expert_profile,
            )
            from src.inference.guitar_bass import (  # noqa: PLC0415
                GuitarChart,
                GuitarChord,
                GuitarNote,
            )

            audio = load_audio_mono_22050(audio_path)
            if audio is None:
                raise WorkerRequestError("chart input audio is unreadable")
            profile = load_bass_neural_expert_profile(bundle, profile_id)
            events = _run_without_legacy_output(
                lambda: BassNeuralCharter.from_bundle_profile(
                    bundle, profile, device=plan["device"]
                ).transcribe(
                    audio,
                    onset_threshold=profile.onset_threshold,
                    min_distance_frames=profile.peak_min_distance_frames,
                    fret_thresholds_per_bit=profile.fret_thresholds,
                )
            )
            chart = GuitarChart(tempo_bpm=120.0, instrument="bass")
            for event in events:
                if len(event.frets) >= 2:
                    chart.chords.append(
                        GuitarChord(
                            time_ms=event.time_sec * 1000.0,
                            frets=list(event.frets),
                            duration_ms=profile.note_duration_ms,
                        )
                    )
                elif event.frets:
                    chart.notes.append(
                        GuitarNote(
                            time_ms=event.time_sec * 1000.0,
                            fret=event.frets[0],
                            duration_ms=profile.note_duration_ms,
                        )
                    )
            midi_path = output_dir / "notes.mid"
            _write_expert_bass_midi(chart, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {
                "bass_neural": {
                    "status": "succeeded",
                    "expert_event_count": len(events),
                    "evaluation_sha256": profile.evaluation_sha256,
                }
            }
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument="bass",
                stage_name="expert_chart",
                artifact_ids=("notes_midi",),
            )
            response = {"output_name": midi_path.name, "expert_event_count": len(events)}
        elif plan["capability"] == "keys.neural-v1-expert/v1":
            audio_path = Path(request["audio_path"])
            if not audio_path.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from scripts.preprocess_guitar_windows import load_audio_mono_22050  # noqa: PLC0415
            from src.inference.guitar_bass import (  # noqa: PLC0415
                GuitarChart,
                GuitarChord,
                GuitarNote,
            )
            from src.inference.keys_neural_profile import (  # noqa: PLC0415
                KeysNeuralCharter,
                load_keys_neural_expert_profile,
            )

            audio = load_audio_mono_22050(audio_path)
            if audio is None:
                raise WorkerRequestError("chart input audio is unreadable")
            profile = load_keys_neural_expert_profile(bundle, profile_id)
            events = _run_without_legacy_output(
                lambda: KeysNeuralCharter.from_bundle_profile(
                    bundle, profile, device=plan["device"]
                ).transcribe(
                    audio,
                    onset_threshold=profile.onset_threshold,
                    min_distance_frames=profile.peak_min_distance_frames,
                    fret_thresholds_per_bit=profile.fret_thresholds,
                )
            )
            chart = GuitarChart(tempo_bpm=120.0, instrument="keys")
            for event in events:
                if len(event.frets) >= 2:
                    chart.chords.append(
                        GuitarChord(
                            time_ms=event.time_sec * 1000.0,
                            frets=list(event.frets),
                            duration_ms=profile.note_duration_ms,
                        )
                    )
                elif event.frets:
                    chart.notes.append(
                        GuitarNote(
                            time_ms=event.time_sec * 1000.0,
                            fret=event.frets[0],
                            duration_ms=profile.note_duration_ms,
                        )
                    )
            midi_path = output_dir / "notes.mid"
            _write_expert_keys_midi(chart, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {
                "keys_neural": {
                    "status": "succeeded",
                    "expert_event_count": len(events),
                    "evaluation_sha256": profile.evaluation_sha256,
                }
            }
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument="keys",
                stage_name="expert_chart",
                artifact_ids=("notes_midi",),
            )
            response = {"output_name": midi_path.name, "expert_event_count": len(events)}
        elif plan["capability"] == "drums.v14-expert/v1":
            audio = Path(request["audio_path"])
            if not audio.is_file():
                raise WorkerRequestError("chart input audio is unavailable")
            from src.inference.drums_v14_profile import (
                load_drums_v14_expert_profile,  # noqa: PLC0415
            )
            from src.inference.drums_v14_runtime import DrumsV14Runtime  # noqa: PLC0415

            profile = load_drums_v14_expert_profile(bundle, profile_id)
            component = bundle.component(profile.component_id)
            if component is None or component.checkpoint is None:
                raise WorkerRequestError("drums V14 component is incomplete")
            events = _run_without_legacy_output(
                lambda: DrumsV14Runtime.from_profile(
                    profile,
                    checkpoint_path=component.checkpoint,
                    model_parameters=profile.model_parameters,
                    device=plan["device"],
                ).transcribe_audio_file(audio)
            )
            midi_path = output_dir / "notes.mid"
            _write_expert_drums_midi(events, midi_path)
            artifacts = {"notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)}}
            stages = {"drums": {"status": "succeeded", "expert_event_count": len(events)}}
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument="drums",
                stage_name="expert_chart",
                artifact_ids=("notes_midi",),
            )
            response = {"output_name": midi_path.name, "expert_event_count": len(events)}
        elif plan["capability"] == "difficulty.transform/v1":
            import torch  # noqa: PLC0415

            from scripts.infer_chart_transform import parse_source_events, predict  # noqa: PLC0415
            from scripts.prepare_guitar_chart_pairs import (  # noqa: PLC0415
                parse_instrument_difficulties,
            )

            component_id = plan["components"][0]["id"]
            component = bundle.component(component_id)
            if component is None or component.checkpoint is None or component.config is None:
                raise WorkerRequestError("difficulty transform component is incomplete")
            config = json.loads(component.config.read_text(encoding="utf-8"))
            instrument = config.get("instrument") if isinstance(config, dict) else None
            target_difficulty = (
                config.get("target_difficulty") if isinstance(config, dict) else None
            )
            if not isinstance(instrument, str) or instrument not in plan["instruments"]:
                raise WorkerRequestError("difficulty transform component instrument is invalid")
            checkpoint = torch.load(component.checkpoint, map_location="cpu", weights_only=True)
            lane_count = checkpoint.get("lane_count") if isinstance(checkpoint, dict) else None
            if not isinstance(lane_count, int) or lane_count != 5:
                raise WorkerRequestError("difficulty transform checkpoint has invalid lane count")
            source_midi = Path(request["source_midi_path"])
            if not source_midi.is_file():
                raise WorkerRequestError("difficulty transform source MIDI is unavailable")
            source_events = parse_source_events(
                parse_instrument_difficulties(source_midi, instrument)["Expert"], lane_count
            )
            song_path = Path(request["song_path"]) if request["song_path"] else None
            if song_path is not None and not song_path.is_file():
                raise WorkerRequestError("difficulty transform song input is unavailable")
            from src.chart_transform_profile import (  # noqa: PLC0415
                promoted_chart_transform_thresholds,
            )

            events = _run_without_legacy_output(
                lambda: predict(
                    component.checkpoint,
                    source_events,
                    song_path=song_path,
                    device_name=plan["device"],
                    threshold=promoted_chart_transform_thresholds(bundle, plan["profile_id"]),
                )
            )
            events_path = output_dir / "events.json"
            events_path.parent.mkdir(parents=True, exist_ok=True)
            events_path.write_text(
                json.dumps(
                    {"instrument": instrument, "difficulty": target_difficulty, "events": events},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            midi_path = output_dir / "notes.mid"
            _write_five_lane_midi(
                events, instrument=instrument, difficulty=target_difficulty, output_path=midi_path
            )
            artifacts = {
                "events": {"name": events_path.name, "sha256": _sha256(events_path)},
                "notes_midi": {"name": midi_path.name, "sha256": _sha256(midi_path)},
            }
            stages = {"difficulty_transform": {"status": "succeeded", "event_count": len(events)}}
            _complete_chart_stage(
                instrument_results,
                difficulty,
                instrument=instrument,
                stage_name="difficulty_transform",
                artifact_ids=("events", "notes_midi"),
            )
            response = {
                "output_name": midi_path.name,
                "event_count": len(events),
                "difficulty": target_difficulty,
            }
        else:
            raise WorkerRequestError("profile has no worker chart execution handler")
        run_manifest = {
            "schema_version": 1,
            "format": CHART_RUN_FORMAT,
            "status": "completed",
            "model_id": plan["model_id"],
            "profile_id": plan["profile_id"],
            "capability": plan["capability"],
            "difficulty_policy": plan["difficulty_policy"],
            "manifest_sha256": manifest_sha256,
            "components": plan["components"],
            "profile_configuration_sha256": plan["profile_configuration_sha256"],
            "profile_configuration_byte_length": plan["profile_configuration_byte_length"],
            "artifacts": artifacts,
            "instrument_results": instrument_results,
            "difficulty": difficulty,
            "stages": stages,
        }
        (output_dir / "run.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except WorkerRequestError:
        raise
    except (BundleValidationError, OSError, RuntimeError, ValueError) as error:
        raise WorkerRequestError("profile chart execution failed") from error
    return {
        "schema_version": 1,
        "format": CHART_RUN_FORMAT,
        "status": "completed",
        "profile_id": plan["profile_id"],
        "manifest_sha256": manifest_sha256,
        "run_manifest_name": "run.json",
        "instrument_results": instrument_results,
        "chart_result": {
            "instrument_results": instrument_results,
            "difficulty": difficulty,
        },
        **response,
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


def _run_event_stream(request_path: Path, stage: str, callback: Any) -> int:
    """Emit safe NDJSON lifecycle events for an OCTAVE-supervised sync job."""
    try:
        job_id = f"strum-{hashlib.sha256(request_path.read_bytes()).hexdigest()[:16]}"
    except OSError:
        job_id = "strum-unreadable-request"
    _print_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "sequence": 1,
            "stage": stage,
            "progress": 0.0,
            "state": "running",
            "code": "started",
        }
    )
    try:
        result = callback()
    except BundleValidationError:
        code, message = "model_bundle_invalid", "model bundle failed validation"
    except CatalogValidationError:
        code, message = "catalog_invalid", "catalog failed validation"
    except WorkerRequestError:
        code, message = "request_invalid", "worker request is invalid"
    else:
        _print_json(
            {
                "protocol_version": PROTOCOL_VERSION,
                "job_id": job_id,
                "sequence": 2,
                "stage": stage,
                "progress": 1.0,
                "state": "succeeded",
                "code": "completed",
                "result": result,
            }
        )
        return 0
    _print_json(
        {
            "protocol_version": PROTOCOL_VERSION,
            "job_id": job_id,
            "sequence": 2,
            "stage": stage,
            "progress": 1.0,
            "state": "failed",
            "code": code,
            "message": message,
        }
    )
    return 2


def _pipeline_by_id(pipeline_id: str) -> PipelineDescriptor:
    for pipeline in PIPELINES:
        if pipeline.id == pipeline_id:
            return pipeline
    raise WorkerRequestError("unknown pipeline_id")


def _bounded_asset_bytes(assets: list[CatalogAsset]) -> tuple[int, bool]:
    """Return a bounded, de-duplicated catalog-input estimate.

    Catalog assets are content addressed, so the same source input can be
    shared across many catalog records.  Counting it once reports the maximum
    additional input footprint a task view will ask STRUM to read, without
    implying a total disk estimate for trainer-created artifacts.
    """
    distinct = {asset.sha256: asset.byte_length for asset in assets}
    total = sum(distinct.values())
    return min(total, MAX_ESTIMATED_STORAGE_BYTES), total > MAX_ESTIMATED_STORAGE_BYTES


def _empty_exclusions(*codes: str) -> dict[str, int]:
    """Keep reason-code output stable without emitting catalog record data."""
    return dict.fromkeys(codes, 0)


def _audio_task_inspection(
    catalog: SongSourceCatalog,
    *,
    instrument: str,
    preferred_role: str,
    fallback_role: str | None,
    required_difficulty: str,
    require_lead_vocal_compatibility: bool = False,
    runtime_admission_label_track: str | None = None,
    profile_grade: bool = False,
    profile_splitter: Callable[[str], str] | None = None,
) -> dict[str, object]:
    """Inspect the same selection and compatibility gates as task preparation."""
    exclusion_codes = [
        "training_use_not_allowed",
        "instrument_not_present",
        "required_difficulty_missing",
        "dedicated_audio_unavailable" if profile_grade else "audio_unavailable",
    ]
    exclusions = _empty_exclusions(*exclusion_codes)
    if profile_grade:
        if profile_splitter is None:
            raise WorkerRequestError("profile-grade inspection split assignment is unavailable")
        try:
            validate_profile_grade_audio_selection(
                instrument=instrument,
                audio_role=preferred_role,
                fallback_audio_role=fallback_role,
            )
        except ValueError as error:
            raise WorkerRequestError(str(error)) from error
    if require_lead_vocal_compatibility:
        exclusions.update(
            _empty_exclusions(
                "vocal_target_incompatible",
                "vocal_audio_incompatible",
            )
        )
    if runtime_admission_label_track is not None:
        exclusions.update(
            _empty_exclusions("runtime_audio_unreadable", "exact_expert_label_missing")
        )
    if require_lead_vocal_compatibility:
        assets: list[CatalogAsset] = []
        eligible_count = 0
        for record in catalog.records:
            if record.training_use != TRAINING_ALLOWED:
                exclusions["training_use_not_allowed"] += 1
                continue
            coverage = record.instruments.get(instrument)
            if coverage is None or coverage.status != "present":
                exclusions["instrument_not_present"] += 1
                continue
            if required_difficulty not in coverage.difficulties:
                exclusions["required_difficulty_missing"] += 1
                continue
            if not has_exact_lead_vocal_label_source(record):
                exclusions["vocal_target_incompatible"] += 1
                continue
            role = select_compatible_vocal_audio_role(record, preferred_role, fallback_role)
            if role is None:
                # This aggregates a missing declared role and a stream that
                # cannot complete the exact soundfile decode used by every
                # lead-Vocal preprocessor.  Both are excluded by preparation
                # at the same boundary, and neither reveals a record identity.
                exclusions["vocal_audio_incompatible"] += 1
                continue
            assets.extend((record.notes_midi, record.audio[role]))
            eligible_count += 1

        estimated_storage_bytes, storage_estimate_capped = _bounded_asset_bytes(assets)
        return {
            "eligible_count": eligible_count,
            "exclusion_reason_counts": exclusions,
            "audio_policy": {
                "kind": "preferred_with_fallback",
                "preferred_role": preferred_role,
                "fallback_role": fallback_role,
                "required": True,
                "compatibility": "soundfile-full-stream-decode/v1",
            },
            "estimated_storage_bytes": estimated_storage_bytes,
            "storage_estimate_capped": storage_estimate_capped,
            "storage_estimate_semantics": CATALOG_STORAGE_ESTIMATE_SEMANTICS,
        }

    selected_roles: dict[str, str] = {}
    for role in (preferred_role, fallback_role):
        if role is None:
            continue
        for source in select_training_sources(
            catalog,
            instrument,
            required_difficulties=(required_difficulty,),
            audio_role=role,
        ):
            selected_roles.setdefault(source.source_id, role)

    assets: list[CatalogAsset] = []
    eligible_count = 0
    profile_source_ids: list[str] = []
    for record in catalog.records:
        if record.training_use != TRAINING_ALLOWED:
            exclusions["training_use_not_allowed"] += 1
            continue
        coverage = record.instruments.get(instrument)
        if coverage is None or coverage.status != "present":
            exclusions["instrument_not_present"] += 1
            continue
        if required_difficulty not in coverage.difficulties:
            exclusions["required_difficulty_missing"] += 1
            continue
        role = selected_roles.get(record.source_id)
        if role is None:
            exclusions["dedicated_audio_unavailable" if profile_grade else "audio_unavailable"] += 1
            continue
        if runtime_admission_label_track is not None:
            reason = classify_five_lane_runtime_source(
                record.audio[role].path,
                record.notes_midi.path,
                label_track=runtime_admission_label_track,
            )
            if reason is not None:
                exclusions[reason] += 1
                continue
        assets.extend((record.notes_midi, record.audio[role]))
        eligible_count += 1
        if profile_grade:
            profile_source_ids.append(record.source_id)

    estimated_storage_bytes, storage_estimate_capped = _bounded_asset_bytes(assets)
    result: dict[str, object] = {
        "eligible_count": eligible_count,
        "exclusion_reason_counts": exclusions,
        "audio_policy": {
            "kind": "exact_dedicated_role" if profile_grade else "preferred_with_fallback",
            "preferred_role": preferred_role,
            "fallback_role": fallback_role,
            "required": True,
        },
        "estimated_storage_bytes": estimated_storage_bytes,
        "storage_estimate_capped": storage_estimate_capped,
        "storage_estimate_semantics": CATALOG_STORAGE_ESTIMATE_SEMANTICS,
    }
    if profile_grade:
        assert profile_splitter is not None
        by_split = {
            split: sum(profile_splitter(source_id) == split for source_id in profile_source_ids)
            for split in ("train", "val", "test")
        }
        result["profile_grade_admission"] = {
            "format": PROFILE_GRADE_ADMISSION_FORMAT,
            "audio_selection": PROFILE_GRADE_ADMISSION["audio_selection"],
            "source_disjoint_minimums": dict(PROFILE_GRADE_MINIMUM_SOURCES_BY_SPLIT),
            "by_split": by_split,
            "meets_minimums": profile_grade_admission_is_met(by_split),
        }
    return result


def _chart_transform_inspection(
    catalog: SongSourceCatalog,
    *,
    instrument: str | None,
    target_difficulty: str | None,
    audio_feature_mode: str = "none",
    audio_role: str | None = None,
    fallback_audio_role: str | None = None,
) -> dict[str, object]:
    """Inspect chart-pair eligibility without parsing or exposing MIDI inputs.

    If an OCTAVE caller has not selected the required chart-transform options
    yet, the count means records that can support *at least one* declared
    instrument/target transform.  Once options are supplied it is exactly the
    selection `dataset prepare` will consider before MIDI parsing.
    """
    supported_instruments = (
        (instrument,) if instrument is not None else ("guitar", "bass", "keys", "drums")
    )
    targets = (
        (target_difficulty.lower(),)
        if target_difficulty is not None
        else ("hard", "medium", "easy")
    )
    exclusions = _empty_exclusions(
        "training_use_not_allowed",
        "instrument_not_present",
        "source_difficulty_missing",
        "target_difficulty_missing",
        "audio_unavailable",
    )
    assets: list[CatalogAsset] = []
    eligible_count = 0
    for record in catalog.records:
        if record.training_use != TRAINING_ALLOWED:
            exclusions["training_use_not_allowed"] += 1
            continue
        coverages = [
            coverage
            for candidate in supported_instruments
            if (coverage := record.instruments.get(candidate)) is not None
            and coverage.status == "present"
        ]
        if not coverages:
            exclusions["instrument_not_present"] += 1
            continue
        expert_coverages = [coverage for coverage in coverages if "expert" in coverage.difficulties]
        if not expert_coverages:
            exclusions["source_difficulty_missing"] += 1
            continue
        if not any(
            any(target in coverage.difficulties for target in targets)
            for coverage in expert_coverages
        ):
            exclusions["target_difficulty_missing"] += 1
            continue
        if audio_feature_mode != "none":
            preferred = audio_role or (instrument if instrument is not None else "mix")
            fallback = "mix" if fallback_audio_role is None else fallback_audio_role
            asset = record.audio.get(preferred) or record.audio.get(fallback)
            if asset is None:
                exclusions["audio_unavailable"] += 1
                continue
            assets.append(asset)
        eligible_count += 1
        assets.append(record.notes_midi)

    estimated_storage_bytes, storage_estimate_capped = _bounded_asset_bytes(assets)
    return {
        "eligible_count": eligible_count,
        "exclusion_reason_counts": exclusions,
        "audio_policy": (
            {
                "kind": "task_view_audio_conditioning",
                "preferred_role": audio_role or instrument,
                "fallback_role": "mix" if fallback_audio_role is None else fallback_audio_role,
                "required": True,
                "duration_validation": "performed during dataset prepare",
            }
            if audio_feature_mode != "none"
            else {"kind": "not_required", "required": False}
        ),
        "estimated_storage_bytes": estimated_storage_bytes,
        "storage_estimate_capped": storage_estimate_capped,
        "storage_estimate_semantics": CATALOG_STORAGE_ESTIMATE_SEMANTICS,
        "eligibility_selection": {
            "mode": "requested_prepare_options"
            if instrument is not None
            else "any_declared_chart_transform_option",
            "instrument": instrument,
            "target_difficulty": target_difficulty,
        },
    }


def _read_catalog_inspection_options(raw: str | None) -> dict[str, Any]:
    """Parse an optional, local-only subset of preparation selection options."""
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise WorkerRequestError("catalog inspection options are not valid JSON") from error
    if not isinstance(value, dict):
        raise WorkerRequestError("catalog inspection options must be an object")
    return value


def _inspect_pipeline_catalog(
    catalog: SongSourceCatalog,
    descriptor: PipelineDescriptor,
    options: dict[str, Any],
) -> dict[str, object]:
    """Produce one uniform, path-free planning summary for a pipeline."""
    pipeline_id = descriptor.id
    if pipeline_id == "vocals.harmony-source-policy/v1":
        permitted = {"harmony_tracks"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Vocal harmony source-policy inspection option")
        tracks = options.get("harmony_tracks")
        if tracks is not None and (
            not isinstance(tracks, list) or not all(isinstance(track, str) for track in tracks)
        ):
            raise WorkerRequestError("vocal harmony harmony_tracks must be an array of strings")
        result = inspect_vocal_harmony_source_catalog(catalog.root, harmony_tracks=tracks)
        result["storage_estimate_semantics"] = CATALOG_STORAGE_ESTIMATE_SEMANTICS
        return result
    if pipeline_id == "guitar.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Guitar catalog inspection option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Guitar profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Guitar profile_grade does not permit a fallback audio role")
        return _audio_task_inspection(
            catalog,
            instrument="guitar",
            preferred_role=options.get("audio_role", "guitar"),
            fallback_role=(
                options.get("fallback_audio_role")
                if profile_grade
                else options.get("fallback_audio_role", "mix")
            ),
            required_difficulty=options.get("required_difficulty", "expert"),
            runtime_admission_label_track="PART GUITAR",
            profile_grade=profile_grade,
            profile_splitter=guitar_deterministic_split,
        )
    if pipeline_id == "bass.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Bass catalog inspection option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Bass profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Bass profile_grade does not permit a fallback audio role")
        return _audio_task_inspection(
            catalog,
            instrument="bass",
            preferred_role=options.get("audio_role", "bass"),
            fallback_role=(
                options.get("fallback_audio_role")
                if profile_grade
                else options.get("fallback_audio_role", "mix")
            ),
            required_difficulty=options.get("required_difficulty", "expert"),
            runtime_admission_label_track="PART BASS",
            profile_grade=profile_grade,
            profile_splitter=lambda source_id: catalog_task_deterministic_split(
                source_id, seed="catalog-source-id/v1"
            ),
        )
    if pipeline_id == "keys.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Keys catalog inspection option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Keys profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Keys profile_grade does not permit a fallback audio role")
        return _audio_task_inspection(
            catalog,
            instrument="keys",
            preferred_role=options.get("audio_role", "keys"),
            fallback_role=(
                options.get("fallback_audio_role")
                if profile_grade
                else options.get("fallback_audio_role", "mix")
            ),
            required_difficulty=options.get("required_difficulty", "expert"),
            runtime_admission_label_track="PART KEYS",
            profile_grade=profile_grade,
            profile_splitter=lambda source_id: catalog_task_deterministic_split(
                source_id, seed="catalog-source-id/v1"
            ),
        )
    if pipeline_id in {
        "vocals.note-activity/v1",
        "vocals.phrase-boundaries/v1",
        "vocals.lyric-alignment/v1",
        "vocals.talky-activity/v1",
    }:
        permitted = {
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
            "split_ratios",
            "split_seed",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Vocal catalog inspection option")
        return _audio_task_inspection(
            catalog,
            instrument="vocals",
            preferred_role=options.get("audio_role", "vocals"),
            fallback_role=options.get("fallback_audio_role", "mix"),
            required_difficulty=options.get("required_difficulty", "expert"),
            require_lead_vocal_compatibility=True,
        )
    if pipeline_id == "drums.onset-classifier/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Drums catalog inspection option")
        return _audio_task_inspection(
            catalog,
            instrument="drums",
            preferred_role=options.get("audio_role", "drums"),
            fallback_role=options.get("fallback_audio_role", "mix"),
            required_difficulty=options.get("required_difficulty", "expert"),
        )
    if pipeline_id == "chart_transform.five_lane/v1":
        permitted = {
            "instrument",
            "target_difficulty",
            "audio_feature_mode",
            "audio_role",
            "fallback_audio_role",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported chart-transform catalog inspection option")
        instrument = options.get("instrument")
        target_difficulty = options.get("target_difficulty")
        audio_feature_mode = options.get("audio_feature_mode", "none")
        audio_role = options.get("audio_role")
        fallback_audio_role = options.get("fallback_audio_role")
        if audio_feature_mode not in {"none", "rms_onset_v1"}:
            raise WorkerRequestError("chart-transform audio_feature_mode is unsupported")
        if audio_feature_mode == "none" and (
            audio_role is not None or fallback_audio_role is not None
        ):
            raise WorkerRequestError("chart-transform audio roles require audio_feature_mode")
        if audio_feature_mode != "none" and (
            audio_role is not None
            and audio_role not in AUDIO_ROLES
            or fallback_audio_role is not None
            and fallback_audio_role not in AUDIO_ROLES
        ):
            raise WorkerRequestError("chart-transform audio role is unsupported")
        if instrument is None and target_difficulty is None:
            return _chart_transform_inspection(
                catalog,
                instrument=None,
                target_difficulty=None,
                audio_feature_mode=audio_feature_mode,
                audio_role=audio_role,
                fallback_audio_role=fallback_audio_role,
            )
        if instrument not in {"guitar", "bass", "keys", "drums"} or target_difficulty not in {
            "Hard",
            "Medium",
            "Easy",
        }:
            raise WorkerRequestError(
                "chart-transform catalog inspection requires a supported instrument and target_difficulty"
            )
        return _chart_transform_inspection(
            catalog,
            instrument=instrument,
            target_difficulty=target_difficulty,
            audio_feature_mode=audio_feature_mode,
            audio_role=audio_role,
            fallback_audio_role=fallback_audio_role,
        )

    task_kind = next(
        (kind for kind, value in CATALOG_TASK_PIPELINES.items() if value == pipeline_id), None
    )
    if task_kind is None:
        raise WorkerRequestError("pipeline has no catalog task adapter")
    permitted = {"audio_role", "fallback_audio_role", "disable_fallback", "required_difficulty"}
    if set(options) - permitted:
        raise WorkerRequestError("unsupported catalog task inspection option")
    default_preferred, default_fallback = CATALOG_TASK_DEFAULT_AUDIO_ROLES[task_kind]
    disable_fallback = options.get("disable_fallback", False)
    if not isinstance(disable_fallback, bool):
        raise WorkerRequestError("catalog task disable_fallback must be a boolean")
    requested_fallback = options.get("fallback_audio_role")
    return _audio_task_inspection(
        catalog,
        instrument=CATALOG_TASK_INSTRUMENTS[task_kind],
        preferred_role=options.get("audio_role", default_preferred),
        fallback_role=None
        if disable_fallback
        else (default_fallback if requested_fallback is None else requested_fallback),
        required_difficulty=options.get("required_difficulty", "expert"),
        require_lead_vocal_compatibility=task_kind == "vocals",
    )


def inspect_catalog(
    catalog_root: str | Path,
    pipeline_id: str | None = None,
    *,
    options: dict[str, Any] | None = None,
) -> dict[str, object]:
    """Validate a catalog and return a path-free, pipeline-aware planning summary.

    The optional options are the subset of a pipeline's preparation controls
    that changes eligibility.  STRUM intentionally returns neither the
    options nor source identifiers, records, paths, rights text, or provenance.
    """
    if pipeline_id is not None:
        descriptor = _pipeline_by_id(pipeline_id)
    else:
        descriptor = None
    catalog = load_catalog(catalog_root)
    allowed = sum(record.training_use == TRAINING_ALLOWED for record in catalog.records)
    response: dict[str, object] = {
        "status": "ready",
        "catalog_id": catalog.catalog_id,
        "record_count": len(catalog.records),
        "allowed_record_count": allowed,
        "pipeline_id": pipeline_id,
    }
    if descriptor is not None:
        response.update(_inspect_pipeline_catalog(catalog, descriptor, options or {}))
    return response


def _read_prepare_request(request_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("request is unreadable or not valid JSON") from error
    if not isinstance(raw, dict) or set(raw) != {
        "catalog_root",
        "pipeline_id",
        "output",
        "options",
    }:
        raise WorkerRequestError(
            "request must contain catalog_root, pipeline_id, output, and options"
        )
    if not all(
        isinstance(raw[key], str) and raw[key] for key in ("catalog_root", "pipeline_id", "output")
    ):
        raise WorkerRequestError("request locations and pipeline_id must be non-empty strings")
    if not isinstance(raw["options"], dict):
        raise WorkerRequestError("request options must be an object")
    return raw


def _task_view_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def prepare_dataset_request(request_path: Path) -> dict[str, object]:
    """Materialize one catalog-only task view from a strict host request.

    Paths are accepted only as worker-local configuration. The response exposes
    the known output basename and stable identity, never original package paths
    or a catalog location.
    """
    request = _read_prepare_request(request_path)
    pipeline_id = request["pipeline_id"]
    descriptor = _pipeline_by_id(pipeline_id)
    if descriptor.preparation_status != "available":
        raise WorkerRequestError("pipeline does not support dataset preparation")
    catalog_root, output, options = (
        request["catalog_root"],
        Path(request["output"]),
        request["options"],
    )
    if pipeline_id == "guitar.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Guitar preparation option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Guitar profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Guitar profile_grade does not permit a fallback audio role")
        manifest_options = dict(options)
        manifest_options.pop("profile_grade", None)
        if profile_grade:
            manifest_options.setdefault("audio_role", "guitar")
            manifest_options.setdefault("fallback_audio_role", None)
        manifest = build_guitar_manifest(
            catalog_root,
            runtime_admission=True,
            profile_grade=profile_grade,
            **manifest_options,
        )
        written = write_guitar_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id == "vocals.harmony-source-policy/v1":
        permitted = {"harmony_tracks", "split_ratios", "split_seed"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Vocal harmony source-policy preparation option")
        tracks = options.get("harmony_tracks")
        ratios = options.get("split_ratios", (80, 10, 10))
        seed = options.get("split_seed", "catalog-source-id/v1")
        if (
            not isinstance(ratios, list | tuple)
            or len(ratios) != 3
            or not all(isinstance(value, int) for value in ratios)
        ):
            raise WorkerRequestError("vocal harmony split_ratios must be three integers")
        if not isinstance(seed, str):
            raise WorkerRequestError("vocal harmony split_seed must be a string")
        try:
            manifest = build_vocal_harmony_source_task(
                catalog_root,
                harmony_tracks=tracks,
                split_ratios=tuple(ratios),
                split_seed=seed,
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError("vocal harmony source-policy preparation failed") from error
        written = write_vocal_harmony_source_task(output, manifest)
        summary = manifest["summary"]
        assert isinstance(summary, dict)
        record_count = summary["record_count"]
        assert isinstance(record_count, int)
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id == "bass.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Bass preparation option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Bass profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Bass profile_grade does not permit a fallback audio role")
        manifest_options = dict(options)
        manifest_options.pop("profile_grade", None)
        if profile_grade:
            manifest_options.setdefault("audio_role", "bass")
            manifest_options["disable_fallback"] = True
        manifest = build_catalog_task_manifest(
            catalog_root,
            "bass_onset_fret",
            runtime_admission=True,
            profile_grade=profile_grade,
            **manifest_options,
        )
        written = write_catalog_task_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id == "keys.onset-fret/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty", "profile_grade"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Keys preparation option")
        profile_grade = options.get("profile_grade", False)
        if not isinstance(profile_grade, bool):
            raise WorkerRequestError("Keys profile_grade must be a boolean")
        if profile_grade and options.get("fallback_audio_role") is not None:
            raise WorkerRequestError("Keys profile_grade does not permit a fallback audio role")
        manifest_options = dict(options)
        manifest_options.pop("profile_grade", None)
        if profile_grade:
            manifest_options.setdefault("audio_role", "keys")
            manifest_options["disable_fallback"] = True
        manifest = build_catalog_task_manifest(
            catalog_root,
            "keys_onset_fret",
            runtime_admission=True,
            profile_grade=profile_grade,
            **manifest_options,
        )
        written = write_catalog_task_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id in {
        "vocals.note-activity/v1",
        "vocals.phrase-boundaries/v1",
        "vocals.lyric-alignment/v1",
        "vocals.talky-activity/v1",
    }:
        permitted = {
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
            "split_ratios",
            "split_seed",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Vocal preparation option")
        task_kind = (
            "vocals_activity"
            if pipeline_id == "vocals.note-activity/v1"
            else (
                "vocals_phrase_boundaries"
                if pipeline_id == "vocals.phrase-boundaries/v1"
                else (
                    "vocals_lyric_alignment"
                    if pipeline_id == "vocals.lyric-alignment/v1"
                    else "vocals_talky_activity"
                )
            )
        )
        manifest = build_catalog_task_manifest(catalog_root, task_kind, **options)
        written = write_catalog_task_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    elif pipeline_id == "drums.onset-classifier/v1":
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Drums preparation option")
        manifest = build_drums_manifest(catalog_root, **options)
        written = write_drums_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = manifest.get("task_view_sha256") or _task_view_digest(manifest)
    elif pipeline_id == "chart_transform.five_lane/v1":
        permitted = {
            "instrument",
            "target_difficulty",
            "split_seed",
            "calibration_fraction",
            "test_fraction",
            "dataset_id",
            "overwrite",
            "audio_feature_mode",
            "audio_role",
            "fallback_audio_role",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported chart-transform preparation option")
        try:
            result = prepare_catalog_chart_pairs(
                catalog_root,
                output,
                CatalogChartPairOptions(
                    **{key: value for key, value in options.items() if key != "overwrite"}
                ),
                overwrite=bool(options.get("overwrite", False)),
            )
        except (TypeError, ValueError) as error:
            raise WorkerRequestError("invalid chart-transform preparation options") from error
        written = result["manifest_path"]
        record_count = result["record_count"]
        task_view_id = result["task_view_id"]
    elif pipeline_id in {
        "strum.instrument-chart/pro-guitar/v1",
        "strum.instrument-chart/pro-bass/v1",
        "strum.instrument-chart/pro-keys/v1",
    }:
        task_kind = next(
            (kind for kind, value in CATALOG_TASK_PIPELINES.items() if value == pipeline_id), None
        )
        if task_kind is None:
            raise WorkerRequestError("pipeline has no Pro catalog task adapter")
        permitted = {
            "audio_role",
            "fallback_audio_role",
            "disable_fallback",
            "required_difficulty",
            "split_ratios",
            "split_seed",
            "preprocessing",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Pro catalog preparation option")
        manifest = build_catalog_pro_target_manifest(catalog_root, task_kind, **options)
        written = write_catalog_pro_target_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    else:
        task_kind = next(
            (kind for kind, value in CATALOG_TASK_PIPELINES.items() if value == pipeline_id), None
        )
        if task_kind is None:
            raise WorkerRequestError("pipeline has no catalog task adapter")
        permitted = {
            "audio_role",
            "fallback_audio_role",
            "disable_fallback",
            "required_difficulty",
            "split_ratios",
            "split_seed",
            "preprocessing",
        }
        if set(options) - permitted:
            raise WorkerRequestError("unsupported catalog task preparation option")
        manifest = build_catalog_task_manifest(catalog_root, task_kind, **options)
        written = write_catalog_task_manifest(output, manifest)
        record_count = manifest["summary"]["record_count"]
        task_view_id = _task_view_digest(manifest)
    return {
        "status": "prepared",
        "pipeline_id": pipeline_id,
        "task_view_id": task_view_id,
        "record_count": record_count,
        "output_name": written.name,
    }


def _read_train_request(request_path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("request is unreadable or not valid JSON") from error
    base_fields = {"pipeline_id", "task_view", "output", "options"}
    permitted = base_fields | {"catalog_root", "parent_bundle"}
    if not isinstance(raw, dict) or set(raw) - permitted or not base_fields <= set(raw):
        raise WorkerRequestError("training request has unsupported fields")
    if not all(
        isinstance(raw[key], str) and raw[key] for key in ("pipeline_id", "task_view", "output")
    ):
        raise WorkerRequestError(
            "training request locations and pipeline_id must be non-empty strings"
        )
    if not isinstance(raw["options"], dict):
        raise WorkerRequestError("training options must be an object")
    if raw["pipeline_id"] in {
        "guitar.onset-fret/v1",
        "bass.onset-fret/v1",
        "keys.onset-fret/v1",
        "vocals.note-activity/v1",
        "vocals.phrase-boundaries/v1",
        "vocals.lyric-alignment/v1",
        "vocals.talky-activity/v1",
        "drums.onset-classifier/v1",
        "strum.fret-mapper/guitar/v1",
        "strum.fret-mapper/bass/v1",
        "strum.section-classifier/guitar/v1",
        "strum.section-classifier/bass/v1",
        "strum.instrument-chart/pro-guitar/v1",
        "strum.instrument-chart/pro-bass/v1",
        "strum.instrument-chart/pro-keys/v1",
    }:
        if set(raw) != base_fields | {"catalog_root"}:
            raise WorkerRequestError("catalog-backed training request has unsupported fields")
        if not isinstance(raw["catalog_root"], str) or not raw["catalog_root"]:
            raise WorkerRequestError("catalog-backed training requires worker-local catalog_root")
    elif raw["pipeline_id"] == "chart_transform.five_lane/v1":
        allowed = (
            base_fields,
            base_fields | {"catalog_root"},
            base_fields | {"parent_bundle"},
            base_fields | {"catalog_root", "parent_bundle"},
        )
        if set(raw) not in allowed:
            raise WorkerRequestError("chart-transform training request has unsupported fields")
        if "catalog_root" in raw and (
            not isinstance(raw["catalog_root"], str) or not raw["catalog_root"]
        ):
            raise WorkerRequestError("chart-transform catalog_root must be a non-empty string")
    elif set(raw) != base_fields:
        raise WorkerRequestError("training request has unsupported fields")
    if "parent_bundle" in raw and (
        not isinstance(raw["parent_bundle"], str) or not raw["parent_bundle"]
    ):
        raise WorkerRequestError("training parent_bundle must be a non-empty string")
    return raw


def _slug_component_part(value: str) -> str:
    """Return the portable component-name spelling used by chart transforms."""
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip(
        "_"
    )


def _chart_transform_component_id(*, instrument: str, source: str, target: str) -> str:
    return ".".join(
        (
            "chart_transform",
            _slug_component_part(instrument),
            f"{_slug_component_part(source)}_to_{_slug_component_part(target)}",
        )
    )


def _chart_transform_catalog_audio_manifest(
    *,
    dataset: dict[str, Any],
    catalog_root: str,
    output_dir: Path,
) -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, object]]:
    """Materialize worker-private audio inputs for an approved chart task.

    A chart-pair view is path-free by design.  Audio conditioning therefore
    reopens only the selected OCTAVE catalog in the worker, verifies its
    identity and every selected source asset, and creates a short-lived local
    manifest.  Neither the manifest nor copied/hard-linked audio survives the
    job; durable artifacts retain only source IDs, hashes, roles, and lineage.
    """
    task_view = dataset.get("task_view")
    if not isinstance(task_view, dict):
        raise WorkerRequestError("audio-conditioned chart training requires a catalog task view")
    lineage = task_view.get("catalog")
    source_inputs = task_view.get("source_inputs")
    audio_conditioning = task_view.get("audio_conditioning")
    instrument = dataset.get("instrument")
    target_difficulty = dataset.get("target_difficulty")
    if (
        not isinstance(lineage, dict)
        or not isinstance(source_inputs, list)
        or not isinstance(audio_conditioning, dict)
        or not isinstance(instrument, str)
        or not isinstance(target_difficulty, str)
    ):
        raise WorkerRequestError("chart-transform catalog task view is invalid")
    if (
        set(audio_conditioning) != {"mode", "preferred_role", "fallback_role"}
        or audio_conditioning.get("mode") != "rms_onset_v1"
        or not isinstance(audio_conditioning.get("preferred_role"), str)
        or audio_conditioning["preferred_role"] not in AUDIO_ROLES
        or (
            audio_conditioning.get("fallback_role") is not None
            and audio_conditioning["fallback_role"] not in AUDIO_ROLES
        )
    ):
        raise WorkerRequestError("chart-transform task view has invalid audio conditioning")
    preferred_role = audio_conditioning["preferred_role"]
    fallback_role = audio_conditioning["fallback_role"]

    catalog = load_catalog(catalog_root)
    records_path_value: object
    try:
        raw_catalog = json.loads((catalog.root / CATALOG_FILENAME).read_text(encoding="utf-8"))
        records_path_value = raw_catalog["records"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart-transform catalog lineage is unreadable") from error
    if (
        not isinstance(records_path_value, str)
        or Path(records_path_value).is_absolute()
        or catalog.catalog_id != lineage.get("catalog_id")
        or _sha256(catalog.root / CATALOG_FILENAME) != lineage.get("manifest_sha256")
        or _sha256((catalog.root / records_path_value).resolve()) != lineage.get("records_sha256")
    ):
        raise WorkerRequestError("chart-transform catalog lineage does not match its task view")

    records = {record.source_id: record for record in catalog.records}
    selected: list[tuple[str, CatalogAsset, str]] = []
    seen_source_ids: set[str] = set()
    expected_difficulties = {"expert", target_difficulty.lower()}
    for source in source_inputs:
        if (
            not isinstance(source, dict)
            or set(source)
            != {
                "source_id",
                "notes_midi_sha256",
                "audio_role",
                "audio_sha256",
                "audio_byte_length",
            }
            or not isinstance(source.get("source_id"), str)
            or not isinstance(source.get("notes_midi_sha256"), str)
            or not isinstance(source.get("audio_role"), str)
            or not isinstance(source.get("audio_sha256"), str)
            or not isinstance(source.get("audio_byte_length"), int)
            or source["audio_byte_length"] < 0
            or source["source_id"] in seen_source_ids
        ):
            raise WorkerRequestError("chart-transform task view has invalid source inputs")
        source_id = source["source_id"]
        record = records.get(source_id)
        coverage = record.instruments.get(instrument) if record else None
        if (
            record is None
            or record.training_use != TRAINING_ALLOWED
            or record.notes_midi.sha256 != source["notes_midi_sha256"]
            or coverage is None
            or coverage.status != "present"
            or not expected_difficulties <= coverage.difficulties
        ):
            raise WorkerRequestError("chart-transform task source no longer matches the catalog")
        role = source["audio_role"]
        asset = record.audio.get(role)
        if (
            asset is None
            or asset.sha256 != source["audio_sha256"]
            or asset.byte_length != source["audio_byte_length"]
        ):
            raise WorkerRequestError(
                "chart-transform conditioning audio no longer matches task view"
            )
        selected.append((source_id, asset, role))
        seen_source_ids.add(source_id)
    if not selected:
        raise WorkerRequestError("chart-transform task view has no conditioning audio sources")

    output_parent = output_dir.expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    private_dir = tempfile.TemporaryDirectory(prefix=".strum-chart-audio-", dir=output_parent)
    private_root = Path(private_dir.name)
    try:
        assets: list[dict[str, str]] = []
        provenance_assets: list[dict[str, object]] = []
        for index, (source_id, asset, role) in enumerate(selected):
            suffix = (
                asset.path.suffix
                if asset.path.suffix and len(asset.path.suffix) <= 16
                else ".audio"
            )
            local_name = f"audio-{index:04d}{suffix}"
            local_path = private_root / local_name
            try:
                os.link(asset.path, local_path)
            except OSError:
                shutil.copyfile(asset.path, local_path)
            assets.append({"song_id": source_id, "audio": local_name})
            provenance_assets.append(
                {
                    "source_id": source_id,
                    "audio_sha256": asset.sha256,
                    "audio_byte_length": asset.byte_length,
                    "audio_role": role,
                }
            )
        manifest_path = private_root / "audio-manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "format": "strum-local-audio-assets/v1",
                    "assets": assets,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception:
        private_dir.cleanup()
        raise
    return (
        private_dir,
        manifest_path,
        {
            "format": "strum-catalog-audio-conditioning/v1",
            "catalog_id": catalog.catalog_id,
            "catalog_manifest_sha256": lineage["manifest_sha256"],
            "catalog_records_sha256": lineage["records_sha256"],
            "preferred_role": preferred_role,
            "fallback_role": fallback_role,
            "assets": provenance_assets,
        },
    )


def _write_chart_transform_catalog_audio_provenance(
    output_dir: Path, provenance: dict[str, object]
) -> None:
    """Attach safe catalog-audio lineage after the private manifest is gone."""
    for name in ("training-metadata.json", "experiment.json"):
        path = output_dir / name
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise WorkerRequestError("chart-transform output metadata is unreadable") from error
        if not isinstance(value, dict):
            raise WorkerRequestError("chart-transform output metadata is invalid")
        if name == "training-metadata.json":
            conditioning = value.get("audio_conditioning")
            if not isinstance(conditioning, dict):
                raise WorkerRequestError("chart-transform audio metadata is invalid")
            conditioning["catalog"] = provenance
        else:
            value["catalog_audio_conditioning"] = provenance
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chart_transform_catalog_profile_call(
    *,
    catalog_root: str | Path,
    dataset_manifest: str | Path,
    candidate_root: str | Path,
    scratch_output: str | Path,
    call: Callable[[Path], dict[str, object]],
) -> dict[str, object]:
    """Run one audio-conditioned promotion operation with private catalog audio.

    Candidates deliberately retain the hash of their transient training audio
    manifest, never that manifest or any host asset path.  Promotion must
    therefore reconstruct the *same* manifest from the immutable task view
    and approved catalog, rather than asking OCTAVE to retain a private
    training scratch directory.  ``call`` only receives the temporary manifest
    path; it must return a portable result before this function removes it.
    """
    try:
        dataset = json.loads(Path(dataset_manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("chart-transform dataset manifest is unreadable") from error
    if not isinstance(dataset, dict):
        raise WorkerRequestError("chart-transform dataset manifest must be an object")

    # Check the candidate before reopening private catalog assets.  Besides
    # rejecting non-candidates, this guarantees catalog re-materialization is
    # used only for an audio-conditioned transform candidate.
    from src.chart_transform_profile import (  # noqa: PLC0415
        ChartTransformPromotionError,
        _candidate,
    )

    try:
        _bundle, _component_id, config = _candidate(candidate_root)
    except ChartTransformPromotionError as error:
        raise WorkerRequestError("chart-transform candidate failed verification") from error
    if config.get("audio_feature_mode") != "rms_onset_v1":
        raise WorkerRequestError(
            "catalog audio re-materialization requires an audio-conditioned transform candidate"
        )

    private_audio_dir: tempfile.TemporaryDirectory[str] | None = None
    try:
        try:
            private_audio_dir, audio_manifest, _provenance = (
                _chart_transform_catalog_audio_manifest(
                    dataset=dataset,
                    catalog_root=str(catalog_root),
                    output_dir=Path(scratch_output),
                )
            )
        except CatalogValidationError as error:
            raise WorkerRequestError("chart-transform catalog failed verification") from error
        return call(audio_manifest)
    finally:
        if private_audio_dir is not None:
            private_audio_dir.cleanup()


def evaluate_catalog_chart_transform_candidate(
    *,
    bundle_root: str | Path,
    dataset_manifest: str | Path,
    catalog_root: str | Path,
    output_path: str | Path,
    device: str = "cpu",
) -> dict[str, object]:
    """Evaluate an audio candidate from its task view and private catalog.

    This is intentionally a worker-local route.  It accepts catalog and
    candidate identities but never serializes an audio manifest or catalog
    location into its held-out report.
    """
    from src.chart_transform_profile import evaluate_chart_transform_candidate  # noqa: PLC0415

    return _chart_transform_catalog_profile_call(
        catalog_root=catalog_root,
        dataset_manifest=dataset_manifest,
        candidate_root=bundle_root,
        scratch_output=output_path,
        call=lambda audio_manifest: evaluate_chart_transform_candidate(
            bundle_root=bundle_root,
            dataset_manifest=dataset_manifest,
            output_path=output_path,
            device=device,
            audio_manifest=audio_manifest,
        ),
    )


def package_catalog_chart_transform_profile(
    *,
    experiment_dir: str | Path,
    evaluation_path: str | Path,
    dataset_manifest: str | Path,
    catalog_root: str | Path,
    output_dir: str | Path,
    profile_id: str,
    device: str = "cpu",
) -> dict[str, object]:
    """Package an audio candidate after worker-local independent evaluation."""
    from src.chart_transform_profile import package_chart_transform_profile  # noqa: PLC0415

    return _chart_transform_catalog_profile_call(
        catalog_root=catalog_root,
        dataset_manifest=dataset_manifest,
        candidate_root=experiment_dir,
        scratch_output=output_dir,
        call=lambda audio_manifest: package_chart_transform_profile(
            experiment_dir=experiment_dir,
            evaluation_path=evaluation_path,
            dataset_manifest=dataset_manifest,
            output_dir=output_dir,
            profile_id=profile_id,
            device=device,
            audio_manifest=audio_manifest,
        ),
    )


def _validated_chart_transform_parent(
    parent_bundle: str,
    config: Any,
    *,
    instrument: str,
) -> tuple[Path, dict[str, str]]:
    """Resolve one compatible, integrity-verified transform parent.

    This is the worker's only route from a fine-tune request to a checkpoint.
    In particular, a caller cannot smuggle an arbitrary ``.pt`` location into
    the trainer: the file must be declared by a portable bundle and pass its
    hash/size preflight before the tensor-only loader sees it.
    """
    component_id = _chart_transform_component_id(
        instrument=instrument,
        source=config.source_difficulty,
        target=config.target_difficulty,
    )
    try:
        preflight = preflight_bundle(parent_bundle, required_components=(component_id,))
        bundle = load_model_bundle(parent_bundle, check_files=True)
    except BundleValidationError as error:
        raise WorkerRequestError("fine-tune parent bundle failed verification") from error
    component = bundle.component(component_id)
    if component is None or component.checkpoint is None or component.config is None:
        raise WorkerRequestError(
            "fine-tune parent does not declare a complete chart-transform component"
        )
    if component.architecture != "EventTransformMLP/v1":
        raise WorkerRequestError(
            "fine-tune parent has an incompatible chart-transform architecture"
        )
    if component.preprocessing != "midi-five-lane-events/v1":
        raise WorkerRequestError("fine-tune parent has incompatible chart-transform preprocessing")

    try:
        parent_config = json.loads(component.config.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("fine-tune parent configuration is unreadable") from error
    if not isinstance(parent_config, dict):
        raise WorkerRequestError("fine-tune parent configuration must be an object")
    compatible_fields = (
        "source_difficulty",
        "target_difficulty",
        "lane_count",
        "hidden_dim",
        "audio_feature_mode",
        "audio_sample_rate",
        "audio_window_ms",
        "audio_max_duration_seconds",
    )
    if parent_config.get("instrument") != instrument:
        raise WorkerRequestError("fine-tune parent is incompatible: instrument differs")
    for field in compatible_fields:
        if parent_config.get(field) != getattr(config, field):
            raise WorkerRequestError(f"fine-tune parent is incompatible: {field} differs")

    manifest_sha256 = preflight.get("manifest_sha256")
    if not isinstance(manifest_sha256, str) or component.sha256 is None:
        # ``preflight_bundle`` guarantees this today.  Keep the invariant local
        # in case the generic bundle preflight later adds optional components.
        raise WorkerRequestError("fine-tune parent is missing verified identity")
    return component.checkpoint, {
        "model_id": bundle.model_id,
        "manifest_sha256": manifest_sha256,
        "component": component_id,
        "checkpoint_sha256": component.sha256,
    }


def run_training_request(request_path: Path) -> dict[str, object]:
    """Run one explicit, synchronous training job for a worker-supported pipeline.

    OCTAVE owns job scheduling. It can run this command in a background child
    process and persist opaque locations itself; STRUM returns only portable
    model identity, preflight data, and metrics.
    """
    request = _read_train_request(request_path)
    pipeline_id = request["pipeline_id"]
    descriptor = _pipeline_by_id(pipeline_id)
    if descriptor.training_status != "available":
        raise WorkerRequestError("pipeline does not support worker training")
    if pipeline_id == "guitar.onset-fret/v1":
        from src.guitar_worker_training import (  # noqa: PLC0415
            GuitarTrainingError,
            GuitarTrainingOptions,
            run_catalog_guitar_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("Guitar training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Guitar training requires worker-local catalog_root")
        try:
            options = GuitarTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_guitar_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (GuitarTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Guitar training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "bass.onset-fret/v1":
        from src.bass_worker_training import (  # noqa: PLC0415
            BassTrainingError,
            BassTrainingOptions,
            run_catalog_bass_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("Bass training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Bass training requires worker-local catalog_root")
        try:
            options = BassTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_bass_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (BassTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Bass training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "keys.onset-fret/v1":
        from src.keys_worker_training import (  # noqa: PLC0415
            KeysTrainingError,
            KeysTrainingOptions,
            run_catalog_keys_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("Keys training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Keys training requires worker-local catalog_root")
        try:
            options = KeysTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_keys_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (KeysTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Keys training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id in {
        "strum.instrument-chart/pro-guitar/v1",
        "strum.instrument-chart/pro-bass/v1",
        "strum.instrument-chart/pro-keys/v1",
    }:
        if "parent_bundle" in request:
            raise WorkerRequestError("Pro event candidate training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError(
                "Pro event candidate training requires worker-local catalog_root"
            )
        raw_options = dict(request["options"])
        candidate_kind = raw_options.pop("candidate_kind", KNOWN_EVENT_CANDIDATE_KIND)
        try:
            task_kind = _pro_task_kind_for_pipeline(pipeline_id)
            selected_contract = resolve_pro_candidate_contract(task_kind, candidate_kind)
            required_components = (selected_contract.component_id,)
            revision, dirty = _revision()
            if candidate_kind == KNOWN_EVENT_CANDIDATE_KIND:
                from src.pro_event_worker_training import (  # noqa: PLC0415
                    ProEventTrainingOptions,
                    run_catalog_pro_event_training,
                )

                # A renderer may submit every descriptor default.  Proposal
                # sampling defaults carry no meaning for the historical
                # known-event candidate, but non-default values must not be
                # silently misrepresented as having been used.
                for key, default in (("negative_ratio", 4), ("negative_exclusion_ms", 80)):
                    value = raw_options.pop(key, default)
                    if value != default:
                        raise WorkerRequestError(
                            "Pro proposal sampling options require free_running_event_proposal/v1"
                        )
                options = ProEventTrainingOptions.from_mapping(raw_options)
                result = run_catalog_pro_event_training(
                    task_view_path=Path(request["task_view"]),
                    output_dir=Path(request["output"]),
                    catalog_root=Path(catalog_root),
                    pipeline_id=pipeline_id,
                    options=options,
                    strum_revision=revision,
                    strum_source_dirty=dirty,
                )
            elif candidate_kind == FREE_RUNNING_PROPOSAL_CANDIDATE_KIND:
                from src.pro_event_proposal_worker_training import (  # noqa: PLC0415
                    ProEventProposalTrainingOptions,
                    run_catalog_pro_event_proposal_training,
                )

                options = ProEventProposalTrainingOptions.from_mapping(raw_options)
                result = run_catalog_pro_event_proposal_training(
                    task_view_path=Path(request["task_view"]),
                    output_dir=Path(request["output"]),
                    catalog_root=Path(catalog_root),
                    pipeline_id=pipeline_id,
                    options=options,
                    strum_revision=revision,
                    strum_source_dirty=dirty,
                )
            else:  # pragma: no cover - resolver rejects unknown values first
                raise WorkerRequestError("Pro event candidate kind is invalid")
            component_id = result.get("component_id")
            if component_id != selected_contract.component_id:
                raise WorkerRequestError(
                    "Pro candidate component disagrees with its selected contract"
                )
            # Generic preflight only establishes portable-file integrity.  A
            # raw Pro candidate has an additional selected-kind boundary: the
            # produced bundle may contain exactly one mapped component and its
            # manifest/config semantics must still agree with this request.
            validate_pro_candidate_bundle(result["bundle_dir"], selected_contract)
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=required_components
            )
        except ProCandidateContractError as error:
            raise WorkerRequestError(
                "Pro candidate output bundle does not satisfy selected contract"
            ) from error
        except (BundleValidationError, CatalogValidationError):
            raise
        except (OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Pro event candidate training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
            "runtime": result["runtime"],
        }
    if pipeline_id in {"strum.fret-mapper/guitar/v1", "strum.fret-mapper/bass/v1"}:
        from src.fret_mapper_worker_training import (  # noqa: PLC0415
            FretMapperTrainingError,
            FretMapperTrainingOptions,
            run_catalog_fret_mapper_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("fret-mapper training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("fret-mapper training requires worker-local catalog_root")
        try:
            options = FretMapperTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_fret_mapper_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                pipeline_id=pipeline_id,
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (FretMapperTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "fret-mapper training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id in {"strum.section-classifier/guitar/v1", "strum.section-classifier/bass/v1"}:
        from src.section_worker_training import (  # noqa: PLC0415
            SectionTrainingError,
            SectionTrainingOptions,
            run_catalog_section_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("section training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("section training requires worker-local catalog_root")
        try:
            options = SectionTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_section_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                pipeline_id=pipeline_id,
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (SectionTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "section training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "vocals.note-activity/v1":
        from src.vocals_worker_training import (  # noqa: PLC0415
            VocalsTrainingError,
            VocalsTrainingOptions,
            run_catalog_vocals_training,
        )

        if "parent_bundle" in request:
            raise WorkerRequestError("Vocal training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Vocal training requires worker-local catalog_root")
        try:
            options = VocalsTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_vocals_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (VocalsTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Vocal training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "vocals.phrase-boundaries/v1":
        from src.vocals_phrase_worker_training import (  # noqa: PLC0415
            VocalPhraseTrainingError,
            run_catalog_vocal_phrase_training,
        )
        from src.vocals_worker_training import VocalsTrainingOptions  # noqa: PLC0415

        if "parent_bundle" in request:
            raise WorkerRequestError("Vocal phrase training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Vocal phrase training requires worker-local catalog_root")
        try:
            options = VocalsTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_vocal_phrase_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (VocalPhraseTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Vocal phrase training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "vocals.lyric-alignment/v1":
        from src.vocals_lyric_worker_training import (  # noqa: PLC0415
            VocalLyricTrainingError,
            run_catalog_vocal_lyric_training,
        )
        from src.vocals_worker_training import VocalsTrainingOptions  # noqa: PLC0415

        if "parent_bundle" in request:
            raise WorkerRequestError("Vocal lyric training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Vocal lyric training requires worker-local catalog_root")
        try:
            options = VocalsTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_vocal_lyric_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (VocalLyricTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Vocal lyric training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "vocals.talky-activity/v1":
        from src.vocals_talky_worker_training import (  # noqa: PLC0415
            VocalTalkyTrainingError,
            run_catalog_vocal_talky_training,
        )
        from src.vocals_worker_training import VocalsTrainingOptions  # noqa: PLC0415

        if "parent_bundle" in request:
            raise WorkerRequestError("Vocal talky training does not accept parent_bundle")
        catalog_root = request.get("catalog_root")
        if not isinstance(catalog_root, str) or not catalog_root:
            raise WorkerRequestError("Vocal talky training requires worker-local catalog_root")
        try:
            options = VocalsTrainingOptions.from_mapping(request["options"])
            revision, _dirty = _revision()
            result = run_catalog_vocal_talky_training(
                task_view_path=Path(request["task_view"]),
                output_dir=Path(request["output"]),
                catalog_root=Path(catalog_root),
                options=options,
                strum_revision=revision,
            )
            preflight = preflight_bundle(
                result["bundle_dir"], required_components=descriptor.checkpoint_outputs
            )
        except (BundleValidationError, CatalogValidationError):
            raise
        except (VocalTalkyTrainingError, OSError, TypeError, ValueError) as error:
            raise WorkerRequestError(
                "Vocal talky training request failed validation or execution"
            ) from error
        return {
            "status": "completed",
            "pipeline_id": pipeline_id,
            "model_id": preflight["model_id"],
            "bundle_name": Path(result["bundle_dir"]).name,
            "manifest_sha256": preflight["manifest_sha256"],
            "components": preflight["components"],
            "metrics": result["metrics"],
            "deployment_status": result["deployment_status"],
        }
    if pipeline_id == "drums.onset-classifier/v1":
        from src.drums_onset_training import (  # noqa: PLC0415
            DrumsTrainingError,
            run_drums_onset_training,
        )

        try:
            return _run_without_legacy_output(
                lambda: run_drums_onset_training(
                    request["task_view"],
                    request["output"],
                    request["options"],
                    catalog_root=request["catalog_root"],
                )
            )
        except DrumsTrainingError as error:
            raise WorkerRequestError("Drums onset training request failed validation") from error
    if pipeline_id != "chart_transform.five_lane/v1":
        raise WorkerRequestError("pipeline has no worker training handler")

    # Importing PyTorch belongs to an actual job, not `strum-worker probe`.
    from scripts.train_chart_transform import (  # noqa: PLC0415
        DatasetValidationError,
        TrainingConfig,
        train,
    )

    options = request["options"]
    permitted = {
        "model_id",
        "checkpoint_mode",
        "parent_artifact_id",
        "seed",
        "lane_count",
        "alignment_tolerance_ms",
        "hidden_dim",
        "learning_rate",
        "epochs",
        "device",
        "audio_sample_rate",
        "audio_window_ms",
        "audio_max_duration_seconds",
        "strum_revision",
    }
    if set(options) - permitted or not isinstance(options.get("model_id"), str):
        raise WorkerRequestError("invalid chart-transform training options")
    checkpoint_mode = options.get("checkpoint_mode", "fresh")
    if checkpoint_mode == "resume":
        raise WorkerRequestError(
            "chart-transform resume is not supported; use fresh or fine_tune with a verified bundle"
        )
    if checkpoint_mode not in {"fresh", "fine_tune"}:
        raise WorkerRequestError("chart-transform checkpoint_mode must be fresh or fine_tune")
    parent_artifact_id = options.get("parent_artifact_id")
    if parent_artifact_id is not None and (
        not isinstance(parent_artifact_id, str) or not parent_artifact_id
    ):
        raise WorkerRequestError("chart-transform parent_artifact_id must be a non-empty string")
    parent_bundle = request.get("parent_bundle")
    if checkpoint_mode == "fresh" and (parent_bundle is not None or parent_artifact_id is not None):
        raise WorkerRequestError("fresh chart-transform training must not select a parent artifact")
    if checkpoint_mode == "fine_tune" and (parent_bundle is None or parent_artifact_id is None):
        raise WorkerRequestError(
            "fine_tune chart-transform training requires a parent_artifact_id and private parent_bundle"
        )
    try:
        dataset = json.loads(Path(request["task_view"]).read_text(encoding="utf-8"))
        task_view = dataset.get("task_view") if isinstance(dataset, dict) else None
        task_audio = task_view.get("audio_conditioning") if isinstance(task_view, dict) else None
        audio_feature_mode = (
            task_audio.get("mode", "none") if isinstance(task_audio, dict) else "none"
        )
        if audio_feature_mode not in {"none", "rms_onset_v1"}:
            raise WorkerRequestError("chart-transform task view has unsupported audio conditioning")
        config_values: dict[str, Any] = {
            "dataset_manifest": request["task_view"],
            "output_dir": request["output"],
            "model_id": options["model_id"],
            "source_difficulty": dataset["source_difficulty"],
            "target_difficulty": dataset["target_difficulty"],
            "checkpoint_mode": checkpoint_mode,
            "audio_feature_mode": audio_feature_mode,
            **{
                key: value
                for key, value in options.items()
                if key
                not in {
                    "model_id",
                    "checkpoint_mode",
                    "parent_artifact_id",
                }
            },
        }
        # Build a typed compatibility target before a parent checkpoint has
        # been resolved.  ``fine_tune`` becomes valid only once the verified
        # parent supplies its declared checkpoint below.
        compatibility_config_values = {
            **config_values,
            "checkpoint_mode": "fresh" if checkpoint_mode == "fine_tune" else checkpoint_mode,
        }
        if audio_feature_mode != "none":
            # This object is used only to check a prospective fine-tune
            # parent. The real, worker-private manifest is created below.
            compatibility_config_values["audio_manifest"] = "worker-private-audio-manifest"
        config = TrainingConfig.from_mapping(compatibility_config_values)
        if parent_bundle is not None:
            instrument = dataset.get("instrument", "guitar")
            if not isinstance(instrument, str):
                raise WorkerRequestError("chart-transform task view has an invalid instrument")
            init_checkpoint, parent_provenance = _validated_chart_transform_parent(
                parent_bundle, config, instrument=instrument
            )
            config = TrainingConfig.from_mapping(
                {
                    **config_values,
                    "init_checkpoint": str(init_checkpoint),
                    "parent_provenance": parent_provenance,
                }
            )
        private_audio_dir: tempfile.TemporaryDirectory[str] | None = None
        audio_provenance: dict[str, object] | None = None
        try:
            if audio_feature_mode != "none":
                catalog_root = request.get("catalog_root")
                if not isinstance(catalog_root, str) or not catalog_root:
                    raise WorkerRequestError(
                        "audio-conditioned chart training requires worker-local catalog_root"
                    )
                instrument = dataset.get("instrument")
                if not isinstance(instrument, str):
                    raise WorkerRequestError("chart-transform task view has an invalid instrument")
                private_audio_dir, audio_manifest, audio_provenance = (
                    _chart_transform_catalog_audio_manifest(
                        dataset=dataset,
                        catalog_root=catalog_root,
                        output_dir=Path(request["output"]),
                    )
                )
                config = TrainingConfig.from_mapping(
                    {
                        **config_values,
                        "audio_manifest": str(audio_manifest),
                        "init_checkpoint": str(init_checkpoint)
                        if parent_bundle is not None
                        else None,
                        "parent_provenance": parent_provenance
                        if parent_bundle is not None
                        else None,
                    }
                )
            result = train(config)
        finally:
            if private_audio_dir is not None:
                private_audio_dir.cleanup()
        if audio_provenance is not None:
            _write_chart_transform_catalog_audio_provenance(
                Path(result["bundle_dir"]), audio_provenance
            )
        preflight = preflight_bundle(result["bundle_dir"])
    except WorkerRequestError:
        raise
    except (KeyError, OSError, TypeError, ValueError, DatasetValidationError) as error:
        raise WorkerRequestError("chart-transform training request failed validation") from error
    metrics = result["metrics"]
    evaluation_split = "calibration" if "calibration" in metrics else "validation"
    return {
        "status": "completed",
        "pipeline_id": pipeline_id,
        "model_id": preflight["model_id"],
        "bundle_name": Path(result["bundle_dir"]).name,
        "manifest_sha256": preflight["manifest_sha256"],
        "components": preflight["components"],
        "evaluation_split": evaluation_split,
        "evaluation": metrics[evaluation_split],
        "deployment_status": "requires_transform_profile_evaluation_and_promotion",
    }


PROMOTION_RESULT_FORMAT = "strum-post-train-job-result/v1"


def _promotion_job_by_id(pipeline_id: str, job_id: str) -> PromotionJobDescriptor:
    descriptor = _pipeline_by_id(pipeline_id)
    for job in descriptor.promotion_jobs:
        if job.id == job_id:
            return job
    raise WorkerRequestError("post-training job is not available for this pipeline")


def _read_promotion_request(
    request_path: Path,
) -> tuple[dict[str, Any], PromotionJobDescriptor]:
    """Read a strict, host-owned post-training request without echoing paths."""
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError("post-training request is unreadable or not valid JSON") from error
    if not isinstance(raw, dict):
        raise WorkerRequestError("post-training request must be an object")
    pipeline_id, job_id = raw.get("pipeline_id"), raw.get("job_id")
    if not isinstance(pipeline_id, str) or not isinstance(job_id, str):
        raise WorkerRequestError("post-training request pipeline_id and job_id are required")
    job = _promotion_job_by_id(pipeline_id, job_id)
    required = {"pipeline_id", "job_id", "options", *job.private_request_fields}
    permitted = required | set(job.optional_private_request_fields)
    if set(raw) != required and not (required <= set(raw) <= permitted):
        raise WorkerRequestError("post-training request has unsupported fields")
    if not isinstance(raw.get("options"), dict):
        raise WorkerRequestError("post-training job options must be an object")
    for field in (*job.private_request_fields, *job.optional_private_request_fields):
        if field in raw and (not isinstance(raw[field], str) or not raw[field]):
            raise WorkerRequestError("post-training private locations must be non-empty strings")
    schema = job.options_schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    required_options = schema.get("required") if isinstance(schema, dict) else None
    if not isinstance(properties, dict) or not isinstance(required_options, list):
        raise RuntimeError("post-training job descriptor has invalid options schema")
    if set(raw["options"]) - set(properties) or not set(required_options) <= set(raw["options"]):
        raise WorkerRequestError("post-training options do not match the advertised schema")
    return raw, job


def _promotion_options(job: PromotionJobDescriptor, raw_options: dict[str, Any]) -> dict[str, Any]:
    """Apply only descriptor-declared JSON defaults before strict handler dispatch."""
    schema = job.options_schema
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):  # pragma: no cover - descriptor guard
        raise RuntimeError("post-training job descriptor has invalid options schema")
    values = dict(raw_options)
    for key, definition in properties.items():
        if key not in values and isinstance(definition, dict) and "default" in definition:
            values[key] = definition["default"]
    return values


def _promotion_result_summary(
    *, pipeline_id: str, job: PromotionJobDescriptor, result: dict[str, object]
) -> dict[str, object]:
    """Return only allowlisted, path-free result facts for host rendering.

    Worker handlers may evolve to include diagnostic strings.  Do not attempt
    to recognize a path inside those values: promotion output is a strict DTO
    and omits every free-form field by default.
    """
    if not isinstance(result, dict):  # pragma: no cover - handler guard
        raise WorkerRequestError("post-training job returned an invalid result")
    summary: dict[str, object] = {}
    safe_token = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
    safe_capability = re.compile(r"[a-z][a-z0-9._-]*/v[1-9][0-9]*\Z")
    for key in (
        "model_id",
        "profile_id",
        "component_id",
        "instrument",
        "split",
        "source_difficulty",
        "target_difficulty",
        "dataset_id",
        "quality_policy_id",
        "quality_gate_status",
        "task_view_id",
        "bundle_name",
        "execution_scope",
    ):
        value = result.get(key)
        if isinstance(value, str) and safe_token.fullmatch(value):
            summary[key] = value
    capability = result.get("capability")
    if isinstance(capability, str) and safe_capability.fullmatch(capability):
        summary["capability"] = capability
    for key in (
        "audio_manifest_sha256",
        "bundle_manifest_sha256",
        "candidate_lineage_sha256",
        "candidate_manifest_sha256",
        "component_configuration_sha256",
        "component_sha256",
        "dataset_manifest_sha256",
        "dataset_records_sha256",
        "manifest_sha256",
        "quality_policy_sha256",
        "task_view_sha256",
    ):
        value = result.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value):
            summary[key] = value
    records = result.get("records_evaluated")
    if isinstance(records, int) and not isinstance(records, bool) and records >= 0:
        summary["records_evaluated"] = records
    metrics = result.get("metrics")
    if isinstance(metrics, dict) and all(
        isinstance(key, str)
        and re.fullmatch(r"[A-Za-z][A-Za-z0-9._-]*\Z", key)
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        for key, value in metrics.items()
    ):
        summary["metrics"] = {key: float(value) for key, value in metrics.items()}
    return {
        "schema_version": 1,
        "format": PROMOTION_RESULT_FORMAT,
        "status": "completed",
        "pipeline_id": pipeline_id,
        "job_id": job.id,
        "output_kind": job.output_kind,
        "deployment_scope": job.deployment_scope,
        "result": summary,
    }


def run_promotion_request(request_path: Path) -> dict[str, object]:
    """Run one descriptor-advertised evaluation or packaging gate.

    This is intentionally a narrow adapter over existing strict evaluators and
    packagers.  It adds no promotion shortcut: every underlying held-out,
    lineage, metric, immutable-copy, and profile validation gate remains the
    authority.
    """
    request, job = _read_promotion_request(request_path)
    pipeline_id = request["pipeline_id"]
    assert isinstance(pipeline_id, str)
    options = _promotion_options(job, request["options"])
    result: dict[str, object]
    try:
        if job.id in {
            "guitar.profile-evaluate/v1",
            "bass.profile-evaluate/v1",
            "keys.profile-evaluate/v1",
        }:
            handlers = {
                "guitar.profile-evaluate/v1": (
                    "src.guitar_profile_packaging",
                    "evaluate_guitar_candidate",
                ),
                "bass.profile-evaluate/v1": (
                    "src.bass_profile_packaging",
                    "evaluate_bass_candidate",
                ),
                "keys.profile-evaluate/v1": (
                    "src.keys_profile_packaging",
                    "evaluate_keys_candidate",
                ),
            }
            module_name, function_name = handlers[job.id]
            module = __import__(module_name, fromlist=[function_name])
            evaluate = getattr(module, function_name)
            result = evaluate(
                bundle_root=Path(request["bundle_root"]),
                task_view_path=Path(request["task_view"]),
                catalog_root=Path(request["catalog_root"]),
                output_path=Path(request["output"]),
                device=options["device"],
                tolerance_ms=options["tolerance_ms"],
                limit_songs=options["limit_songs"],
            )
        elif job.id in {
            "guitar.profile-package/v1",
            "bass.profile-package/v1",
            "keys.profile-package/v1",
        }:
            handlers = {
                "guitar.profile-package/v1": (
                    "src.guitar_profile_packaging",
                    "package_guitar_profile",
                ),
                "bass.profile-package/v1": ("src.bass_profile_packaging", "package_bass_profile"),
                "keys.profile-package/v1": ("src.keys_profile_packaging", "package_keys_profile"),
            }
            module_name, function_name = handlers[job.id]
            module = __import__(module_name, fromlist=[function_name])
            package = getattr(module, function_name)
            result = package(
                experiment_dir=Path(request["experiment"]),
                evaluation_path=Path(request["evaluation"]),
                output_dir=Path(request["output"]),
                profile_id=options["profile_id"],
            )
        elif job.id == "chart-transform.profile-evaluate/v1":
            if "catalog_root" in request:
                result = evaluate_catalog_chart_transform_candidate(
                    bundle_root=request["bundle_root"],
                    dataset_manifest=request["dataset_manifest"],
                    catalog_root=request["catalog_root"],
                    output_path=request["output"],
                    device=options["device"],
                )
            else:
                from src.chart_transform_profile import (  # noqa: PLC0415
                    evaluate_chart_transform_candidate,
                )

                result = evaluate_chart_transform_candidate(
                    bundle_root=request["bundle_root"],
                    dataset_manifest=request["dataset_manifest"],
                    output_path=request["output"],
                    device=options["device"],
                )
        elif job.id == "chart-transform.profile-package/v1":
            if "catalog_root" in request:
                result = package_catalog_chart_transform_profile(
                    experiment_dir=request["experiment"],
                    evaluation_path=request["evaluation"],
                    dataset_manifest=request["dataset_manifest"],
                    catalog_root=request["catalog_root"],
                    output_dir=request["output"],
                    profile_id=options["profile_id"],
                    device=options["device"],
                )
            else:
                from src.chart_transform_profile import (
                    package_chart_transform_profile,  # noqa: PLC0415
                )

                result = package_chart_transform_profile(
                    experiment_dir=request["experiment"],
                    evaluation_path=request["evaluation"],
                    dataset_manifest=request["dataset_manifest"],
                    output_dir=request["output"],
                    profile_id=options["profile_id"],
                    device=options["device"],
                )
        elif job.id.startswith("section."):
            from src.section_profile_evaluation import (  # noqa: PLC0415
                evaluate_section_candidate,
                package_section_evaluation_profile,
            )

            instrument = "guitar" if ".guitar." in job.id else "bass"
            if job.id.endswith("profile-evaluate/v1"):
                result = evaluate_section_candidate(
                    bundle_root=Path(request["bundle_root"]),
                    task_view_path=Path(request["task_view"]),
                    catalog_root=Path(request["catalog_root"]),
                    output_path=Path(request["output"]),
                    instrument=instrument,
                    device=options["device"],
                )
            else:
                result = package_section_evaluation_profile(
                    experiment_dir=Path(request["experiment"]),
                    evaluation_path=Path(request["evaluation"]),
                    output_dir=Path(request["output"]),
                    profile_id=options["profile_id"],
                    instrument=instrument,
                    minimum_accuracy=options["minimum_accuracy"],
                    maximum_expected_calibration_error=options[
                        "maximum_expected_calibration_error"
                    ],
                )
        elif job.id == "drums.onset-classifier.package-evaluation/v1":
            from src.drums_onset_profile_package import (  # noqa: PLC0415
                package_drums_onset_experiment,
            )

            result = package_drums_onset_experiment(request["experiment_root"], request["output"])
        else:  # pragma: no cover - descriptor/dispatch parity guard
            raise RuntimeError("post-training job has no handler")
    except (BundleValidationError, CatalogValidationError):
        raise
    except Exception as error:
        raise WorkerRequestError("post-training job failed validation or execution") from error
    return _promotion_result_summary(pipeline_id=pipeline_id, job=job, result=result)


def package_checkpoint_request(request_path: Path) -> dict[str, object]:
    """Package one worker-owned experiment through a strict, path-private request."""
    try:
        raw = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkerRequestError(
            "checkpoint package request is unreadable or not valid JSON"
        ) from error
    if not isinstance(raw, dict) or set(raw) != {"pipeline_id", "experiment_root", "output"}:
        raise WorkerRequestError("checkpoint package request has unsupported fields")
    if raw.get("pipeline_id") != "drums.onset-classifier/v1":
        raise WorkerRequestError("checkpoint package pipeline is unsupported")
    if not all(isinstance(raw.get(key), str) and raw[key] for key in ("experiment_root", "output")):
        raise WorkerRequestError("checkpoint package locations must be non-empty strings")
    from src.drums_onset_profile_package import (  # noqa: PLC0415
        DrumsProfilePackagingError,
        package_drums_onset_experiment,
    )

    try:
        return package_drums_onset_experiment(raw["experiment_root"], raw["output"])
    except DrumsProfilePackagingError as error:
        raise WorkerRequestError("Drums checkpoint package request failed validation") from error


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Versioned STRUM worker contract")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe", help="describe runtime capabilities").add_argument(
        "--json", action="store_true"
    )
    pipeline = commands.add_parser("pipeline", help="inspect STRUM pipelines")
    pipeline_commands = pipeline.add_subparsers(dest="pipeline_command", required=True)
    pipeline_commands.add_parser("list", help="list pipeline descriptors").add_argument(
        "--json", action="store_true"
    )
    catalog = commands.add_parser("catalog", help="validate OCTAVE song-source catalogs")
    catalog_commands = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_inspect = catalog_commands.add_parser("inspect", help="inspect one catalog")
    catalog_inspect.add_argument("--catalog-root", type=Path, required=True)
    catalog_inspect.add_argument("--pipeline")
    catalog_inspect.add_argument(
        "--options",
        help="JSON preparation-selection subset; values are used locally and never echoed",
    )
    catalog_inspect.add_argument("--json", action="store_true")
    dataset = commands.add_parser("dataset", help="prepare STRUM task views")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_commands.add_parser("prepare", help="materialize one catalog task view")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--json", action="store_true")
    prepare.add_argument("--json-events", action="store_true")
    vocal = commands.add_parser(
        "vocal", help="revalidate Lead-Vocal catalog data without loading models"
    )
    vocal_commands = vocal.add_subparsers(dest="vocal_command", required=True)
    lead_admission = vocal_commands.add_parser(
        "lead-admission", help="recompute four-view lead-label data admission evidence"
    )
    lead_admission.add_argument("--catalog-root", type=Path, required=True)
    lead_admission.add_argument("--activity-task-view", type=Path, required=True)
    lead_admission.add_argument("--phrase-task-view", type=Path, required=True)
    lead_admission.add_argument("--lyric-task-view", type=Path, required=True)
    lead_admission.add_argument("--talky-task-view", type=Path, required=True)
    lead_admission.add_argument("--json", action="store_true")
    training = commands.add_parser("train", help="run worker-managed training")
    training_commands = training.add_subparsers(dest="training_command", required=True)
    training_run = training_commands.add_parser("run", help="run one synchronous training job")
    training_run.add_argument("--request", type=Path, required=True)
    training_run.add_argument("--json", action="store_true")
    training_run.add_argument("--json-events", action="store_true")
    training_start = training_commands.add_parser(
        "start", help="start one OCTAVE-supervised training job"
    )
    training_start.add_argument("--request", type=Path, required=True)
    training_start.add_argument("--json-events", action="store_true")
    promotion = commands.add_parser(
        "promotion", help="run descriptor-advertised post-training evaluation and package jobs"
    )
    promotion_commands = promotion.add_subparsers(dest="promotion_command", required=True)
    promotion_start = promotion_commands.add_parser(
        "start", help="run one OCTAVE-supervised post-training job"
    )
    promotion_start.add_argument("--request", type=Path, required=True)
    promotion_start.add_argument("--json-events", action="store_true")
    guitar = commands.add_parser("guitar", help="evaluate and package Guitar V1 profiles")
    guitar_commands = guitar.add_subparsers(dest="guitar_command", required=True)
    guitar_profile = guitar_commands.add_parser("profile", help="manage Guitar neural profiles")
    guitar_profile_commands = guitar_profile.add_subparsers(
        dest="guitar_profile_command", required=True
    )
    guitar_evaluate = guitar_profile_commands.add_parser(
        "evaluate", help="evaluate an un-packaged catalog-trained Guitar pair"
    )
    guitar_evaluate.add_argument("--bundle-root", type=Path, required=True)
    guitar_evaluate.add_argument("--task-view", type=Path, required=True)
    guitar_evaluate.add_argument("--catalog-root", type=Path, required=True)
    guitar_evaluate.add_argument("--output", type=Path, required=True)
    guitar_evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    guitar_evaluate.add_argument("--tolerance-ms", type=float, default=50.0)
    guitar_evaluate.add_argument("--limit-songs", type=int, default=0)
    guitar_evaluate.add_argument("--json", action="store_true")
    guitar_package = guitar_profile_commands.add_parser(
        "package", help="copy an evaluated Guitar experiment into a deployable bundle"
    )
    guitar_package.add_argument("--experiment", type=Path, required=True)
    guitar_package.add_argument("--evaluation", type=Path, required=True)
    guitar_package.add_argument("--output", type=Path, required=True)
    guitar_package.add_argument("--profile", required=True)
    guitar_package.add_argument("--json", action="store_true")
    bass = commands.add_parser("bass", help="evaluate and package Bass V1 profiles")
    bass_commands = bass.add_subparsers(dest="bass_command", required=True)
    bass_profile = bass_commands.add_parser("profile", help="manage Bass neural profiles")
    bass_profile_commands = bass_profile.add_subparsers(dest="bass_profile_command", required=True)
    bass_evaluate = bass_profile_commands.add_parser(
        "evaluate", help="evaluate an un-packaged catalog-trained Bass pair"
    )
    bass_evaluate.add_argument("--bundle-root", type=Path, required=True)
    bass_evaluate.add_argument("--task-view", type=Path, required=True)
    bass_evaluate.add_argument("--catalog-root", type=Path, required=True)
    bass_evaluate.add_argument("--output", type=Path, required=True)
    bass_evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    bass_evaluate.add_argument("--tolerance-ms", type=float, default=50.0)
    bass_evaluate.add_argument("--limit-songs", type=int, default=0)
    bass_evaluate.add_argument("--json", action="store_true")
    bass_package = bass_profile_commands.add_parser(
        "package", help="copy an evaluated Bass experiment into a deployable bundle"
    )
    bass_package.add_argument("--experiment", type=Path, required=True)
    bass_package.add_argument("--evaluation", type=Path, required=True)
    bass_package.add_argument("--output", type=Path, required=True)
    bass_package.add_argument("--profile", required=True)
    bass_package.add_argument("--json", action="store_true")
    keys = commands.add_parser("keys", help="evaluate and package Keys V1 profiles")
    keys_commands = keys.add_subparsers(dest="keys_command", required=True)
    keys_profile = keys_commands.add_parser("profile", help="manage Keys neural profiles")
    keys_profile_commands = keys_profile.add_subparsers(dest="keys_profile_command", required=True)
    keys_evaluate = keys_profile_commands.add_parser(
        "evaluate", help="evaluate an un-packaged catalog-trained Keys pair"
    )
    keys_evaluate.add_argument("--bundle-root", type=Path, required=True)
    keys_evaluate.add_argument("--task-view", type=Path, required=True)
    keys_evaluate.add_argument("--catalog-root", type=Path, required=True)
    keys_evaluate.add_argument("--output", type=Path, required=True)
    keys_evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    keys_evaluate.add_argument("--tolerance-ms", type=float, default=50.0)
    keys_evaluate.add_argument("--limit-songs", type=int, default=0)
    keys_evaluate.add_argument("--json", action="store_true")
    keys_package = keys_profile_commands.add_parser(
        "package", help="copy an evaluated Keys experiment into a deployable bundle"
    )
    keys_package.add_argument("--experiment", type=Path, required=True)
    keys_package.add_argument("--evaluation", type=Path, required=True)
    keys_package.add_argument("--output", type=Path, required=True)
    keys_package.add_argument("--profile", required=True)
    keys_package.add_argument("--json", action="store_true")
    section = commands.add_parser(
        "section", help="calibrate and package SectionClassifier evaluation profiles"
    )
    section_commands = section.add_subparsers(dest="section_command", required=True)
    section_profile = section_commands.add_parser(
        "profile", help="manage non-executable Section classifier profiles"
    )
    section_profile_commands = section_profile.add_subparsers(
        dest="section_profile_command", required=True
    )
    section_evaluate = section_profile_commands.add_parser(
        "evaluate", help="calibrate on validation and evaluate held-out test windows"
    )
    section_evaluate.add_argument("--bundle-root", type=Path, required=True)
    section_evaluate.add_argument("--task-view", type=Path, required=True)
    section_evaluate.add_argument("--catalog-root", type=Path, required=True)
    section_evaluate.add_argument("--output", type=Path, required=True)
    section_evaluate.add_argument("--instrument", choices=("guitar", "bass"), required=True)
    section_evaluate.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    section_evaluate.add_argument("--json", action="store_true")
    section_package = section_profile_commands.add_parser(
        "package", help="package a held-out Section evaluator, never a chart profile"
    )
    section_package.add_argument("--experiment", type=Path, required=True)
    section_package.add_argument("--evaluation", type=Path, required=True)
    section_package.add_argument("--output", type=Path, required=True)
    section_package.add_argument("--profile", required=True)
    section_package.add_argument("--instrument", choices=("guitar", "bass"), required=True)
    section_package.add_argument("--minimum-accuracy", type=float, required=True)
    section_package.add_argument("--maximum-expected-calibration-error", type=float, required=True)
    section_package.add_argument("--json", action="store_true")
    transform = commands.add_parser(
        "transform", help="evaluate and promote five-lane difficulty transforms"
    )
    transform_commands = transform.add_subparsers(dest="transform_command", required=True)
    transform_profile = transform_commands.add_parser(
        "profile", help="manage held-out transform promotion artifacts"
    )
    transform_profile_commands = transform_profile.add_subparsers(
        dest="transform_profile_command", required=True
    )
    transform_evaluate = transform_profile_commands.add_parser(
        "evaluate", help="recompute declared held-out transform metrics"
    )
    transform_evaluate.add_argument("--bundle-root", type=Path, required=True)
    transform_evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    transform_evaluate.add_argument("--output", type=Path, required=True)
    transform_evaluate.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    transform_evaluate_audio = transform_evaluate.add_mutually_exclusive_group()
    transform_evaluate_audio.add_argument("--audio-manifest", type=Path)
    transform_evaluate_audio.add_argument("--catalog-root", type=Path)
    transform_evaluate.add_argument("--json", action="store_true")
    transform_package = transform_profile_commands.add_parser(
        "package", help="copy one evaluated transform candidate into an executable profile"
    )
    transform_package.add_argument("--experiment", type=Path, required=True)
    transform_package.add_argument("--evaluation", type=Path, required=True)
    transform_package.add_argument("--dataset-manifest", type=Path, required=True)
    transform_package.add_argument("--output", type=Path, required=True)
    transform_package.add_argument("--profile", required=True)
    transform_package.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    transform_package_audio = transform_package.add_mutually_exclusive_group()
    transform_package_audio.add_argument("--audio-manifest", type=Path)
    transform_package_audio.add_argument("--catalog-root", type=Path)
    transform_package.add_argument("--json", action="store_true")
    model = commands.add_parser("model", help="inspect model bundles")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    preflight = model_commands.add_parser("preflight", help="validate a deployable model bundle")
    preflight.add_argument("--model-root", type=Path, required=True)
    preflight.add_argument("--require-component", action="append", default=[])
    preflight.add_argument("--json", action="store_true")
    checkpoint = commands.add_parser("checkpoint", help="inspect checkpoint bundle metadata")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    discover = checkpoint_commands.add_parser(
        "discover", help="discover valid model bundles below one private model folder"
    )
    discover.add_argument("--model-root", type=Path, required=True)
    discover.add_argument("--json", action="store_true")
    inspect = checkpoint_commands.add_parser("inspect", help="inspect a checkpoint bundle")
    inspect.add_argument("--model-root", type=Path, required=True)
    inspect.add_argument("--json", action="store_true")
    package = checkpoint_commands.add_parser(
        "package", help="package a worker experiment as an evaluation-only bundle"
    )
    package.add_argument("--request", type=Path, required=True)
    package.add_argument("--json", action="store_true")
    package.add_argument("--json-events", action="store_true")
    inference = commands.add_parser("inference", help="validate deployable inference profiles")
    inference_commands = inference.add_subparsers(dest="inference_command", required=True)
    profile = inference_commands.add_parser("profile", help="inspect one inference profile")
    profile_commands = profile.add_subparsers(dest="profile_command", required=True)
    validate_profile = profile_commands.add_parser(
        "validate", help="validate one inference profile"
    )
    validate_profile.add_argument("--model-root", type=Path, required=True)
    validate_profile.add_argument("--profile", required=True)
    validate_profile.add_argument("--difficulty-policy", required=True)
    validate_profile.add_argument("--json", action="store_true")
    chart = commands.add_parser("chart", help="preflight typed auto-chart requests")
    chart_commands = chart.add_subparsers(dest="chart_command", required=True)
    chart_preflight = chart_commands.add_parser(
        "preflight", help="validate a chart profile request"
    )
    chart_preflight.add_argument("--request", type=Path, required=True)
    chart_preflight.add_argument("--json", action="store_true")
    chart_run = chart_commands.add_parser("run", help="execute an explicit chart profile")
    chart_run.add_argument("--request", type=Path, required=True)
    chart_run.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "probe":
            _print_json(_runtime_payload())
            return 0
        if args.command == "pipeline" and args.pipeline_command == "list":
            _print_json(
                {
                    "protocol_version": PROTOCOL_VERSION,
                    "pipelines": [pipeline.as_json() for pipeline in PIPELINES],
                }
            )
            return 0
        if args.command == "catalog" and args.catalog_command == "inspect":
            _print_json(
                inspect_catalog(
                    args.catalog_root,
                    args.pipeline,
                    options=_read_catalog_inspection_options(args.options),
                )
            )
            return 0
        if args.command == "dataset" and args.dataset_command == "prepare":
            if args.json_events:
                return _run_event_stream(
                    args.request, "dataset_prepare", lambda: prepare_dataset_request(args.request)
                )
            _print_json(prepare_dataset_request(args.request))
            return 0
        if args.command == "train" and args.training_command in {"run", "start"}:
            if args.json_events:
                return _run_event_stream(
                    args.request, "training", lambda: run_training_request(args.request)
                )
            _print_json(run_training_request(args.request))
            return 0
        if args.command == "promotion" and args.promotion_command == "start":
            if args.json_events:
                return _run_event_stream(
                    args.request,
                    "post_training_promotion",
                    lambda: run_promotion_request(args.request),
                )
            _print_json(run_promotion_request(args.request))
            return 0
        if args.command == "guitar" and args.guitar_command == "profile":
            from src.guitar_profile_packaging import (  # noqa: PLC0415
                GuitarProfilePackagingError,
                evaluate_guitar_candidate,
                package_guitar_profile,
            )

            if args.guitar_profile_command == "evaluate":
                try:
                    _print_json(
                        evaluate_guitar_candidate(
                            bundle_root=args.bundle_root,
                            task_view_path=args.task_view,
                            catalog_root=args.catalog_root,
                            output_path=args.output,
                            device=args.device,
                            tolerance_ms=args.tolerance_ms,
                            limit_songs=args.limit_songs,
                        )
                    )
                except GuitarProfilePackagingError as error:
                    raise WorkerRequestError(
                        "Guitar profile evaluation request is invalid"
                    ) from error
                return 0
            try:
                _print_json(
                    package_guitar_profile(
                        experiment_dir=args.experiment,
                        evaluation_path=args.evaluation,
                        output_dir=args.output,
                        profile_id=args.profile,
                    )
                )
            except GuitarProfilePackagingError as error:
                raise WorkerRequestError("Guitar profile packaging request is invalid") from error
            return 0
        if args.command == "bass" and args.bass_command == "profile":
            from src.bass_profile_packaging import (  # noqa: PLC0415
                BassProfilePackagingError,
                evaluate_bass_candidate,
                package_bass_profile,
            )

            if args.bass_profile_command == "evaluate":
                try:
                    _print_json(
                        evaluate_bass_candidate(
                            bundle_root=args.bundle_root,
                            task_view_path=args.task_view,
                            catalog_root=args.catalog_root,
                            output_path=args.output,
                            device=args.device,
                            tolerance_ms=args.tolerance_ms,
                            limit_songs=args.limit_songs,
                        )
                    )
                except BassProfilePackagingError as error:
                    raise WorkerRequestError(
                        "Bass profile evaluation request is invalid"
                    ) from error
                return 0
            try:
                _print_json(
                    package_bass_profile(
                        experiment_dir=args.experiment,
                        evaluation_path=args.evaluation,
                        output_dir=args.output,
                        profile_id=args.profile,
                    )
                )
            except BassProfilePackagingError as error:
                raise WorkerRequestError("Bass profile packaging request is invalid") from error
            return 0
        if args.command == "keys" and args.keys_command == "profile":
            from src.keys_profile_packaging import (  # noqa: PLC0415
                KeysProfilePackagingError,
                evaluate_keys_candidate,
                package_keys_profile,
            )

            if args.keys_profile_command == "evaluate":
                try:
                    _print_json(
                        evaluate_keys_candidate(
                            bundle_root=args.bundle_root,
                            task_view_path=args.task_view,
                            catalog_root=args.catalog_root,
                            output_path=args.output,
                            device=args.device,
                            tolerance_ms=args.tolerance_ms,
                            limit_songs=args.limit_songs,
                        )
                    )
                except KeysProfilePackagingError as error:
                    raise WorkerRequestError(
                        "Keys profile evaluation request is invalid"
                    ) from error
                return 0
            try:
                _print_json(
                    package_keys_profile(
                        experiment_dir=args.experiment,
                        evaluation_path=args.evaluation,
                        output_dir=args.output,
                        profile_id=args.profile,
                    )
                )
            except KeysProfilePackagingError as error:
                raise WorkerRequestError("Keys profile packaging request is invalid") from error
            return 0
        if args.command == "section" and args.section_command == "profile":
            from src.section_profile_evaluation import (  # noqa: PLC0415
                SectionProfileEvaluationError,
                evaluate_section_candidate,
                package_section_evaluation_profile,
            )

            try:
                if args.section_profile_command == "evaluate":
                    _print_json(
                        evaluate_section_candidate(
                            bundle_root=args.bundle_root,
                            task_view_path=args.task_view,
                            catalog_root=args.catalog_root,
                            output_path=args.output,
                            instrument=args.instrument,
                            device=args.device,
                        )
                    )
                else:
                    _print_json(
                        package_section_evaluation_profile(
                            experiment_dir=args.experiment,
                            evaluation_path=args.evaluation,
                            output_dir=args.output,
                            profile_id=args.profile,
                            instrument=args.instrument,
                            minimum_accuracy=args.minimum_accuracy,
                            maximum_expected_calibration_error=(
                                args.maximum_expected_calibration_error
                            ),
                        )
                    )
            except SectionProfileEvaluationError as error:
                raise WorkerRequestError("Section profile request is invalid") from error
            return 0
        if args.command == "transform" and args.transform_command == "profile":
            from src.chart_transform_profile import (  # noqa: PLC0415
                ChartTransformPromotionError,
                evaluate_chart_transform_candidate,
                package_chart_transform_profile,
            )

            try:
                if args.transform_profile_command == "evaluate":
                    if args.catalog_root is not None:
                        _print_json(
                            evaluate_catalog_chart_transform_candidate(
                                bundle_root=args.bundle_root,
                                dataset_manifest=args.dataset_manifest,
                                catalog_root=args.catalog_root,
                                output_path=args.output,
                                device=args.device,
                            )
                        )
                    else:
                        _print_json(
                            evaluate_chart_transform_candidate(
                                bundle_root=args.bundle_root,
                                dataset_manifest=args.dataset_manifest,
                                output_path=args.output,
                                device=args.device,
                                audio_manifest=args.audio_manifest,
                            )
                        )
                else:
                    if args.catalog_root is not None:
                        _print_json(
                            package_catalog_chart_transform_profile(
                                experiment_dir=args.experiment,
                                evaluation_path=args.evaluation,
                                dataset_manifest=args.dataset_manifest,
                                catalog_root=args.catalog_root,
                                output_dir=args.output,
                                profile_id=args.profile,
                                device=args.device,
                            )
                        )
                    else:
                        _print_json(
                            package_chart_transform_profile(
                                experiment_dir=args.experiment,
                                evaluation_path=args.evaluation,
                                dataset_manifest=args.dataset_manifest,
                                output_dir=args.output,
                                profile_id=args.profile,
                                device=args.device,
                                audio_manifest=args.audio_manifest,
                            )
                        )
            except ChartTransformPromotionError as error:
                raise WorkerRequestError("Transform profile request is invalid") from error
            return 0
        if args.command == "model" and args.model_command == "preflight":
            _print_json(
                preflight_bundle(args.model_root, required_components=args.require_component)
            )
            return 0
        if args.command == "checkpoint" and args.checkpoint_command == "inspect":
            _print_json(inspect_model_bundle(args.model_root))
            return 0
        if args.command == "checkpoint" and args.checkpoint_command == "discover":
            _print_json(discover_model_bundles(args.model_root))
            return 0
        if args.command == "checkpoint" and args.checkpoint_command == "package":
            if args.json_events:
                return _run_event_stream(
                    args.request,
                    "checkpoint_package",
                    lambda: package_checkpoint_request(args.request),
                )
            _print_json(package_checkpoint_request(args.request))
            return 0
        if (
            args.command == "inference"
            and args.inference_command == "profile"
            and args.profile_command == "validate"
        ):
            _print_json(
                validate_inference_profile(
                    args.model_root,
                    profile_id=args.profile,
                    difficulty_policy=args.difficulty_policy,
                )
            )
            return 0
        if args.command == "chart" and args.chart_command == "preflight":
            _print_json(preflight_chart_request(args.request))
            return 0
        if args.command == "chart" and args.chart_command == "run":
            _print_json(run_chart_request(args.request))
            return 0
        if args.command == "vocal" and args.vocal_command == "lead-admission":
            try:
                _print_json(
                    resolve_vocal_lead_catalog_admission(
                        catalog_root=args.catalog_root,
                        task_view_paths={
                            "vocals_activity": args.activity_task_view,
                            "vocals_phrase_boundaries": args.phrase_task_view,
                            "vocals_lyric_alignment": args.lyric_task_view,
                            "vocals_talky_activity": args.talky_task_view,
                        },
                    )
                )
            except VocalLeadCatalogAdmissionError as error:
                raise WorkerRequestError(
                    "lead Vocal catalog admission request is invalid"
                ) from error
            return 0
    except BundleValidationError:
        _print_json(
            {
                "status": "invalid",
                "code": "model_bundle_invalid",
                "message": "model bundle failed validation",
            }
        )
        return 2
    except CatalogValidationError:
        _print_json(
            {"status": "invalid", "code": "catalog_invalid", "message": "catalog failed validation"}
        )
        return 2
    except WorkerRequestError:
        _print_json(
            {"status": "invalid", "code": "request_invalid", "message": "worker request is invalid"}
        )
        return 2
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
