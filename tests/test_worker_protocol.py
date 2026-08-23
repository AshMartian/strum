from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import mido
import pytest
import torch

from src.model_bundle import MANIFEST_FILENAME, BundleValidationError
from src.models.chart_transform import EventTransformMLP
from src.worker import (
    PIPELINES,
    PROTOCOL_VERSION,
    WorkerRequestError,
    _chart_result_contract,
    _revision,
    _run_without_legacy_output,
    _runtime_payload,
    _write_expert_guitar_midi,
    discover_model_bundles,
    inspect_catalog,
    inspect_model_bundle,
    main,
    preflight_bundle,
    preflight_chart_request,
    prepare_dataset_request,
    run_chart_request,
    run_training_request,
    validate_inference_profile,
)


def _bundle(root: Path, component: dict[str, object]) -> Path:
    checkpoint = root / "weights" / "model.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"verified weights")
    component.update(
        {
            "checkpoint": "weights/model.pt",
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "byte_length": checkpoint.stat().st_size,
        }
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "verified-test-bundle",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {"guitar.onset": component},
            }
        )
    )
    return root


def _composed_profile_bundle(root: Path) -> Path:
    components: dict[str, dict[str, object]] = {}
    for component_id in (
        "separation.demucs",
        "guitar.onset",
        "guitar.mapper",
        "guitar.assembly",
    ):
        checkpoint = root / "weights" / f"{component_id}.bin"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(component_id.encode())
        components[component_id] = {
            "checkpoint": checkpoint.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "byte_length": checkpoint.stat().st_size,
        }
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "composed-test-bundle",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": components,
                "companions": {"demucs": {"kind": "runtime", "version": ">=4.0"}},
                "profiles": {
                    "guitar-composed": {
                        "capability": "guitar.composed/v1",
                        "instruments": ["guitar"],
                        "required_components": list(components),
                        "required_companions": ["demucs"],
                        "difficulty_policies": ["expert_only"],
                        "graph": {
                            "stages": [
                                {
                                    "id": "separate",
                                    "kind": "audio_separation",
                                    "required": True,
                                    "component_ids": ["separation.demucs"],
                                    "companion_ids": ["demucs"],
                                    "depends_on": [],
                                    "inputs": ["source.audio.mix"],
                                    "outputs": ["artifact.stem.guitar"],
                                },
                                {
                                    "id": "detect",
                                    "kind": "onset_detection",
                                    "instrument": "guitar",
                                    "required": True,
                                    "component_ids": ["guitar.onset"],
                                    "companion_ids": [],
                                    "depends_on": ["separate"],
                                    "inputs": ["artifact.stem.guitar"],
                                    "outputs": ["artifact.guitar.onsets"],
                                    "difficulty": "Expert",
                                },
                                {
                                    "id": "map",
                                    "kind": "fret_mapping",
                                    "instrument": "guitar",
                                    "required": True,
                                    "component_ids": ["guitar.mapper"],
                                    "companion_ids": [],
                                    "depends_on": ["detect"],
                                    "inputs": ["artifact.guitar.onsets"],
                                    "outputs": ["artifact.guitar.events"],
                                    "difficulty": "Expert",
                                },
                                {
                                    "id": "assemble",
                                    "kind": "chart_assembly",
                                    "instrument": "guitar",
                                    "required": True,
                                    "component_ids": ["guitar.assembly"],
                                    "companion_ids": [],
                                    "depends_on": ["map"],
                                    "inputs": ["artifact.guitar.events"],
                                    "outputs": ["chart.guitar.expert"],
                                    "difficulty": "Expert",
                                },
                            ],
                            "outputs": [
                                {
                                    "instrument": "guitar",
                                    "stage_id": "assemble",
                                    "artifact_id": "chart.guitar.expert",
                                    "difficulty": "Expert",
                                }
                            ],
                        },
                    }
                },
            }
        )
    )
    return root


def _discovery_bundle(
    root: Path,
    *,
    model_id: str,
    profile_id: str,
    capability: str,
    instrument: str,
) -> Path:
    checkpoint = root / "weights" / "model.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{model_id}-weights".encode())
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": model_id,
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    f"{instrument}.onset": {
                        "checkpoint": "weights/model.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "byte_length": checkpoint.stat().st_size,
                    }
                },
                "profiles": {
                    profile_id: {
                        "capability": capability,
                        "instruments": [instrument],
                        "required_components": [f"{instrument}.onset"],
                        "difficulty_policies": ["expert_only"],
                    }
                },
            }
        )
    )
    return root


def test_probe_declares_versioned_runtime_and_available_pipelines() -> None:
    payload = _runtime_payload()

    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["python_requires"] == ">=3.11"
    assert "guitar.onset-fret/v1" in payload["pipelines"]
    assert "dataset_prepare" in payload["capabilities"]
    assert "chart_run" in payload["capabilities"]
    assert "typed_chart_results" in payload["capabilities"]
    assert isinstance(payload["optional_dependencies"]["basic_pitch"]["available"], bool)
    assert "model_bundle_preflight" in payload["capabilities"]
    assert "checkpoint_discovery" in payload["capabilities"]
    assert payload["chart_result_formats"] == [
        "strum-chart-preflight/v1",
        "strum-chart-run/v1",
    ]


def test_runtime_revision_ignores_untracked_non_source_files(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def check_output(command: list[str], **_kwargs: object) -> str:
        calls.append(command)
        return "" if "--untracked-files=no" in command else "0123456789abcdef\n"

    monkeypatch.delenv("STRUM_SOURCE_REVISION", raising=False)
    monkeypatch.setattr("src.worker.subprocess.check_output", check_output)

    assert _revision() == ("0123456789abcdef", False)
    assert calls[1][-2:] == ["--porcelain", "--untracked-files=no"]


def test_chart_transform_schema_exposes_opaque_parent_artifact_selection() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "chart_transform.five_lane/v1")
    assert descriptor.train_schema is not None
    properties = descriptor.train_schema["properties"]
    assert properties["checkpoint_mode"] == {
        "type": "string",
        "enum": ["fresh", "fine_tune"],
        "default": "fresh",
    }
    assert properties["parent_artifact_id"] == {
        "type": "string",
        "format": "strum-model-bundle-artifact-id",
    }
    assert "parent_bundle" not in properties
    assert descriptor.private_request_fields == ("catalog_root", "parent_bundle")
    assert descriptor.prepare_schema["properties"]["audio_feature_mode"]["enum"] == [
        "none",
        "rms_onset_v1",
    ]
    assert "audio_feature_mode" not in properties
    assert "audio_manifest" not in properties


def test_pipeline_descriptors_advertise_safe_host_orchestration_requirements() -> None:
    guitar = next(item for item in PIPELINES if item.id == "guitar.onset-fret/v1")
    assert guitar.private_request_fields == ("catalog_root",)
    assert guitar.training_requirements == ("profile_evaluation", "profile_packaging")
    assert guitar.catalog_inspection_option_keys == (
        "audio_role",
        "fallback_audio_role",
        "required_difficulty",
    )
    rendered = guitar.as_json()
    assert rendered["private_request_fields"] == ["catalog_root"]
    assert rendered["catalog_inspection_option_keys"] == [
        "audio_role",
        "fallback_audio_role",
        "required_difficulty",
    ]
    for pipeline_id in (
        "bass.onset-fret/v1",
        "keys.onset-fret/v1",
        "vocals.note-activity/v1",
        "drums.onset-classifier/v1",
    ):
        descriptor = next(item for item in PIPELINES if item.id == pipeline_id)
        assert descriptor.private_request_fields == ("catalog_root",)
        assert descriptor.catalog_inspection_option_keys == (
            "audio_role",
            "fallback_audio_role",
            "required_difficulty",
        )
    assert next(
        item for item in PIPELINES if item.id == "bass.onset-fret/v1"
    ).training_requirements == (
        "bass_profile_evaluation",
        "bass_profile_packaging",
    )
    assert next(
        item for item in PIPELINES if item.id == "keys.onset-fret/v1"
    ).training_requirements == (
        "keys_profile_evaluation",
        "keys_profile_packaging",
    )
    for pipeline_id in ("strum.fret-mapper/guitar/v1", "strum.fret-mapper/bass/v1"):
        descriptor = next(item for item in PIPELINES if item.id == pipeline_id)
        assert descriptor.private_request_fields == ("catalog_root",)
        assert "strum_pitch_extra" in descriptor.training_requirements


@pytest.mark.parametrize(
    ("pipeline_id", "task_kind", "schema_id", "tracks", "required_stages"),
    [
        (
            "strum.instrument-chart/pro-guitar/v1",
            "pro_guitar",
            "pro-string-fret-midi/v1",
            ["PART REAL_GUITAR", "PART REAL_GUITAR_22"],
            {"pro_guitar_audio_preprocessor/v1", "pro_guitar_chart_execution/v1"},
        ),
        (
            "strum.instrument-chart/pro-bass/v1",
            "pro_bass",
            "pro-string-fret-midi/v1",
            ["PART REAL_BASS", "PART REAL_BASS_22"],
            {"pro_bass_audio_preprocessor/v1", "pro_bass_chart_execution/v1"},
        ),
        (
            "strum.instrument-chart/pro-keys/v1",
            "pro_keys",
            "pro-keys-pitch-midi/v1",
            ["PART REAL_KEYS_X"],
            {"pro_keys_audio_preprocessor/v1", "pro_keys_chart_execution/v1"},
        ),
    ],
)
def test_pro_descriptors_publish_non_executable_real_midi_training_contracts(
    pipeline_id: str,
    task_kind: str,
    schema_id: str,
    tracks: list[str],
    required_stages: set[str],
) -> None:
    descriptor = next(item for item in PIPELINES if item.id == pipeline_id)

    assert descriptor.training_status == "planned"
    assert descriptor.train_schema is None
    assert descriptor.inference_capability is None
    assert descriptor.catalog_requirements["label_schema"] == schema_id
    assert descriptor.catalog_requirements["label_tracks"] == tracks
    assert (
        descriptor.catalog_requirements["prepared_task_view_format"]
        == "strum-pro-target-task-manifest/v1"
    )
    assert (
        descriptor.catalog_requirements["prepared_target_encoding"]
        == "strum-pro-midi-target-decoder/v1"
    )
    contract = descriptor.as_json()["training_contract"]
    assert contract == {
        **contract,
        "format": "strum-planned-training-contract/v1",
        "training_status": "planned",
        "execution": {"status": "not_available", "inference_capability": None},
    }
    assert contract["label_source"]["schema_id"] == schema_id
    assert contract["label_source"]["tracks"] == tracks
    assert contract["prepared_target_encoding"] == "strum-pro-midi-target-decoder/v1"
    assert set(contract["required_stages"]) == set(descriptor.training_requirements)
    assert required_stages <= set(contract["required_stages"])
    assert task_kind in pipeline_id.replace("-", "_")


def test_legacy_inference_output_is_not_exposed_to_worker_clients(
    capfd: pytest.CaptureFixture[str],
) -> None:
    secret_path = "/private/catalog/audio.opus"

    def legacy_inference() -> str:
        print(f"Predicting MIDI for {secret_path}")
        os.write(2, secret_path.encode())
        return "chart"

    assert _run_without_legacy_output(legacy_inference) == "chart"
    captured = capfd.readouterr()
    assert secret_path not in captured.out
    assert secret_path not in captured.err


def _catalog_asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    path = root / "assets" / "sha256" / digest / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": path.relative_to(root).as_posix(),
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _guitar_catalog(root: Path) -> None:
    source_id = "octave-src-aaaaaaaa"
    record = {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {"training_use": "allowed", "provenance": "Reviewed", "license": "test-only"},
        "metadata": {"name": "Fixture"},
        "chart": {
            "notes_midi": _catalog_asset(root, b"midi", "notes.mid"),
            "instruments": {
                "guitar": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART GUITAR"],
                }
            },
        },
        "audio": {"guitar": _catalog_asset(root, b"audio", "guitar.ogg")},
    }
    (root / "records.jsonl").write_text(json.dumps(record) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "worker-fixture",
                "records": "records.jsonl",
            }
        )
    )


def _catalog_record(
    root: Path,
    suffix: str,
    *,
    training_use: str = "allowed",
    instruments: dict[str, list[str]],
    audio_roles: tuple[str, ...] = (),
) -> dict[str, object]:
    source_id = f"octave-src-{suffix}"
    return {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {
            "training_use": training_use,
            "provenance": "Reviewed private collection",
            "license": "test-only",
        },
        "metadata": {"name": "Fixture"},
        "chart": {
            "notes_midi": _catalog_asset(root, f"midi-{suffix}".encode(), "notes.mid"),
            "instruments": {
                instrument: {
                    "status": "present",
                    "difficulties": difficulties,
                    "track_names": [f"PART {instrument.upper()}"],
                }
                for instrument, difficulties in instruments.items()
            },
        },
        "audio": {
            role: _catalog_asset(root, f"audio-{role}-{suffix}".encode(), f"{role}.ogg")
            for role in audio_roles
        },
    }


def _write_catalog(root: Path, records: list[dict[str, object]]) -> None:
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "worker-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_catalog_inspect_and_prepare_emit_path_free_task_view(tmp_path: Path) -> None:
    _guitar_catalog(tmp_path)
    inspection = inspect_catalog(tmp_path, "guitar.onset-fret/v1")
    assert inspection == {
        "status": "ready",
        "catalog_id": "worker-fixture",
        "record_count": 1,
        "allowed_record_count": 1,
        "pipeline_id": "guitar.onset-fret/v1",
        "eligible_count": 1,
        "exclusion_reason_counts": {
            "training_use_not_allowed": 0,
            "instrument_not_present": 0,
            "required_difficulty_missing": 0,
            "audio_unavailable": 0,
        },
        "audio_policy": {
            "kind": "preferred_with_fallback",
            "preferred_role": "guitar",
            "fallback_role": "mix",
            "required": True,
        },
        "estimated_storage_bytes": 9,
        "storage_estimate_capped": False,
        "storage_estimate_semantics": (
            "sum of distinct catalog input assets selected by the declared policy; "
            "excludes generated task views, preprocessing caches, checkpoints, and "
            "existing catalog storage"
        ),
    }
    request_path = tmp_path / "request.json"
    output = tmp_path / "views" / "guitar.json"
    request_path.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "guitar.onset-fret/v1",
                "output": str(output),
                "options": {"required_difficulty": "expert"},
            }
        )
    )

    result = prepare_dataset_request(request_path)

    assert result["status"] == "prepared"
    assert result["output_name"] == "guitar.json"
    assert result["record_count"] == 1
    assert str(tmp_path) not in output.read_text()


def test_catalog_inspect_is_pipeline_specific_and_path_free(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _catalog_record(
                tmp_path,
                "aaaaaaaa",
                instruments={"guitar": ["expert", "hard"]},
                audio_roles=("guitar",),
            ),
            _catalog_record(
                tmp_path,
                "bbbbbbbb",
                instruments={"guitar": ["expert", "hard"]},
            ),
            _catalog_record(
                tmp_path,
                "cccccccc",
                instruments={"drums": ["expert"]},
                audio_roles=("drums",),
            ),
            _catalog_record(
                tmp_path,
                "dddddddd",
                instruments={"guitar": ["expert"]},
            ),
            _catalog_record(
                tmp_path,
                "eeeeeeee",
                training_use="review_required",
                instruments={"guitar": ["expert", "hard"]},
                audio_roles=("guitar",),
            ),
        ],
    )

    guitar = inspect_catalog(tmp_path, "guitar.onset-fret/v1")
    assert guitar["eligible_count"] == 1
    assert guitar["exclusion_reason_counts"] == {
        "training_use_not_allowed": 1,
        "instrument_not_present": 1,
        "required_difficulty_missing": 0,
        "audio_unavailable": 2,
    }
    assert guitar["audio_policy"] == {
        "kind": "preferred_with_fallback",
        "preferred_role": "guitar",
        "fallback_role": "mix",
        "required": True,
    }
    assert isinstance(guitar["estimated_storage_bytes"], int)
    assert guitar["estimated_storage_bytes"] > 0

    drums = inspect_catalog(tmp_path, "drums.onset-classifier/v1")
    assert drums["eligible_count"] == 1
    assert drums["exclusion_reason_counts"] == {
        "training_use_not_allowed": 1,
        "instrument_not_present": 3,
        "required_difficulty_missing": 0,
        "audio_unavailable": 0,
    }
    assert drums["audio_policy"]["preferred_role"] == "drums"

    transform = inspect_catalog(
        tmp_path,
        "chart_transform.five_lane/v1",
        options={"instrument": "guitar", "target_difficulty": "Hard"},
    )
    assert transform["eligible_count"] == 2
    assert transform["exclusion_reason_counts"] == {
        "training_use_not_allowed": 1,
        "instrument_not_present": 1,
        "source_difficulty_missing": 0,
        "target_difficulty_missing": 1,
        "audio_unavailable": 0,
    }
    assert transform["audio_policy"] == {"kind": "not_required", "required": False}
    assert transform["eligibility_selection"] == {
        "mode": "requested_prepare_options",
        "instrument": "guitar",
        "target_difficulty": "Hard",
    }

    rendered = json.dumps({"guitar": guitar, "drums": drums, "transform": transform})
    assert str(tmp_path) not in rendered
    assert "Reviewed private collection" not in rendered


def test_catalog_inspect_has_a_uniform_planning_summary_for_every_pipeline(tmp_path: Path) -> None:
    _guitar_catalog(tmp_path)
    required_keys = {
        "status",
        "catalog_id",
        "record_count",
        "allowed_record_count",
        "pipeline_id",
        "eligible_count",
        "exclusion_reason_counts",
        "audio_policy",
        "estimated_storage_bytes",
        "storage_estimate_capped",
        "storage_estimate_semantics",
    }
    for descriptor in PIPELINES:
        summary = inspect_catalog(tmp_path, descriptor.id)
        assert required_keys <= set(summary)
        assert summary["pipeline_id"] == descriptor.id
        assert isinstance(summary["eligible_count"], int)
        assert isinstance(summary["exclusion_reason_counts"], dict)
        assert isinstance(summary["audio_policy"], dict)
        assert isinstance(summary["estimated_storage_bytes"], int)


def test_catalog_inspect_uses_the_derived_task_adapter_fallback_policy(tmp_path: Path) -> None:
    _write_catalog(
        tmp_path,
        [
            _catalog_record(
                tmp_path,
                "ffffffff",
                instruments={"bass": ["expert"]},
                audio_roles=("mix",),
            )
        ],
    )
    pipeline_id = "strum.instrument-chart/bass/v1"
    inspection = inspect_catalog(tmp_path, pipeline_id)
    assert inspection["eligible_count"] == 1
    assert inspection["audio_policy"] == {
        "kind": "preferred_with_fallback",
        "preferred_role": "bass",
        "fallback_role": "mix",
        "required": True,
    }

    request_path = tmp_path / "prepare.json"
    request_path.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": pipeline_id,
                "output": str(tmp_path / "views" / "bass.json"),
                "options": {},
            }
        )
    )
    assert prepare_dataset_request(request_path)["record_count"] == inspection["eligible_count"]


def test_drums_pipeline_exposes_a_strict_worker_training_schema() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "drums.onset-classifier/v1")

    assert descriptor.training_status == "available"
    assert descriptor.checkpoint_outputs == ("drums_onset_classifier",)
    assert descriptor.inference_capability is None
    assert descriptor.train_schema is not None
    assert "catalog_root" not in descriptor.train_schema["properties"]
    assert descriptor.train_schema == {
        "type": "object",
        "additionalProperties": False,
        "properties": {
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
        "required": ["model_id"],
    }


def test_drums_training_request_routes_catalog_task_view_to_existing_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    task_view = tmp_path / "drums-task.json"
    task_view.write_text("{}")
    request = tmp_path / "train.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "drums.onset-classifier/v1",
                "task_view": str(task_view),
                "output": str(tmp_path / "experiment"),
                "catalog_root": str(tmp_path / "catalog"),
                "options": {
                    "model_id": "curated-drums-v1",
                    "epochs": 1,
                    "max_train_batches": 1,
                    "max_test_batches": 1,
                },
            }
        )
    )
    observed: dict[str, object] = {}

    def fake_train(
        received_task_view: str | Path,
        output_dir: str | Path,
        options: object,
        *,
        catalog_root: str | Path,
    ) -> dict[str, object]:
        print(f"legacy trainer input={received_task_view}")
        observed["task_view"] = received_task_view
        observed["output_dir"] = output_dir
        observed["options"] = options
        observed["catalog_root"] = catalog_root
        return {
            "status": "completed",
            "pipeline_id": "drums.onset-classifier/v1",
            "model_id": "curated-drums-v1",
            "experiment_name": "experiment.json",
            "task_view_sha256": "a" * 64,
            "checkpoint": {"name": "checkpoints/best_f1.pt"},
            "metrics": {"overall_f1": 0.5},
        }

    monkeypatch.setattr("src.drums_onset_training.run_drums_onset_training", fake_train)

    result = run_training_request(request)

    assert result["status"] == "completed"
    assert observed["task_view"] == str(task_view)
    assert observed["output_dir"] == str(tmp_path / "experiment")
    assert observed["options"] == json.loads(request.read_text())["options"]
    assert observed["catalog_root"] == str(tmp_path / "catalog")
    captured = capfd.readouterr()
    assert str(task_view) not in captured.out


def _chart_transform_task_view(root: Path, *, target_difficulty: str = "Hard") -> Path:
    records = [
        {
            "song_id": "song-a",
            "instrument": "guitar",
            "source_difficulty": "Expert",
            "target_difficulty": target_difficulty,
            "source_events": [{"time_ms": 0, "lanes": [0]}],
            "target_events": [{"time_ms": 0, "lanes": [0]}],
        },
        {
            "song_id": "song-b",
            "instrument": "guitar",
            "source_difficulty": "Expert",
            "target_difficulty": target_difficulty,
            "source_events": [{"time_ms": 0, "lanes": [1]}],
            "target_events": [{"time_ms": 0, "lanes": [1]}],
        },
    ]
    (root / "pairs.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    task_view = root / "dataset-manifest.json"
    task_view.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-chart-pairs/v1",
                "dataset_id": "worker-chart-transform-fixture",
                "records": "pairs.jsonl",
                "provenance": "synthetic worker fixture",
                "license": "test-only",
                "instrument": "guitar",
                "source_difficulty": "Expert",
                "target_difficulty": target_difficulty,
            }
        )
    )
    return task_view


def _chart_transform_train_request(
    root: Path,
    *,
    task_view: Path,
    output: Path,
    model_id: str,
    checkpoint_mode: str = "fresh",
    parent_bundle: Path | None = None,
    hidden_dim: int = 4,
) -> Path:
    options: dict[str, object] = {
        "model_id": model_id,
        "checkpoint_mode": checkpoint_mode,
        "validation_fraction": 0.5,
        "hidden_dim": hidden_dim,
        "epochs": 1,
        "device": "cpu",
    }
    if parent_bundle is not None:
        options["parent_artifact_id"] = f"artifact:{parent_bundle.name}"
    request = root / f"{model_id}-request.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "chart_transform.five_lane/v1",
                "task_view": str(task_view),
                "output": str(output),
                "options": options,
                **({"parent_bundle": str(parent_bundle)} if parent_bundle is not None else {}),
            }
        )
    )
    return request


def test_chart_transform_worker_fine_tune_requires_a_verified_compatible_parent(
    tmp_path: Path,
) -> None:
    task_view = _chart_transform_task_view(tmp_path)
    parent_root = tmp_path / "parent"
    fresh = _chart_transform_train_request(
        tmp_path,
        task_view=task_view,
        output=parent_root,
        model_id="parent-transform",
    )
    assert run_training_request(fresh)["status"] == "completed"

    child_root = tmp_path / "child"
    fine_tune = _chart_transform_train_request(
        tmp_path,
        task_view=task_view,
        output=child_root,
        model_id="child-transform",
        checkpoint_mode="fine_tune",
        parent_bundle=parent_root,
    )

    result = run_training_request(fine_tune)

    experiment = json.loads((child_root / "experiment.json").read_text())
    metadata = json.loads((child_root / "training-metadata.json").read_text())
    assert result["status"] == "completed"
    assert experiment["checkpoint_mode"] == "fine_tune"
    assert experiment["parent"]["model_id"] == "parent-transform"
    assert experiment["parent"]["component"] == "chart_transform.guitar.expert_to_hard"
    assert metadata["initialization"]["parent"] == experiment["parent"]
    assert str(parent_root) not in json.dumps(experiment)
    assert str(parent_root) not in json.dumps(metadata)

    incompatible = _chart_transform_train_request(
        tmp_path,
        task_view=task_view,
        output=tmp_path / "incompatible",
        model_id="incompatible-transform",
        checkpoint_mode="fine_tune",
        parent_bundle=parent_root,
        hidden_dim=8,
    )
    with pytest.raises(ValueError, match="hidden_dim differs"):
        run_training_request(incompatible)


def test_chart_transform_worker_rejects_resume_and_unverified_parent_paths(tmp_path: Path) -> None:
    task_view = _chart_transform_task_view(tmp_path)
    resume = _chart_transform_train_request(
        tmp_path,
        task_view=task_view,
        output=tmp_path / "resume",
        model_id="resume-transform",
        checkpoint_mode="resume",
    )
    with pytest.raises(ValueError, match="resume is not supported"):
        run_training_request(resume)

    arbitrary_checkpoint = tmp_path / "arbitrary.pt"
    arbitrary_checkpoint.write_bytes(b"not a bundle")
    fine_tune = _chart_transform_train_request(
        tmp_path,
        task_view=task_view,
        output=tmp_path / "unverified",
        model_id="unverified-transform",
        checkpoint_mode="fine_tune",
        parent_bundle=arbitrary_checkpoint,
    )
    with pytest.raises(ValueError, match="parent bundle failed verification") as error:
        run_training_request(fine_tune)
    assert str(arbitrary_checkpoint) not in str(error.value)


def test_preflight_requires_hash_and_length_for_deployable_components(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})

    result = preflight_bundle(root, required_components=["guitar.onset"])

    assert result["status"] == "ready"
    assert result["components"][0]["id"] == "guitar.onset"
    assert result["manifest_sha256"]


def test_preflight_requires_config_fingerprint_when_component_declares_config(
    tmp_path: Path,
) -> None:
    root = _bundle(
        tmp_path,
        {"architecture": "GuitarOnsetCRNN/v1", "config": "configs/guitar.json"},
    )
    config = root / "configs" / "guitar.json"
    config.parent.mkdir()
    config.write_text("{}")

    with pytest.raises(BundleValidationError, match="config requires sha256"):
        preflight_bundle(root, required_components=["guitar.onset"])


def test_preflight_rejects_incomplete_or_missing_components(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {})
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    del manifest["components"]["guitar.onset"]["byte_length"]
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest))

    with pytest.raises(BundleValidationError, match="byte_length"):
        preflight_bundle(root, required_components=["guitar.onset"])
    with pytest.raises(BundleValidationError, match="not declared"):
        preflight_bundle(root, required_components=["guitar.fret"])


def test_profile_validation_requires_declared_companions_and_difficulty_policy(
    tmp_path: Path,
) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"] = {
        "guitar-default": {
            "capability": "guitar.audio_to_chart/v1",
            "instruments": ["guitar"],
            "required_components": ["guitar.onset"],
            "difficulty_policies": ["expert_only", "deterministic-v1"],
        }
    }
    manifest_path.write_text(json.dumps(manifest))

    result = validate_inference_profile(
        root, profile_id="guitar-default", difficulty_policy="expert_only"
    )

    assert result["capability"] == "guitar.audio_to_chart/v1"
    assert result["instruments"] == ["guitar"]
    with pytest.raises(BundleValidationError, match="does not support"):
        validate_inference_profile(
            root, profile_id="guitar-default", difficulty_policy="learned:bad"
        )


def test_checkpoint_discovery_returns_path_free_dynamic_profile_candidates(tmp_path: Path) -> None:
    model_root = tmp_path / "user-selected-checkpoints"
    _discovery_bundle(
        model_root / "guitar",
        model_id="guitar-candidate",
        profile_id="guitar-profile",
        capability="guitar.neural-v1-expert/v1",
        instrument="guitar",
    )
    _discovery_bundle(
        model_root / "bass",
        model_id="bass-candidate",
        profile_id="bass-profile",
        capability="bass.neural-v1-expert/v1",
        instrument="bass",
    )
    _discovery_bundle(
        model_root / "nested" / "keys",
        model_id="keys-candidate",
        profile_id="keys-profile",
        capability="keys.neural-v1-expert/v1",
        instrument="keys",
    )
    _discovery_bundle(
        model_root / "drums",
        model_id="drums-candidate",
        profile_id="drums-profile",
        capability="drums.v14-expert/v1",
        instrument="drums",
    )
    invalid = model_root / "invalid"
    invalid.mkdir(parents=True)
    (invalid / MANIFEST_FILENAME).write_text("not a model bundle")

    result = discover_model_bundles(model_root)

    assert result["format"] == "strum-model-bundle-discovery/v1"
    assert result["candidate_count"] == 4
    assert result["profile_count"] == 4
    assert result["rejected_bundle_count"] == 1
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    assert {
        (candidate["model_id"], candidate["profiles"][0]["instruments"][0])
        for candidate in candidates
    } == {
        ("guitar-candidate", "guitar"),
        ("bass-candidate", "bass"),
        ("keys-candidate", "keys"),
        ("drums-candidate", "drums"),
    }
    assert all(
        candidate["artifact_id"].startswith("strum-model-bundle/") for candidate in candidates
    )
    assert all(candidate["deployment_status"] == "not_deployable" for candidate in candidates)
    assert str(model_root) not in json.dumps(result)


def test_checkpoint_inspection_requires_hashes_and_marks_only_typed_profile_executable(
    tmp_path: Path,
) -> None:
    root = tmp_path / "drums"
    checkpoint = root / "weights" / "v14.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified V14 checkpoint")
    component_config = root / "configs" / "drums-v14.yaml"
    component_config.parent.mkdir()
    component_config.write_text("model: drums-v14\n")
    profile_config = root / "profiles" / "drums-v14.json"
    profile_config.parent.mkdir()
    profile_config.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "strum-drums-v14-expert-profile/v1",
                "model_architecture": "TwoStageDrumsCRNN/v14",
                "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                "segment_duration_seconds": 10,
                "overlap": 0.5,
                "onset_threshold": 0.4,
                "class_thresholds": [0.3, 0.25, 0.35, 0.12, 0.28, 0.12, 0.35, 0.12],
                "min_distance_ms": 20,
                "postprocess": "none",
                "class_to_midi": [96, 97, 98, 98, 99, 99, 100, 100],
                "model_parameters": {
                    "n_mels": 128,
                    "conv_channels": [64, 128, 256, 512],
                    "freq_subbands": [32, 64, 96, 128],
                    "subband_proj_dim": 256,
                    "lstm_hidden": 640,
                    "lstm_layers": 3,
                    "attention_heads": 10,
                    "attention_type": "flash",
                    "attention_window": 512,
                    "dropout": 0.0,
                    "onset_detector_hidden": 320,
                    "classifier_hidden": 640,
                    "num_classes": 8,
                    "predict_velocity": True,
                },
            }
        )
    )
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "drums-v14-candidate",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    "drums.v14": {
                        "checkpoint": "weights/v14.pt",
                        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        "byte_length": checkpoint.stat().st_size,
                        "config": "configs/drums-v14.yaml",
                        "config_sha256": hashlib.sha256(component_config.read_bytes()).hexdigest(),
                        "config_byte_length": component_config.stat().st_size,
                        "architecture": "TwoStageDrumsCRNN/v14",
                        "preprocessing": "drums-logmel-44100-2048-512-128-v1",
                    }
                },
                "profiles": {
                    "drums-v14-expert": {
                        "capability": "drums.v14-expert/v1",
                        "instruments": ["drums"],
                        "required_components": ["drums.v14"],
                        "difficulty_policies": ["expert_only"],
                        "configuration": "profiles/drums-v14.json",
                        "configuration_sha256": hashlib.sha256(
                            profile_config.read_bytes()
                        ).hexdigest(),
                        "configuration_byte_length": profile_config.stat().st_size,
                    }
                },
            }
        )
    )

    result = inspect_model_bundle(root)

    assert result["format"] == "strum-model-bundle-inspection/v1"
    assert result["deployment_status"] == "ready"
    assert result["profiles"] == [
        {
            "profile_id": "drums-v14-expert",
            "capability": "drums.v14-expert/v1",
            "instruments": ["drums"],
            "difficulty_policies": ["expert_only"],
            "required_components": ["drums.v14"],
            "required_companions": [],
            "profile_configuration_sha256": hashlib.sha256(profile_config.read_bytes()).hexdigest(),
            "profile_configuration_byte_length": profile_config.stat().st_size,
            "execution": {"status": "available", "difficulty_policies": ["expert_only"]},
        }
    ]
    assert str(root) not in json.dumps(result)


def test_chart_preflight_returns_an_explicit_non_execution_plan(tmp_path: Path) -> None:
    root = _bundle(tmp_path, {"architecture": "GuitarOnsetCRNN/v1"})
    manifest_path = root / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["profiles"] = {
        "guitar-default": {
            "capability": "guitar.audio_to_chart/v1",
            "instruments": ["guitar"],
            "required_components": ["guitar.onset"],
            "difficulty_policies": ["expert_only"],
        }
    }
    manifest_path.write_text(json.dumps(manifest))
    request = tmp_path / "chart-request.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(root),
                "profile_id": "guitar-default",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
                "device": "cuda",
            }
        )
    )

    plan = preflight_chart_request(request)

    assert plan["status"] == "ready"
    assert plan["format"] == "strum-chart-preflight/v1"
    assert plan["execution"] == "not_available"
    assert plan["components"][0]["id"] == "guitar.onset"
    assert plan["instrument_results"] == {
        "guitar": {
            "status": "not_available",
            "stages": {
                "expert_chart": {
                    "status": "unavailable",
                    "required": True,
                    "component_ids": ["guitar.onset"],
                    "difficulty": "Expert",
                    "reason": "execution_handler_not_declared",
                },
                "difficulty_transform": {
                    "status": "not_requested",
                    "required": False,
                    "component_ids": [],
                    "difficulty": "Expert",
                    "reason": "difficulty_policy_expert_only",
                },
            },
        }
    }
    assert plan["difficulty"] == {
        "policy": "expert_only",
        "status": "expert_only",
        "source_difficulty": None,
        "target_difficulty": "Expert",
    }


def test_composed_profile_preflight_resolves_every_stage_without_claiming_execution(
    tmp_path: Path,
) -> None:
    root = _composed_profile_bundle(tmp_path)
    request = tmp_path / "chart-request.json"
    request.write_text(
        json.dumps(
            {
                "model_root": str(root),
                "profile_id": "guitar-composed",
                "difficulty_policy": "expert_only",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )

    validation = validate_inference_profile(
        root, profile_id="guitar-composed", difficulty_policy="expert_only"
    )
    plan = preflight_chart_request(request)

    assert validation["required_companions"] == [
        {"id": "demucs", "kind": "runtime", "version": ">=4.0"}
    ]
    assert validation["composition"]["format"] == "strum-profile-composition/v1"
    assert plan["status"] == "ready"
    assert plan["execution"] == "not_available"
    assert plan["composition"]["required_components"] == [
        "separation.demucs",
        "guitar.onset",
        "guitar.mapper",
        "guitar.assembly",
    ]
    assert plan["composition"]["required_companions"] == [
        {"id": "demucs", "kind": "runtime", "version": ">=4.0"}
    ]
    assert [stage["status"] for stage in plan["composition"]["stages"]] == [
        "unavailable",
        "unavailable",
        "unavailable",
        "unavailable",
    ]
    assert plan["composition"]["outputs"] == [
        {
            "instrument": "guitar",
            "stage_id": "assemble",
            "artifact_id": "chart.guitar.expert",
            "difficulty": "Expert",
            "status": "unavailable",
        }
    ]
    assert plan["instrument_results"] == {
        "guitar": {
            "status": "not_available",
            "stages": {
                "detect": {
                    "status": "unavailable",
                    "required": True,
                    "component_ids": ["guitar.onset"],
                    "companion_ids": [],
                    "depends_on": ["separate"],
                    "difficulty": "Expert",
                    "reason": "execution_handler_not_declared",
                },
                "map": {
                    "status": "unavailable",
                    "required": True,
                    "component_ids": ["guitar.mapper"],
                    "companion_ids": [],
                    "depends_on": ["detect"],
                    "difficulty": "Expert",
                    "reason": "execution_handler_not_declared",
                },
                "assemble": {
                    "status": "unavailable",
                    "required": True,
                    "component_ids": ["guitar.assembly"],
                    "companion_ids": [],
                    "depends_on": ["map"],
                    "difficulty": "Expert",
                    "reason": "execution_handler_not_declared",
                },
            },
        }
    }
    assert str(tmp_path) not in json.dumps(plan)

    run_request = tmp_path / "chart-run.json"
    run_request.write_text(
        json.dumps(
            {
                "preflight_request": str(request),
                "source_midi_path": str(tmp_path / "source.mid"),
                "song_path": None,
                "output_dir": str(tmp_path / "output"),
                "threshold": 0.5,
            }
        )
    )
    with pytest.raises(WorkerRequestError, match="no worker chart execution handler"):
        run_chart_request(run_request)


def test_checkpoint_inspection_discovers_composed_profile_without_locations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = _composed_profile_bundle(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["strum-worker", "checkpoint", "inspect", "--model-root", str(root), "--json"],
    )

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert len(payload["profiles"]) == 1
    profile = payload["profiles"][0]
    assert profile["profile_id"] == "guitar-composed"
    assert profile["required_companions"] == [
        {"id": "demucs", "kind": "runtime", "version": ">=4.0"}
    ]
    assert profile["composition"]["format"] == "strum-profile-composition/v1"
    assert [stage["id"] for stage in profile["composition"]["stages"]] == [
        "separate",
        "detect",
        "map",
        "assemble",
    ]
    assert profile["composition"]["outputs"] == [
        {
            "instrument": "guitar",
            "stage_id": "assemble",
            "artifact_id": "chart.guitar.expert",
            "difficulty": "Expert",
        }
    ]
    assert str(tmp_path) not in json.dumps(payload)


@pytest.mark.parametrize(
    ("capability", "instrument", "component_id"),
    [
        ("guitar.hybrid-v2-rule/v1", "guitar", "guitar.onset"),
        ("drums.v14-expert/v1", "drums", "drums.onset_classifier"),
    ],
)
def test_direct_chart_profiles_declare_the_omitted_difficulty_stage(
    capability: str, instrument: str, component_id: str
) -> None:
    instrument_results, difficulty = _chart_result_contract(
        {
            "capability": capability,
            "difficulty_policy": "expert_only",
            "instruments": [instrument],
            "components": [{"id": component_id}],
        },
        execution="available",
    )

    assert instrument_results[instrument] == {
        "status": "ready",
        "stages": {
            "expert_chart": {
                "status": "ready",
                "required": True,
                "component_ids": [component_id],
                "difficulty": "Expert",
            },
            "difficulty_transform": {
                "status": "not_requested",
                "required": False,
                "component_ids": [],
                "difficulty": "Expert",
                "reason": "difficulty_policy_expert_only",
            },
        },
    }
    assert difficulty == {
        "policy": "expert_only",
        "status": "expert_only",
        "source_difficulty": None,
        "target_difficulty": "Expert",
    }


def test_chart_transform_profile_runs_from_expert_midi_without_path_leaks(tmp_path: Path) -> None:
    root = tmp_path / "transform-bundle"
    checkpoint_path = root / "weights" / "chart_transform.pt"
    checkpoint_path.parent.mkdir(parents=True)
    model = EventTransformMLP(lane_count=5, hidden_dim=4, audio_feature_dim=0)
    torch.save(
        {
            "model_type": "EventTransformMLP",
            "lane_count": 5,
            "hidden_dim": 4,
            "audio_feature_dim": 0,
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    config_path = root / "configs" / "training-config.json"
    config_path.parent.mkdir()
    config_path.write_text(json.dumps({"instrument": "guitar", "target_difficulty": "Hard"}))
    component_id = "chart_transform.guitar.expert_to_hard"
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_id": "transform-fixture",
                "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
                "components": {
                    component_id: {
                        "checkpoint": "weights/chart_transform.pt",
                        "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                        "byte_length": checkpoint_path.stat().st_size,
                        "config": "configs/training-config.json",
                        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                        "config_byte_length": config_path.stat().st_size,
                        "architecture": "EventTransformMLP/v1",
                        "preprocessing": "midi-five-lane-events/v1",
                    }
                },
                "profiles": {
                    "difficulty-transform-guitar": {
                        "capability": "difficulty.transform/v1",
                        "instruments": ["guitar"],
                        "required_components": [component_id],
                        "difficulty_policies": [f"learned:{component_id}"],
                    }
                },
            }
        )
    )
    source_midi = tmp_path / "expert.mid"
    source = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    source.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="PART GUITAR", time=0))
    track.append(mido.Message("note_on", note=96, velocity=100, time=0))
    track.append(mido.Message("note_off", note=96, velocity=0, time=120))
    source.save(source_midi)
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "model_root": str(root),
                "profile_id": "difficulty-transform-guitar",
                "difficulty_policy": f"learned:{component_id}",
                "instruments": ["guitar"],
                "device": "cpu",
            }
        )
    )
    output = tmp_path / "result"
    request = tmp_path / "run.json"
    request.write_text(
        json.dumps(
            {
                "preflight_request": str(preflight),
                "source_midi_path": str(source_midi),
                "song_path": None,
                "output_dir": str(output),
                "threshold": 0.01,
            }
        )
    )

    result = run_chart_request(request)

    assert result["status"] == "completed"
    assert result["format"] == "strum-chart-run/v1"
    assert result["difficulty"] == "Hard"
    assert result["chart_result"]["difficulty"] == {
        "policy": f"learned:{component_id}",
        "status": "succeeded",
        "source_difficulty": "Expert",
        "target_difficulty": "Hard",
    }
    manifest = json.loads((output / "run.json").read_text())
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["instrument_results"] == {
        "guitar": {
            "status": "succeeded",
            "stages": {
                "expert_chart": {
                    "status": "provided",
                    "required": True,
                    "component_ids": [],
                    "difficulty": "Expert",
                    "reason": "source_midi_required",
                },
                "difficulty_transform": {
                    "status": "succeeded",
                    "required": True,
                    "component_ids": [component_id],
                    "difficulty": "Hard",
                    "artifact_ids": ["events", "notes_midi"],
                },
            },
        }
    }
    assert manifest["difficulty"] == {
        "policy": f"learned:{component_id}",
        "status": "succeeded",
        "source_difficulty": "Expert",
        "target_difficulty": "Hard",
    }
    note_ons = {
        message.note
        for track in mido.MidiFile(output / "notes.mid").tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    assert note_ons <= {84, 85, 86, 87, 88}


def test_expert_guitar_export_never_materializes_lower_difficulties(tmp_path: Path) -> None:
    chart = SimpleNamespace(
        tempo_bpm=120.0,
        notes=[SimpleNamespace(time_ms=100.0, duration_ms=200.0, fret=2)],
        chords=[SimpleNamespace(time_ms=500.0, duration_ms=100.0, frets=[0, 4])],
    )
    output = tmp_path / "notes.mid"

    _write_expert_guitar_midi(chart, output)

    note_ons = {
        message.note
        for track in mido.MidiFile(output).tracks
        for message in track
        if message.type == "note_on" and message.velocity > 0
    }
    assert note_ons == {96, 98, 100}
