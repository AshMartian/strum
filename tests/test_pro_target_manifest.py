from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import sys
import wave
from pathlib import Path

import mido
import numpy as np
import pytest
import torch

from src.catalog_task_manifest import build_catalog_task_manifest
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError
from src.pro_audio_preprocessing import _tempo_segments, _tick_seconds, prepare_pro_audio_windows
from src.pro_candidate_contract import (
    FREE_RUNNING_PROPOSAL_CANDIDATE_KIND,
    KNOWN_EVENT_CANDIDATE_KIND,
    ProCandidateContractError,
    resolve_pro_candidate_contract,
    validate_pro_candidate_bundle,
)
from src.pro_event_proposal_preprocessing import (
    PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID,
    _negative_centers,
    canonical_pro_event_proposal_negative_policy,
    prepare_pro_event_proposal_windows,
    validate_pro_event_proposal_negative_policy,
)
from src.pro_event_proposal_training_options import ProEventProposalTrainingOptions
from src.pro_event_training_options import ProEventTrainingOptions
from src.pro_event_worker_training import _source_inputs
from src.pro_target_manifest import (
    PRO_AUDIO_PREPROCESSING_ID,
    PRO_TARGET_MANIFEST_FORMAT,
    build_catalog_pro_target_manifest,
    normalize_pro_audio_preprocessing,
    resolve_catalog_pro_target_manifest_songs,
)
from src.song_source_catalog import CatalogValidationError
from src.worker import inspect_model_bundle, prepare_dataset_request, run_training_request


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _proposal_policy(
    *, negative_ratio: int = 4, negative_exclusion_ms: int = 80, negative_seed: int = 20260822
) -> tuple[dict[str, object], dict[str, object]]:
    """Return the full policy emitted by the real proposal cache."""
    audio_features = normalize_pro_audio_preprocessing(None)
    policy = canonical_pro_event_proposal_negative_policy(
        audio_features=audio_features,
        requested_options={
            "negative_ratio": negative_ratio,
            "negative_exclusion_ms": negative_exclusion_ms,
            "negative_seed": negative_seed,
        },
    )
    return audio_features, policy


def _proposal_training(
    *,
    model_id: str = "fixture",
    negative_ratio: int = 4,
    negative_exclusion_ms: int = 80,
    seed: int = 20260822,
) -> dict[str, object]:
    return ProEventProposalTrainingOptions(
        model_id=model_id,
        negative_ratio=negative_ratio,
        negative_exclusion_ms=negative_exclusion_ms,
        seed=seed,
    ).portable()


def _known_event_training(*, model_id: str = "fixture") -> dict[str, object]:
    return ProEventTrainingOptions(model_id=model_id).portable()


def _write_selected_pro_candidate_bundle(
    root: Path, *, task_kind: str, candidate_kind: str
) -> tuple[Path, dict[str, object]]:
    """Write one minimal, hash-bound raw Pro bundle for contract tests."""
    contract = resolve_pro_candidate_contract(task_kind, candidate_kind)
    checkpoint = root / "weights" / "candidate.pt"
    config_path = root / "configs" / "candidate.json"
    checkpoint.parent.mkdir(parents=True)
    config_path.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"raw candidate")
    config: dict[str, object] = {
        "schema_version": 1,
        "format": contract.config_format,
        "task_kind": contract.task_kind,
        "pipeline_id": contract.pipeline_id,
        "model_implementation": contract.model_implementation,
        "input_contract": contract.input_contract,
        "output_contract": contract.output_contract,
        "training": _known_event_training(),
    }
    if contract.target_contract is not None:
        config["target_contract"] = contract.target_contract
        config["preprocessing"] = contract.preprocessing["id"]
    else:
        audio_features, policy = _proposal_policy()
        config["preprocessing"] = {
            "id": contract.preprocessing["id"],
            "audio_features": audio_features,
            "negative_policy": policy,
        }
        config["training"] = _proposal_training()
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "model_id": "pro-contract-fixture",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": {
            contract.component_id: {
                "checkpoint": checkpoint.relative_to(root).as_posix(),
                "sha256": _sha256_path(checkpoint),
                "byte_length": checkpoint.stat().st_size,
                "config": config_path.relative_to(root).as_posix(),
                "config_sha256": _sha256_path(config_path),
                "config_byte_length": config_path.stat().st_size,
                "architecture": contract.model_implementation,
                "preprocessing": contract.preprocessing["id"],
            }
        },
    }
    (root / MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return root, manifest


@pytest.mark.parametrize(
    "candidate_kind",
    [KNOWN_EVENT_CANDIDATE_KIND, FREE_RUNNING_PROPOSAL_CANDIDATE_KIND],
)
def test_selected_pro_candidate_bundle_contract_rejects_combined_or_relabelled_output(
    tmp_path: Path, candidate_kind: str
) -> None:
    contract = resolve_pro_candidate_contract("pro_guitar", candidate_kind)
    bundle, manifest = _write_selected_pro_candidate_bundle(
        tmp_path / candidate_kind.replace("/", "-"),
        task_kind="pro_guitar",
        candidate_kind=candidate_kind,
    )

    assert validate_pro_candidate_bundle(bundle, contract).profiles == {}

    # A generic portable preflight would accept this added well-formed file.
    # The selected-candidate validator must never let a host combine raw
    # known-event and proposal artifacts as one model.
    second_id = (
        "pro.guitar.event_proposal"
        if "attributes" in candidate_kind
        else "pro.guitar.event_attributes"
    )
    original = next(iter(manifest["components"].values()))
    assert isinstance(original, dict)
    manifest["components"][second_id] = dict(original)
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProCandidateContractError, match="component set"):
        validate_pro_candidate_bundle(bundle, contract)

    del manifest["components"][second_id]
    config_path = bundle / "configs" / "candidate.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # Rehashing proves this is not merely generic byte-integrity checking: it
    # is an independent semantic mismatch against the selected map.
    config["input_contract"] = {"format": "strum-pro-arbitrary-audio-window/v1"}
    config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
    original["config_sha256"] = _sha256_path(config_path)
    original["config_byte_length"] = config_path.stat().st_size
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProCandidateContractError, match="config input_contract"):
        validate_pro_candidate_bundle(bundle, contract)


def test_known_event_bundle_requires_exact_writer_config_and_training(
    tmp_path: Path,
) -> None:
    contract = resolve_pro_candidate_contract("pro_guitar", KNOWN_EVENT_CANDIDATE_KIND)
    bundle, manifest = _write_selected_pro_candidate_bundle(
        tmp_path / "known-event-config",
        task_kind="pro_guitar",
        candidate_kind=KNOWN_EVENT_CANDIDATE_KIND,
    )
    config_path = bundle / "configs" / "candidate.json"
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert validate_pro_candidate_bundle(bundle, contract).profiles == {}

    def rewrite_config(config: dict[str, object]) -> None:
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        component = manifest["components"][contract.component_id]
        assert isinstance(component, dict)
        component["config_sha256"] = _sha256_path(config_path)
        component["config_byte_length"] = config_path.stat().st_size
        (bundle / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    for container, key, value, message in (
        ("config", "unrelated", "benign", "known-event config has unsupported fields"),
        (
            "config",
            "unrelated_path",
            "/run/media/ash/portable-ai",
            "known-event config has unsupported fields",
        ),
        ("training", "unrelated", "benign", "known-event training options"),
        (
            "training",
            "unrelated_path",
            "/run/media/ash/portable-ai",
            "known-event training options",
        ),
    ):
        changed = copy.deepcopy(base_config)
        destination = changed if container == "config" else changed["training"]
        assert isinstance(destination, dict)
        destination[key] = value
        rewrite_config(changed)
        with pytest.raises(ProCandidateContractError, match=message):
            validate_pro_candidate_bundle(bundle, contract)


def test_proposal_bundle_requires_full_canonical_cache_negative_policy(tmp_path: Path) -> None:
    contract = resolve_pro_candidate_contract("pro_guitar", FREE_RUNNING_PROPOSAL_CANDIDATE_KIND)
    bundle, manifest = _write_selected_pro_candidate_bundle(
        tmp_path / "proposal-policy",
        task_kind="pro_guitar",
        candidate_kind=FREE_RUNNING_PROPOSAL_CANDIDATE_KIND,
    )
    config_path = bundle / "configs" / "candidate.json"
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = base_config["preprocessing"]["negative_policy"]
    assert policy == _proposal_policy()[1]
    assert validate_pro_candidate_bundle(bundle, contract).profiles == {}

    def rewrite_config(config: dict[str, object]) -> None:
        config_path.write_text(json.dumps(config, sort_keys=True), encoding="utf-8")
        component = manifest["components"][contract.component_id]
        assert isinstance(component, dict)
        component["config_sha256"] = _sha256_path(config_path)
        component["config_byte_length"] = config_path.stat().st_size
        (bundle / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    mutations = (
        ("selection", lambda value: value.__setitem__("selection", "caller-selected/v1")),
        (
            "requested_options",
            lambda value: value["requested_options"].__setitem__("negative_ratio", 1),
        ),
        (
            "feature_window",
            lambda value: value["feature_window"].__setitem__("window_after_frames", 1),
        ),
        (
            "real_event_exclusion",
            lambda value: value["real_event_exclusion"][
                "invalid_negative_center_offset_frames"
            ].__setitem__("minimum", 0),
        ),
    )
    for label, mutate in mutations:
        changed = copy.deepcopy(base_config)
        changed_policy = changed["preprocessing"]["negative_policy"]
        assert isinstance(changed_policy, dict)
        mutate(changed_policy)
        rewrite_config(changed)
        with pytest.raises(
            ProCandidateContractError, match="proposal negative policy|config training"
        ):
            validate_pro_candidate_bundle(bundle, contract)
        assert label

    for field, value in (
        ("negative_ratio", 4.0),
        ("negative_exclusion_ms", 80.0),
        ("seed", 20260822.0),
        ("negative_ratio", True),
        ("negative_exclusion_ms", True),
        ("seed", True),
    ):
        changed = copy.deepcopy(base_config)
        training = changed["training"]
        assert isinstance(training, dict)
        training[field] = value
        rewrite_config(changed)
        with pytest.raises(ProCandidateContractError, match="proposal training options"):
            validate_pro_candidate_bundle(bundle, contract)

    for container, key, value, message in (
        ("config", "unrelated", "benign", "proposal config has unsupported fields"),
        (
            "config",
            "unrelated_path",
            "/run/media/ash/portable-ai",
            "proposal config has unsupported fields",
        ),
        ("training", "unrelated", "benign", "proposal training options"),
        ("training", "unrelated_path", "/run/media/ash/portable-ai", "proposal training options"),
    ):
        changed = copy.deepcopy(base_config)
        destination = changed if container == "config" else changed["training"]
        assert isinstance(destination, dict)
        destination[key] = value
        rewrite_config(changed)
        with pytest.raises(ProCandidateContractError, match=message):
            validate_pro_candidate_bundle(bundle, contract)


def test_proposal_bundle_rejects_profile_companion_and_checkpoint_identity_changes(
    tmp_path: Path,
) -> None:
    contract = resolve_pro_candidate_contract("pro_guitar", FREE_RUNNING_PROPOSAL_CANDIDATE_KIND)
    bundle, manifest = _write_selected_pro_candidate_bundle(
        tmp_path / "proposal-identity",
        task_kind="pro_guitar",
        candidate_kind=FREE_RUNNING_PROPOSAL_CANDIDATE_KIND,
    )
    component = manifest["components"][contract.component_id]
    assert isinstance(component, dict)

    for field, value, message in (
        ("sha256", "0" * 64, "checkpoint sha256 mismatch"),
        ("byte_length", component["byte_length"] + 1, "checkpoint byte length mismatch"),
    ):
        changed = copy.deepcopy(manifest)
        changed_component = changed["components"][contract.component_id]
        assert isinstance(changed_component, dict)
        changed_component[field] = value
        (bundle / MANIFEST_FILENAME).write_text(
            json.dumps(changed, sort_keys=True), encoding="utf-8"
        )
        with pytest.raises((ProCandidateContractError, BundleValidationError), match=message):
            validate_pro_candidate_bundle(bundle, contract)

    (bundle / MANIFEST_FILENAME).write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    changed = copy.deepcopy(manifest)
    changed["companions"] = {"forbidden-runtime": {"kind": "runtime", "version": "v1"}}
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProCandidateContractError, match="companions"):
        validate_pro_candidate_bundle(bundle, contract)

    changed = copy.deepcopy(manifest)
    changed["profiles"] = {
        "forbidden-profile": {
            "capability": "pro.raw-candidate/v1",
            "instruments": ["guitar"],
            "required_components": [contract.component_id],
            "difficulty_policies": ["expert_only"],
        }
    }
    (bundle / MANIFEST_FILENAME).write_text(json.dumps(changed, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProCandidateContractError, match="profiles"):
        validate_pro_candidate_bundle(bundle, contract)


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
    digest = hashlib.sha256(content).hexdigest()
    relative = f"assets/sha256/{digest}/{filename}"
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "asset_id": f"sha256:{digest}",
        "sha256": digest,
        "relative_path": relative,
        "byte_length": len(content),
        "media_type": "audio/midi" if filename.endswith(".mid") else "audio/ogg",
    }


def _track(name: str, messages: list[mido.Message]) -> mido.MidiTrack:
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("track_name", name=name, time=0))
    track.extend(messages)
    return track


def _pro_midi(*, standard_fret: int = 3) -> bytes:
    midi = mido.MidiFile(ticks_per_beat=480)
    midi.tracks.append(
        _track(
            "PART REAL_GUITAR",
            [
                mido.Message("note_on", note=96, velocity=100 + standard_fret, channel=4, time=12),
                mido.Message("note_off", note=96, velocity=0, channel=4, time=120),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_GUITAR_22",
            [
                mido.Message("note_on", note=101, velocity=122, channel=3, time=24),
                mido.Message("note_off", note=101, velocity=0, channel=3, time=60),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_BASS_22",
            [
                mido.Message("note_on", note=98, velocity=115, channel=0, time=36),
                mido.Message("note_off", note=98, velocity=0, channel=0, time=48),
            ],
        )
    )
    midi.tracks.append(
        _track(
            "PART REAL_KEYS_X",
            [
                mido.Message("note_on", note=0, velocity=1, time=4),
                mido.Message("note_off", note=0, velocity=0, time=0),
                mido.Message("note_on", note=60, velocity=100, channel=1, time=8),
                mido.Message("note_off", note=60, velocity=0, channel=1, time=240),
            ],
        )
    )
    output = io.BytesIO()
    midi.save(file=output)
    return output.getvalue()


def _wave_bytes(seconds: float = 1.0, sample_rate: int = 22050) -> bytes:
    """Create a tiny valid PCM asset; the catalog extension is intentionally opaque."""
    samples = (0.1 * np.sin(np.arange(round(seconds * sample_rate)) / sample_rate * 440)).astype(
        np.float32
    )
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes((samples * 32767).astype("<i2").tobytes())
    return output.getvalue()


def _catalog(root: Path, *, standard_fret: int = 3, valid_audio: bool = False) -> None:
    source_id = "octave-src-pro-targets-0001"
    record = {
        "source_id": source_id,
        "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
        "rights": {"training_use": "allowed", "provenance": "reviewed", "license": "test-only"},
        "metadata": {"name": "Pro target test"},
        "chart": {
            "notes_midi": _asset(root, _pro_midi(standard_fret=standard_fret), "notes.mid"),
            "instruments": {
                "pro_guitar": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_GUITAR", "PART REAL_GUITAR_22"],
                },
                "pro_bass": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_BASS_22"],
                },
                "pro_keys": {
                    "status": "present",
                    "difficulties": ["expert"],
                    "track_names": ["PART REAL_KEYS_X"],
                },
            },
        },
        "audio": {
            role: _asset(
                root,
                _wave_bytes() if valid_audio else f"{source_id}:{role}".encode(),
                f"{role}.ogg",
            )
            for role in ("mix", "guitar", "bass", "keys")
        },
    }
    (root / "records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "pro-target-test",
                "records": "records.jsonl",
            }
        ),
        encoding="utf-8",
    )


def _catalog_with_train_and_val(root: Path, *, valid_audio: bool = False) -> None:
    """Create enough immutable catalog records to prove split-disjoint training.

    The shared managed MIDI/audio bytes are intentional: task identity is the
    OCTAVE source ID, while this worker test only needs the exact-track target
    contract and both required splits.  It mocks materialization and training
    so it cannot accidentally exercise a path-bearing external process.
    """
    _catalog(root, valid_audio=valid_audio)
    record = json.loads((root / "records.jsonl").read_text(encoding="utf-8"))
    records = []
    for index in range(32):
        clone = dict(record)
        clone["source_id"] = f"octave-src-proevent-{index:08x}"
        clone["metadata"] = {"name": f"Pro worker fixture {index}"}
        records.append(clone)
    (root / "records.jsonl").write_text(
        "\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8"
    )


@pytest.mark.parametrize(
    ("task_kind", "expected_schema", "expected_event"),
    [
        (
            "pro_guitar",
            "pro-string-fret-events/v1",
            {"string": 0, "fret": 3, "technique": "tapped"},
        ),
        ("pro_bass", "pro-string-fret-events/v1", {"string": 2, "fret": 15, "technique": "normal"}),
        ("pro_keys", "pro-keys-pitch-events/v1", {"pitch": 60, "channel": 1}),
    ],
)
def test_pro_targets_are_decoded_from_exact_tracks_without_path_leaks(
    tmp_path: Path, task_kind: str, expected_schema: str, expected_event: dict[str, object]
) -> None:
    _catalog(tmp_path)

    manifest = build_catalog_pro_target_manifest(tmp_path, task_kind)

    serialized = json.dumps(manifest)
    assert manifest["format"] == PRO_TARGET_MANIFEST_FORMAT
    assert manifest["target_encoding"]["id"] == "strum-pro-midi-target-decoder/v1"
    assert str(tmp_path) not in serialized
    track = manifest["songs"][0]["targets"][0]
    assert track["event_schema"] == expected_schema
    assert expected_event.items() <= track["events"][0].items()
    if task_kind == "pro_guitar":
        assert manifest["songs"][0]["targets"][1]["track_variant"] == "22_fret"
    if task_kind == "pro_keys":
        assert track["range_shifts"] == [{"tick": 4, "anchor": "C"}]

    resolved = resolve_catalog_pro_target_manifest_songs(manifest, tmp_path)
    assert str(tmp_path) in resolved[0]["midi_path"]
    assert resolved[0]["targets"] == manifest["songs"][0]["targets"]


def test_pro_targets_exclude_invalid_standard_fret_before_candidate_training(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path, standard_fret=22)

    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_guitar")

    assert manifest["songs"] == []
    assert manifest["summary"]["coverage_record_count"] == 1
    assert manifest["summary"]["exclusion_reason_counts"] == {
        "Pro string target uses an unsupported technique or fret": 1
    }
    # Decoding labels is not a claim that STRUM can generate playable Pro
    # charts; even the known-event candidate receives no malformed targets.
    assert build_catalog_task_manifest(tmp_path, "pro_guitar")["task"]["kind"] == "pro_guitar"


def test_pro_target_resolution_rejects_tampering_or_catalog_drift(tmp_path: Path) -> None:
    _catalog(tmp_path)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_keys")
    manifest["songs"][0]["targets"][0]["events"][0]["pitch"] = 61
    with pytest.raises(CatalogValidationError, match="targets do not match"):
        resolve_catalog_pro_target_manifest_songs(manifest, tmp_path)


def test_pro_candidate_experiment_lineage_uses_target_view_hashes_without_paths(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_guitar")

    sources = _source_inputs(manifest)

    assert len(sources) == 1
    assert sources[0]["audio_sha256"] == manifest["task_view"]["songs"][0]["audio"]["sha256"]
    assert (
        sources[0]["notes_midi_sha256"] == manifest["task_view"]["songs"][0]["notes_midi"]["sha256"]
    )
    assert str(tmp_path) not in json.dumps(sources)


def test_worker_prepare_returns_a_decoded_pro_target_view(tmp_path: Path) -> None:
    _catalog(tmp_path)
    output = tmp_path / "out" / "pro-keys-targets.json"
    request = tmp_path / "prepare.json"
    request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "strum.instrument-chart/pro-keys/v1",
                "output": str(output),
                "options": {},
            }
        ),
        encoding="utf-8",
    )

    result = prepare_dataset_request(request)
    written = json.loads(output.read_text(encoding="utf-8"))

    assert result["status"] == "prepared"
    assert result["record_count"] == 1
    assert result["output_name"] == output.name
    assert written["format"] == PRO_TARGET_MANIFEST_FORMAT
    assert str(tmp_path) not in json.dumps(written)


def test_worker_runs_pro_candidate_with_dirty_provenance_and_no_deployment_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker must preserve source state without exposing local paths.

    This crosses the actual request parser, exact-Pro task resolver, candidate
    bundle writer, preflight, and inspection boundary.  Audio materialization
    and the CUDA/CPU trainer are mocked so the regression is quick and solely
    tests worker/provenance semantics.
    """
    _catalog_with_train_and_val(tmp_path)
    target_view = tmp_path / "views" / "pro-guitar-targets.json"
    target_view.parent.mkdir()
    target_view.write_text(
        json.dumps(build_catalog_pro_target_manifest(tmp_path, "pro_guitar")),
        encoding="utf-8",
    )
    decoded = json.loads(target_view.read_text(encoding="utf-8"))
    assert {song["split"] for song in decoded["songs"]} >= {"train", "val"}

    output_dir = tmp_path / "candidate"
    request = tmp_path / "pro-train.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "strum.instrument-chart/pro-guitar/v1",
                "task_view": str(target_view),
                "output": str(output_dir),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "pro-worker-dirty-fixture",
                    "epochs": 1,
                    "device": "cpu",
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_preprocess(**kwargs: object) -> dict[str, object]:
        assert kwargs["splits"] == ("train", "val")
        return {
            "preprocessing": {"id": PRO_AUDIO_PREPROCESSING_ID},
            "splits": {"train": {"window_count": 3}, "val": {"window_count": 2}},
        }

    def fake_train(command: list[str]) -> None:
        checkpoint_dir = Path(command[command.index("--checkpoint-dir") + 1])
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best.pt").write_bytes(b"mock known-event candidate")
        (checkpoint_dir / "history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 1,
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "val_known_event_token_f1": 0.75,
                        "val_known_event_state_accuracy": 0.8,
                        "val_known_event_exact_accuracy": 0.7,
                    }
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr("src.pro_event_worker_training.prepare_pro_audio_windows", fake_preprocess)
    monkeypatch.setattr("src.pro_event_worker_training._run_script", fake_train)
    monkeypatch.setattr("src.worker._revision", lambda: ("a" * 40, True))

    result = run_training_request(request)
    inspection = inspect_model_bundle(output_dir / "bundle")
    manifest = json.loads((output_dir / "bundle" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    experiment = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))

    assert result["status"] == "completed"
    assert result["deployment_status"] == (
        "not_deployable_requires_pro_event_proposal_sequence_evaluation_and_packaging"
    )
    assert result["runtime"] == {
        "strum_version": result["runtime"]["strum_version"],
        "strum_revision": "a" * 40,
        "strum_source_dirty": True,
        "device": "cpu",
    }
    assert inspection["profiles"] == []
    assert inspection["deployment_status"] == "not_deployable"
    assert inspection["compatibility"]["strum_source_dirty"] is True
    assert manifest["compatibility"]["strum_revision"] == "a" * 40
    assert manifest["compatibility"]["strum_source_dirty"] is True
    assert "profiles" not in manifest
    assert experiment["candidate_scope"] == "known_reference_event_attributes_only"
    assert experiment["runtime"]["strum_source_dirty"] is True
    assert (
        validate_pro_candidate_bundle(
            output_dir / "bundle",
            resolve_pro_candidate_contract("pro_guitar", KNOWN_EVENT_CANDIDATE_KIND),
        ).profiles
        == {}
    )
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(inspection)
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(experiment)


def test_pro_audio_preprocessing_materializes_exact_event_windows_without_paths(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path, valid_audio=True)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_guitar")
    assert manifest["audio_preprocessing"]["id"] == PRO_AUDIO_PREPROCESSING_ID
    task_view = tmp_path / "pro-guitar-targets.json"
    task_view.write_text(json.dumps(manifest), encoding="utf-8")
    split = manifest["songs"][0]["split"]

    result = prepare_pro_audio_windows(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "cache",
        splits=(split,),
    )

    cache = tmp_path / "cache"
    labels = [
        json.loads(line) for line in (cache / f"{split}_targets.jsonl").read_text().splitlines()
    ]
    features = np.load(cache / f"{split}_logmel.npy")
    serialized = (cache / "preprocess_summary.json").read_text() + "\n".join(
        json.dumps(label) for label in labels
    )
    assert result["format"] == "strum-pro-audio-feature-cache/v1"
    assert features.shape[0] == len(labels) == 2
    assert features.shape[1:] == (128, 22)
    assert {label["track_variant"] for label in labels} == {"standard", "22_fret"}
    assert all(label["target_language"] == "string_fret_technique/v1" for label in labels)
    assert str(tmp_path) not in serialized


def test_worker_runs_proposal_candidate_without_midi_or_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _catalog_with_train_and_val(tmp_path)
    task_view = tmp_path / "views" / "pro-guitar-targets.json"
    task_view.parent.mkdir()
    task_view.write_text(
        json.dumps(build_catalog_pro_target_manifest(tmp_path, "pro_guitar")), encoding="utf-8"
    )
    output_dir = tmp_path / "proposal-candidate"
    request = tmp_path / "pro-proposal-train.json"
    request.write_text(
        json.dumps(
            {
                "pipeline_id": "strum.instrument-chart/pro-guitar/v1",
                "task_view": str(task_view),
                "output": str(output_dir),
                "catalog_root": str(tmp_path),
                "options": {
                    "model_id": "pro-proposal-fixture",
                    "candidate_kind": "free_running_event_proposal/v1",
                    "epochs": 1,
                    "device": "cpu",
                    "negative_ratio": 2,
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_preprocess(**kwargs: object) -> dict[str, object]:
        assert kwargs["splits"] == ("train", "val")
        assert kwargs["negative_ratio"] == 2
        assert kwargs["negative_exclusion_ms"] == 80
        assert kwargs["negative_seed"] == 20260822
        audio_features, negative_policy = _proposal_policy(negative_ratio=2)
        return {
            "preprocessing": {
                "id": "pro-logmel-event-proposal-windows/v1",
                "audio_features": audio_features,
                "negative_policy": negative_policy,
            },
            "splits": {
                "train": {"positive_window_count": 3, "negative_window_count": 6},
                "val": {"positive_window_count": 2, "negative_window_count": 4},
            },
        }

    def fake_train(command: list[str]) -> None:
        checkpoint_dir = Path(command[command.index("--checkpoint-dir") + 1])
        checkpoint_dir.mkdir(parents=True)
        (checkpoint_dir / "best.pt").write_bytes(b"mock proposal candidate")
        (checkpoint_dir / "history.json").write_text(
            json.dumps(
                [
                    {
                        "epoch": 1,
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "val_proposal_precision": 0.75,
                        "val_proposal_recall": 0.5,
                        "val_proposal_f1": 0.6,
                        "val_proposal_balanced_accuracy": 0.7,
                    }
                ]
            ),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "src.pro_event_proposal_worker_training.prepare_pro_event_proposal_windows", fake_preprocess
    )
    monkeypatch.setattr("src.pro_event_proposal_worker_training._run_script", fake_train)
    monkeypatch.setattr("src.worker._revision", lambda: ("b" * 40, False))

    result = run_training_request(request)
    inspection = inspect_model_bundle(output_dir / "bundle")
    manifest = json.loads((output_dir / "bundle" / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    experiment = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
    config = json.loads(
        (output_dir / "bundle" / "configs" / "pro_guitar-event-proposal.json").read_text(
            encoding="utf-8"
        )
    )

    assert result["status"] == "completed"
    assert result["deployment_status"] == (
        "not_deployable_requires_pro_sequence_evaluation_packaging_and_execution"
    )
    assert inspection["profiles"] == []
    assert inspection["deployment_status"] == "not_deployable"
    assert set(manifest["components"]) == {"pro.guitar.event_proposal"}
    assert "profiles" not in manifest
    assert experiment["candidate_scope"] == "free_running_audio_event_proposal_only"
    assert experiment["release_requirements"]["status"] == "blocked"
    audio_features, negative_policy = _proposal_policy(negative_ratio=2)
    assert config["preprocessing"] == {
        "id": "pro-logmel-event-proposal-windows/v1",
        "audio_features": audio_features,
        "negative_policy": negative_policy,
    }
    assert experiment["preprocessing"]["negative_policy_id"] == (
        "pro-event-proposal-asymmetric-window-exclusion/v1"
    )
    assert str(tmp_path) not in json.dumps(result)
    assert str(tmp_path) not in json.dumps(inspection)
    assert str(tmp_path) not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(experiment)


def test_pro_audio_preprocessing_rejects_modified_feature_contract(tmp_path: Path) -> None:
    _catalog(tmp_path)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_keys")
    manifest["audio_preprocessing"]["id"] = "five-lane-logmel/v1"
    task_view = tmp_path / "pro-keys-targets.json"
    task_view.write_text(json.dumps(manifest), encoding="utf-8")
    split = manifest["songs"][0]["split"]

    with pytest.raises(CatalogValidationError, match="audio preprocessing"):
        prepare_pro_audio_windows(
            manifest_path=task_view,
            catalog_root=tmp_path,
            cache_dir=tmp_path / "cache",
            splits=(split,),
        )


def test_pro_keys_audio_preprocessing_retains_chromatic_targets_and_range_shifts(
    tmp_path: Path,
) -> None:
    _catalog(tmp_path, valid_audio=True)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_keys")
    task_view = tmp_path / "pro-keys-targets.json"
    task_view.write_text(json.dumps(manifest), encoding="utf-8")
    split = manifest["songs"][0]["split"]

    prepare_pro_audio_windows(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "cache",
        splits=(split,),
    )

    labels = [
        json.loads(line)
        for line in (tmp_path / "cache" / f"{split}_targets.jsonl").read_text().splitlines()
    ]
    assert len(labels) == 1
    assert labels[0]["event_schema"] == "pro-keys-pitch-events/v1"
    assert labels[0]["target_language"] == "pitch_channel_range_shift/v1"
    assert labels[0]["events"] == [{"channel": 1, "duration_ticks": 240, "pitch": 60, "tick": 12}]
    assert labels[0]["range_shifts"] == [{"anchor": "C", "tick": 4}]


def test_pro_audio_tick_alignment_honors_global_tempo_changes(tmp_path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    tempo = mido.MidiTrack()
    tempo.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    tempo.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=480))
    midi.tracks.append(tempo)
    path = tmp_path / "tempo.mid"
    midi.save(path)

    ticks_per_beat, segments = _tempo_segments(path)

    assert _tick_seconds(480, ticks_per_beat, segments) == pytest.approx(0.5)
    assert _tick_seconds(960, ticks_per_beat, segments) == pytest.approx(1.5)


def test_pro_proposal_preprocessing_generates_deterministic_negative_audio_windows(
    tmp_path: Path,
) -> None:
    _catalog_with_train_and_val(tmp_path, valid_audio=True)
    manifest = build_catalog_pro_target_manifest(tmp_path, "pro_guitar")
    assert {song["split"] for song in manifest["songs"]} >= {"train", "val"}
    task_view = tmp_path / "pro-guitar-targets.json"
    task_view.write_text(json.dumps(manifest), encoding="utf-8")

    first = prepare_pro_event_proposal_windows(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "proposal-cache-first",
        negative_ratio=2,
        negative_seed=31,
    )
    second = prepare_pro_event_proposal_windows(
        manifest_path=task_view,
        catalog_root=tmp_path,
        cache_dir=tmp_path / "proposal-cache-second",
        negative_ratio=2,
        negative_seed=31,
    )

    assert first == second
    policy = first["preprocessing"]["negative_policy"]
    assert policy["id"] == PRO_EVENT_PROPOSAL_NEGATIVE_POLICY_ID
    assert policy == canonical_pro_event_proposal_negative_policy(
        audio_features=first["preprocessing"]["audio_features"],
        requested_options={
            "negative_ratio": 2,
            "negative_exclusion_ms": 80,
            "negative_seed": 31,
        },
    )
    assert (
        validate_pro_event_proposal_negative_policy(
            policy, audio_features=first["preprocessing"]["audio_features"]
        )
        == policy
    )
    assert policy["feature_window"] == {
        "center": "candidate_center_frame",
        "window_before_ms": 100,
        "window_after_ms": 400,
        "window_before_frames": 4,
        "window_after_frames": 17,
    }
    assert policy["real_event_exclusion"] == {
        "rule": "negative-feature-window-must-not-contain-real-event-onset/v1",
        "local_onset_exclusion_frames": 4,
        "invalid_negative_center_offset_frames": {
            "minimum": -17,
            "maximum": 4,
            "inclusive": True,
        },
    }
    for split in ("train", "val"):
        cache = tmp_path / "proposal-cache-first"
        labels = [
            json.loads(line)
            for line in (cache / f"{split}_proposal_targets.jsonl").read_text().splitlines()
        ]
        features = np.load(cache / f"{split}_logmel.npy")
        assert features.shape[0] == len(labels)
        assert any(row["is_event"] for row in labels)
        assert any(not row["is_event"] for row in labels)
        assert all(row["split"] == split for row in labels)
        assert all(str(tmp_path) not in json.dumps(row) for row in labels)
        real_event_centers_by_source: dict[str, list[int]] = {}
        for row in labels:
            if row["is_event"]:
                real_event_centers_by_source.setdefault(row["source_id"], []).append(
                    row["center_frame"]
                )
        # Regression: a negative feature spans [center - 100ms, center +
        # 400ms].  It must never contain a REAL onset from its source, even
        # though the center itself can be farther than the local 80ms
        # tolerance from that onset.
        for row in labels:
            if row["is_event"]:
                continue
            negative_start = row["center_frame"] - policy["feature_window"]["window_before_frames"]
            negative_end = row["center_frame"] + policy["feature_window"]["window_after_frames"]
            assert all(
                not negative_start <= event_center <= negative_end
                for event_center in real_event_centers_by_source[row["source_id"]]
            )
    assert first["format"] == "strum-pro-event-proposal-feature-cache/v1"
    assert (
        str(tmp_path)
        not in (tmp_path / "proposal-cache-first" / "preprocess_summary.json").read_text()
    )


def test_pro_proposal_negative_centers_exclude_asymmetric_feature_window_overlap() -> None:
    """A center far outside local tolerance can still reveal a REAL onset."""
    centers = _negative_centers(
        source_id="geometry-fixture",
        frame_count=80,
        positives=[40],
        count=80,
        before=4,
        after=17,
        local_exclusion=4,
        seed=31,
    )

    # The 80ms local tolerance alone would allow frame 30, yet its
    # [26, 47] feature window contains the REAL onset at frame 40.  All
    # returned negatives obey the complete feature-window boundary instead.
    assert 30 not in centers
    assert all(center < 23 or center > 44 for center in centers)


def test_proposal_trainer_learns_only_binary_audio_window_scores(tmp_path: Path) -> None:
    script = Path(__file__).parents[1] / "scripts" / "train_pro_event_proposals.py"
    spec = importlib.util.spec_from_file_location("train_pro_event_proposals", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cache = tmp_path / "cache"
    cache.mkdir()
    for split in ("train", "val"):
        np.save(cache / f"{split}_logmel.npy", np.zeros((4, 8, 8), dtype=np.float16))
        (cache / f"{split}_proposal_targets.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "source_id": f"source-{split}-{index}",
                        "split": split,
                        "center_frame": index,
                        "is_event": index % 2 == 0,
                    }
                )
                + "\n"
                for index in range(4)
            ),
            encoding="utf-8",
        )
    metrics = module.train_pro_event_proposals(
        module.TrainingSettings(
            cache_dir=cache,
            checkpoint_dir=tmp_path / "checkpoints",
            task_kind="pro_guitar",
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            seed=12,
            channels=2,
        )
    )
    checkpoint = torch.load(tmp_path / "checkpoints" / "best.pt", map_location="cpu")
    assert {"val_proposal_f1", "val_proposal_precision", "val_proposal_recall"} <= metrics.keys()
    assert checkpoint["format"] == "strum-pro-event-proposal-candidate-checkpoint/v1"
    assert "midi" not in checkpoint
