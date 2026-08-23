import hashlib
import json
from pathlib import Path

import pytest

from src.model_bundle import (
    MANIFEST_FILENAME,
    BundleValidationError,
    discover_model_bundles,
    legacy_model_bundle,
    load_model_bundle,
)


def write_manifest(root: Path, components: dict, **overrides: object) -> Path:
    manifest = {
        "schema_version": 1,
        "model_id": "test-model",
        "compatibility": {"manifest_schema": 1, "strum_version": ">=0.1.0"},
        "components": components,
    }
    manifest.update(overrides)
    path = root / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_load_resolves_component_paths_and_checks_hashes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint")
    config = tmp_path / "configs" / "model.yaml"
    config.parent.mkdir()
    config.write_text("model: {}\n", encoding="utf-8")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    write_manifest(
        tmp_path,
        {
            "drums.v14_onset": {
                "checkpoint": "weights/best.pt",
                "config": "configs/model.yaml",
                "sha256": digest,
            }
        },
    )

    bundle = load_model_bundle(tmp_path, check_files=True)

    component = bundle.component("drums.v14_onset")
    assert component is not None
    assert component.checkpoint == checkpoint
    assert component.config == config
    assert bundle.validate(check_files=True, verify_hashes=True) == []


def test_manifest_rejects_path_escape(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"checkpoint": "../outside.pt"}})

    with pytest.raises(BundleValidationError, match="escapes the bundle root"):
        load_model_bundle(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "model_id",
            "/run/media/ash/private-model",
            "model_id must be a lowercase stable identifier",
        ),
        (
            "component_id",
            "../../source-path",
            "components key must be a lowercase stable identifier",
        ),
        ("profile_id", "/home/ash/profile", "profiles key must be a lowercase stable identifier"),
    ],
)
def test_manifest_rejects_path_like_public_identifiers(
    tmp_path: Path, field: str, value: str, match: str
) -> None:
    components = {"guitar.onset": {"checkpoint": "weights/best.pt"}}
    overrides: dict[str, object] = {}
    if field == "model_id":
        overrides["model_id"] = value
    elif field == "component_id":
        components = {value: {"checkpoint": "weights/best.pt"}}
    else:
        overrides["profiles"] = {
            value: {
                "capability": "guitar.neural-v1-expert/v1",
                "instruments": ["guitar"],
                "required_components": ["guitar.onset"],
                "difficulty_policies": ["expert_only"],
            }
        }
    write_manifest(tmp_path, components, **overrides)

    with pytest.raises(BundleValidationError, match=match):
        load_model_bundle(tmp_path)


def test_manifest_rejects_path_like_profile_metadata(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        profiles={
            "guitar-profile": {
                "capability": "/run/media/ash/guitar/v1",
                "instruments": ["guitar"],
                "required_components": ["guitar.onset"],
                "difficulty_policies": ["expert_only"],
            }
        },
    )

    with pytest.raises(BundleValidationError, match="versioned capability identifier"):
        load_model_bundle(tmp_path)


def test_manifest_rejects_checksum_without_checkpoint(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"sha256": "a" * 64}})

    with pytest.raises(BundleValidationError, match="requires a checkpoint"):
        load_model_bundle(tmp_path)


def test_missing_files_are_optional_until_explicitly_checked(tmp_path: Path) -> None:
    write_manifest(tmp_path, {"guitar.onset": {"checkpoint": "weights/missing.pt"}})

    bundle = load_model_bundle(tmp_path)
    assert bundle.validate(check_files=True) == [
        f"guitar.onset: checkpoint not found: {tmp_path / 'weights' / 'missing.pt'}"
    ]
    with pytest.raises(BundleValidationError, match="checkpoint not found"):
        load_model_bundle(tmp_path, check_files=True)


def test_declared_source_revision_is_unverified_without_runtime_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("STRUM_SOURCE_REVISION", raising=False)
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        compatibility={
            "manifest_schema": 1,
            "strum_version": ">=0.1.0",
            "strum_revision": "abc123",
        },
    )

    bundle = load_model_bundle(tmp_path)

    assert bundle.validate() == []
    assert bundle.compatibility_status() == [
        "STRUM source revision abc123: declared, unverified (set STRUM_SOURCE_REVISION to verify)"
    ]


def test_source_revision_mismatch_is_rejected_when_runtime_revision_is_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("STRUM_SOURCE_REVISION", "different")
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        compatibility={
            "manifest_schema": 1,
            "strum_version": ">=0.1.0",
            "strum_revision": "abc123",
        },
    )

    with pytest.raises(BundleValidationError, match="requires STRUM source revision abc123"):
        load_model_bundle(tmp_path)


def test_discovery_lists_valid_child_bundles_only(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    write_manifest(valid, {"guitar.onset": {"checkpoint": "weights/best.pt"}}, model_id="good")
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    (invalid / MANIFEST_FILENAME).write_text("not json", encoding="utf-8")

    bundles = discover_model_bundles(tmp_path)

    assert [bundle.model_id for bundle in bundles] == ["good"]


def test_legacy_bundle_preserves_current_checkpoint_layout(tmp_path: Path) -> None:
    bundle = legacy_model_bundle(tmp_path)

    assert bundle.legacy
    assert bundle.checkpoint("drums.v14_onset") == tmp_path / "checkpoints/drums_v14/best.pt"
    assert bundle.config("drums.ensemble.v17") == tmp_path / "configs/onset_classifier_v17.yaml"


def test_profile_configuration_is_relative_and_checked(tmp_path: Path) -> None:
    checkpoint = tmp_path / "weights" / "best.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"checkpoint")
    config = tmp_path / "profiles" / "guitar-rule.json"
    config.parent.mkdir()
    config.write_text("{}")
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        profiles={
            "guitar-rule": {
                "capability": "guitar.hybrid-v2-rule/v1",
                "instruments": ["guitar"],
                "required_components": ["guitar.onset"],
                "difficulty_policies": ["expert_only"],
                "configuration": "profiles/guitar-rule.json",
                "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                "configuration_byte_length": config.stat().st_size,
            }
        },
    )

    bundle = load_model_bundle(tmp_path, check_files=True)

    assert bundle.profile("guitar-rule").configuration == config
    assert bundle.profile("guitar-rule").configuration_sha256


def test_profile_graph_declares_a_path_free_composed_profile(tmp_path: Path) -> None:
    components: dict[str, dict[str, object]] = {}
    for component_id in (
        "separation.demucs",
        "guitar.onset",
        "guitar.mapper",
        "guitar.assembly",
    ):
        checkpoint = tmp_path / "weights" / f"{component_id}.bin"
        checkpoint.parent.mkdir(exist_ok=True)
        checkpoint.write_bytes(component_id.encode())
        components[component_id] = {
            "checkpoint": checkpoint.relative_to(tmp_path).as_posix(),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "byte_length": checkpoint.stat().st_size,
        }
    write_manifest(
        tmp_path,
        components,
        companions={"demucs": {"kind": "runtime", "version": ">=4.0"}},
        profiles={
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
    )

    bundle = load_model_bundle(tmp_path, check_files=True)

    profile = bundle.profile("guitar-composed")
    assert profile is not None and profile.graph is not None
    assert bundle.profile_summary(profile) == {
        "profile_id": "guitar-composed",
        "capability": "guitar.composed/v1",
        "instruments": ["guitar"],
        "required_components": list(components),
        "required_companions": [{"id": "demucs", "kind": "runtime", "version": ">=4.0"}],
        "difficulty_policies": ["expert_only"],
        "composition": {
            "format": "strum-profile-composition/v1",
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
                    "required": True,
                    "component_ids": ["guitar.onset"],
                    "companion_ids": [],
                    "depends_on": ["separate"],
                    "inputs": ["artifact.stem.guitar"],
                    "outputs": ["artifact.guitar.onsets"],
                    "instrument": "guitar",
                    "difficulty": "Expert",
                },
                {
                    "id": "map",
                    "kind": "fret_mapping",
                    "required": True,
                    "component_ids": ["guitar.mapper"],
                    "companion_ids": [],
                    "depends_on": ["detect"],
                    "inputs": ["artifact.guitar.onsets"],
                    "outputs": ["artifact.guitar.events"],
                    "instrument": "guitar",
                    "difficulty": "Expert",
                },
                {
                    "id": "assemble",
                    "kind": "chart_assembly",
                    "required": True,
                    "component_ids": ["guitar.assembly"],
                    "companion_ids": [],
                    "depends_on": ["map"],
                    "inputs": ["artifact.guitar.events"],
                    "outputs": ["chart.guitar.expert"],
                    "instrument": "guitar",
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


def test_profile_graph_rejects_an_undeclared_required_companion(tmp_path: Path) -> None:
    write_manifest(
        tmp_path,
        {"guitar.onset": {"checkpoint": "weights/best.pt"}},
        profiles={
            "invalid-composed": {
                "capability": "guitar.composed/v1",
                "instruments": ["guitar"],
                "required_components": ["guitar.onset"],
                "difficulty_policies": ["expert_only"],
                "graph": {
                    "stages": [
                        {
                            "id": "detect",
                            "kind": "onset_detection",
                            "instrument": "guitar",
                            "required": True,
                            "component_ids": ["guitar.onset"],
                            "companion_ids": ["demucs"],
                            "depends_on": [],
                            "inputs": ["source.audio.mix"],
                            "outputs": ["chart.guitar.expert"],
                        }
                    ],
                    "outputs": [
                        {
                            "instrument": "guitar",
                            "stage_id": "detect",
                            "artifact_id": "chart.guitar.expert",
                            "difficulty": "Expert",
                        }
                    ],
                },
            }
        },
    )

    with pytest.raises(BundleValidationError, match="undeclared companion demucs"):
        load_model_bundle(tmp_path)
