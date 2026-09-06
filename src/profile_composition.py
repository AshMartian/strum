"""Portable composition of independently promoted Expert chart profiles.

The composer is deliberately worker-owned: callers provide private source
locations in a request file, while the output contains only relative paths and
hashes.  Each child bundle remains byte-for-byte intact so its typed loader can
continue to enforce its original promotion and evaluation evidence.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from src import __version__
from src.model_bundle import (
    MANIFEST_FILENAME,
    BundleValidationError,
    InferenceProfile,
    ModelBundle,
    load_model_bundle,
)

COMPOSITION_PROFILE_ID = "five-lane-composition"
COMPOSITION_CAPABILITY = "five-lane.composition/v1"
COMPOSITION_FORMAT = "strum-five-lane-composition-profile/v1"
COMPOSITION_ASSETS_DIRECTORY = "_composition_assets"

_SUPPORTED_CAPABILITIES = {
    "guitar": {"guitar.hybrid-v2-rule/v1", "guitar.neural-v1-expert/v1"},
    "bass": {"bass.neural-v1-expert/v1"},
    "keys": {"keys.neural-v1-expert/v1"},
    "drums": {"drums.v14-expert/v1"},
}


def _validate_child_execution_profile(bundle: ModelBundle, profile: InferenceProfile) -> None:
    """Run the same typed admission boundary used by the direct executor."""
    if profile.capability == "guitar.hybrid-v2-rule/v1":
        from src.inference.guitar_hybrid_profile import load_guitar_hybrid_rule_profile

        load_guitar_hybrid_rule_profile(bundle, profile.profile_id)
    elif profile.capability == "guitar.neural-v1-expert/v1":
        from src.inference.guitar_neural_profile import load_guitar_neural_expert_profile

        load_guitar_neural_expert_profile(bundle, profile.profile_id)
    elif profile.capability == "bass.neural-v1-expert/v1":
        from src.inference.bass_neural_profile import load_bass_neural_expert_profile

        load_bass_neural_expert_profile(bundle, profile.profile_id)
    elif profile.capability == "keys.neural-v1-expert/v1":
        from src.inference.keys_neural_profile import load_keys_neural_expert_profile

        load_keys_neural_expert_profile(bundle, profile.profile_id)
    elif profile.capability == "drums.v14-expert/v1":
        from src.inference.drums_v14_profile import load_drums_v14_expert_profile

        load_drums_v14_expert_profile(bundle, profile.profile_id)
    else:  # guarded by _SUPPORTED_CAPABILITIES, retained as a fail-closed boundary
        raise BundleValidationError("composition child capability is unsupported")


def _resolve_child_root(bundle: ModelBundle, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise BundleValidationError("composition child path is invalid")
    relative = Path(value)
    if relative.is_absolute() or relative.parts[:1] != (COMPOSITION_ASSETS_DIRECTORY,):
        raise BundleValidationError("composition child path is outside composition assets")
    resolved = (bundle.root / relative).resolve()
    try:
        resolved.relative_to(bundle.root / COMPOSITION_ASSETS_DIRECTORY)
    except ValueError as error:
        raise BundleValidationError("composition child path escapes composition assets") from error
    return resolved


def load_composition_profile(
    bundle: ModelBundle, profile_id: str
) -> dict[str, tuple[ModelBundle, InferenceProfile]]:
    """Validate the outer bundle and return its verified, intact child profiles."""
    profile = bundle.profile(profile_id)
    if (
        profile is None
        or profile.capability != COMPOSITION_CAPABILITY
        or profile.difficulty_policies != ("expert_only",)
        or profile.configuration is None
        or profile.required_companions
        or not 2 <= len(profile.instruments) <= len(_SUPPORTED_CAPABILITIES)
    ):
        raise BundleValidationError("composition profile is invalid")
    try:
        config = json.loads(profile.configuration.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("composition profile configuration is unreadable") from error
    if (
        not isinstance(config, dict)
        or set(config) != {"schema_version", "format", "children"}
        or config.get("schema_version") != 1
        or config.get("format") != COMPOSITION_FORMAT
        or not isinstance(config.get("children"), dict)
        or set(config["children"]) != set(profile.instruments)
    ):
        raise BundleValidationError("composition profile configuration is invalid")
    expected_components: set[str] = set()
    resolved: dict[str, tuple[ModelBundle, InferenceProfile]] = {}
    for instrument in profile.instruments:
        child = config["children"][instrument]
        if not isinstance(child, dict) or set(child) != {
            "path",
            "profile_id",
            "manifest_sha256",
            "capability",
        }:
            raise BundleValidationError("composition child configuration is invalid")
        child_root = _resolve_child_root(bundle, child["path"])
        child_bundle = load_model_bundle(child_root, check_files=True)
        if (
            child_bundle.manifest_path is None
            or _sha256(child_bundle.manifest_path) != child["manifest_sha256"]
        ):
            raise BundleValidationError("composition child manifest identity does not match")
        child_errors = child_bundle.validate(check_files=True, verify_hashes=True)
        if child_errors:
            raise BundleValidationError("composition child bundle failed validation")
        child_profile_id, declared_capability = child.get("profile_id"), child.get("capability")
        if not isinstance(child_profile_id, str) or not isinstance(declared_capability, str):
            raise BundleValidationError("composition child identity is invalid")
        child_profile = child_bundle.profile(child_profile_id)
        if (
            child_profile is None
            or child_profile.capability != declared_capability
            or child_profile.capability not in _SUPPORTED_CAPABILITIES[instrument]
            or child_profile.instruments != (instrument,)
            or child_profile.difficulty_policies != ("expert_only",)
            or child_profile.required_companions
        ):
            raise BundleValidationError("composition child profile is incompatible")
        _validate_child_execution_profile(child_bundle, child_profile)
        expected_components.update(
            f"{instrument}.{component_id}" for component_id in child_profile.required_components
        )
        resolved[instrument] = (child_bundle, child_profile)
    if set(profile.required_components) != expected_components:
        raise BundleValidationError("composition profile component provenance is incomplete")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_request(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleValidationError("composition request is unreadable") from error
    if not isinstance(raw, dict) or set(raw) != {"output", "profiles"}:
        raise BundleValidationError("composition request has unsupported fields")
    return raw


def _require_safe_tree(root: Path) -> None:
    if not root.is_dir():
        raise BundleValidationError("composition child bundle is unavailable")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise BundleValidationError("composition child bundle must not contain symlinks")


def _validated_children(raw: dict[str, Any]) -> list[tuple[str, str, Path, str, tuple[str, ...]]]:
    selected = raw["profiles"]
    if not isinstance(selected, list) or not 2 <= len(selected) <= len(_SUPPORTED_CAPABILITIES):
        raise BundleValidationError("composition requires two to four child profiles")
    children: list[tuple[str, str, Path, str, tuple[str, ...]]] = []
    seen_instruments: set[str] = set()
    for item in selected:
        if not isinstance(item, dict) or set(item) != {"model_root", "profile_id"}:
            raise BundleValidationError("composition child has unsupported fields")
        model_root, profile_id = item.get("model_root"), item.get("profile_id")
        if (
            not isinstance(model_root, str)
            or not model_root
            or not isinstance(profile_id, str)
            or not profile_id
        ):
            raise BundleValidationError("composition child locations and profile IDs are required")
        root = Path(model_root).resolve()
        _require_safe_tree(root)
        bundle = load_model_bundle(root, check_files=True)
        errors = bundle.validate(check_files=True, verify_hashes=True)
        if errors:
            raise BundleValidationError("composition child bundle failed validation")
        profile = bundle.profile(profile_id)
        if (
            profile is None
            or len(profile.instruments) != 1
            or profile.difficulty_policies != ("expert_only",)
            or profile.required_companions
        ):
            raise BundleValidationError(
                "composition child must be a companion-free Expert-only profile"
            )
        instrument = profile.instruments[0]
        if profile.capability not in _SUPPORTED_CAPABILITIES.get(instrument, set()):
            raise BundleValidationError("composition child capability is unsupported")
        if instrument in seen_instruments:
            raise BundleValidationError("composition children must have unique instruments")
        if profile.configuration is None or bundle.manifest_path is None:
            raise BundleValidationError("composition child has incomplete profile provenance")
        components = tuple(profile.required_components)
        if not components:
            raise BundleValidationError("composition child has no required components")
        _validate_child_execution_profile(bundle, profile)
        children.append((instrument, profile_id, root, _sha256(bundle.manifest_path), components))
        seen_instruments.add(instrument)
    return children


def compose_profile_request(request_path: Path) -> dict[str, object]:
    """Create one immutable multi-instrument composition bundle.

    The private request schema is exactly::

        {"output": "/private/destination", "profiles": [
          {"model_root": "/private/guitar", "profile_id": "guitar-expert"}
        ]}
    """
    raw = _read_request(request_path)
    if not isinstance(raw["output"], str) or not raw["output"]:
        raise BundleValidationError("composition output is required")
    output = Path(raw["output"]).resolve()
    if output.exists() or not output.parent.is_dir():
        raise BundleValidationError(
            "composition output must be a new path below an existing directory"
        )
    children = _validated_children(raw)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        root_components: dict[str, dict[str, object]] = {}
        stages: list[dict[str, object]] = []
        graph_outputs: list[dict[str, str]] = []
        child_records: dict[str, object] = {}
        required_components: list[str] = []
        for instrument, profile_id, source_root, manifest_sha256, component_ids in children:
            child_root = staging / COMPOSITION_ASSETS_DIRECTORY / instrument
            shutil.copytree(source_root, child_root)
            child_bundle = load_model_bundle(child_root, check_files=True)
            child_profile = child_bundle.profile(profile_id)
            if child_profile is None:
                raise BundleValidationError("composition child profile disappeared while packaging")
            prefixed_components: list[str] = []
            for component_id in component_ids:
                component = child_bundle.component(component_id)
                if component is None or component.checkpoint is None or component.sha256 is None:
                    raise BundleValidationError("composition child component is incomplete")
                prefixed = f"{instrument}.{component_id}"
                copied_component: dict[str, object] = {
                    "checkpoint": component.checkpoint.relative_to(child_root).as_posix(),
                    "sha256": component.sha256,
                    "byte_length": component.byte_length,
                    **(
                        {
                            "config": component.config.relative_to(child_root).as_posix(),
                            "config_sha256": component.config_sha256,
                            "config_byte_length": component.config_byte_length,
                        }
                        if component.config is not None
                        else {}
                    ),
                    "architecture": component.architecture,
                    "preprocessing": component.preprocessing,
                }
                copied_component["checkpoint"] = (
                    f"{COMPOSITION_ASSETS_DIRECTORY}/{instrument}/{copied_component['checkpoint']}"
                )
                if "config" in copied_component:
                    copied_component["config"] = (
                        f"{COMPOSITION_ASSETS_DIRECTORY}/{instrument}/{copied_component['config']}"
                    )
                root_components[prefixed] = copied_component
                prefixed_components.append(prefixed)
                required_components.append(prefixed)
            stage_id = f"chart-{instrument}"
            stages.append(
                {
                    "id": stage_id,
                    "kind": "profile_child_execution",
                    "instrument": instrument,
                    "required": True,
                    "component_ids": prefixed_components,
                    "companion_ids": [],
                    "depends_on": [],
                    "inputs": ["source.audio.mix"],
                    "outputs": [f"chart.{instrument}.expert"],
                    "difficulty": "Expert",
                }
            )
            graph_outputs.append(
                {
                    "instrument": instrument,
                    "stage_id": stage_id,
                    "artifact_id": f"chart.{instrument}.expert",
                    "difficulty": "Expert",
                }
            )
            child_records[instrument] = {
                "path": f"{COMPOSITION_ASSETS_DIRECTORY}/{instrument}",
                "profile_id": profile_id,
                "manifest_sha256": manifest_sha256,
                "capability": child_profile.capability,
            }
        configuration = {
            "schema_version": 1,
            "format": COMPOSITION_FORMAT,
            "children": child_records,
        }
        config_path = staging / "profiles" / f"{COMPOSITION_PROFILE_ID}.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text(
            json.dumps(configuration, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1,
            "model_id": "five-lane-composition-v1",
            "compatibility": {"manifest_schema": 1, "strum_version": f">={__version__}"},
            "components": root_components,
            "profiles": {
                COMPOSITION_PROFILE_ID: {
                    "capability": COMPOSITION_CAPABILITY,
                    "instruments": [instrument for instrument, *_rest in children],
                    "required_components": required_components,
                    "difficulty_policies": ["expert_only"],
                    "configuration": config_path.relative_to(staging).as_posix(),
                    "configuration_sha256": _sha256(config_path),
                    "configuration_byte_length": config_path.stat().st_size,
                    "graph": {"stages": stages, "outputs": graph_outputs},
                }
            },
        }
        (staging / MANIFEST_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        errors = load_model_bundle(staging, check_files=True).validate(
            check_files=True, verify_hashes=True
        )
        if errors:
            raise BundleValidationError(
                "composition output failed validation: " + "; ".join(errors)
            )
        staging.replace(output)
    return {
        "status": "packaged",
        "profile_id": COMPOSITION_PROFILE_ID,
        "capability": COMPOSITION_CAPABILITY,
        "instruments": [instrument for instrument, *_rest in children],
        "bundle_name": output.name,
        "manifest_sha256": _sha256(output / MANIFEST_FILENAME),
    }
