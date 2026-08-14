"""Versioned, machine-readable STRUM worker contract.

This entry point intentionally exposes discovery and safe preflight first. It
does not import the mutable auto-chart pipeline while OCTAVE is deciding which
runtime or model bundle to use. Training and chart execution are added as
pipeline handlers behind the same protocol rather than requiring callers to
know STRUM's source-tree scripts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from src import PROJECT_ROOT, __version__
from src.model_bundle import BundleValidationError, InferenceProfile, ModelBundle, load_model_bundle

PROTOCOL_VERSION = 1
MODEL_BUNDLE_SCHEMA_VERSIONS = (1,)


@dataclass(frozen=True)
class PipelineDescriptor:
    """A STRUM-owned pipeline capability visible to host applications."""

    id: str
    display_name: str
    kind: str
    version: int
    catalog_requirements: dict[str, object]
    checkpoint_outputs: tuple[str, ...]
    inference_capability: str | None
    status: str

    def as_json(self) -> dict[str, object]:
        data = asdict(self)
        data["checkpoint_outputs"] = list(self.checkpoint_outputs)
        return data


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
        },
        checkpoint_outputs=("guitar_onset", "guitar_fret"),
        inference_capability="guitar.audio_to_chart/v1",
        status="planned",
    ),
    PipelineDescriptor(
        id="difficulty.transform/v1",
        display_name="Learned five-lane difficulty transform",
        kind="chart_to_chart",
        version=1,
        catalog_requirements={
            "instruments": ["guitar", "bass", "keys", "drums"],
            "source_difficulty": "expert",
            "target_difficulties": ["hard", "medium", "easy"],
        },
        checkpoint_outputs=("chart_transform",),
        inference_capability="difficulty.transform/v1",
        status="planned",
    ),
    PipelineDescriptor(
        id="drums.onset-velocity/v1",
        display_name="Drums onset + velocity",
        kind="audio_to_chart",
        version=1,
        catalog_requirements={
            "instrument": "drums",
            "difficulties": ["expert"],
            "audio_roles": ["drums", "mix"],
            "audio_policy": "prefer:drums,fallback:mix",
        },
        checkpoint_outputs=("drums_onset",),
        inference_capability="drums.audio_to_chart/v1",
        status="planned",
    ),
)


def _revision() -> tuple[str | None, bool | None]:
    """Return a best-effort revision without making Git a runtime dependency."""
    configured = os.environ.get("STRUM_SOURCE_REVISION", "").strip()
    if configured:
        return configured, os.environ.get("STRUM_SOURCE_DIRTY", "0") == "1"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _runtime_payload() -> dict[str, object]:
    revision, dirty = _revision()
    capabilities = ["pipeline_discovery", "model_bundle_preflight", "checkpoint_inspect"]
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
        "pipelines": [pipeline.id for pipeline in PIPELINES if pipeline.status == "available"],
    }


def _manifest_sha256(bundle: ModelBundle) -> str | None:
    if bundle.manifest_path is None:
        return None
    return hashlib.sha256(bundle.manifest_path.read_bytes()).hexdigest()


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
        components.append(
            {
                "id": name,
                "checkpoint": str(component.checkpoint),
                "sha256": component.sha256,
                "byte_length": component.byte_length,
                "architecture": component.architecture,
                "preprocessing": component.preprocessing,
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
    return {
        **plan,
        "profile_id": profile.profile_id,
        "capability": profile.capability,
        "instruments": list(profile.instruments),
        "difficulty_policy": difficulty_policy,
    }


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True))


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
    model = commands.add_parser("model", help="inspect model bundles")
    model_commands = model.add_subparsers(dest="model_command", required=True)
    preflight = model_commands.add_parser("preflight", help="validate a deployable model bundle")
    preflight.add_argument("--model-root", type=Path, required=True)
    preflight.add_argument("--require-component", action="append", default=[])
    preflight.add_argument("--json", action="store_true")
    checkpoint = commands.add_parser("checkpoint", help="inspect checkpoint bundle metadata")
    checkpoint_commands = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    inspect = checkpoint_commands.add_parser("inspect", help="inspect a checkpoint bundle")
    inspect.add_argument("--model-root", type=Path, required=True)
    inspect.add_argument("--json", action="store_true")
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
        if args.command == "model" and args.model_command == "preflight":
            _print_json(
                preflight_bundle(args.model_root, required_components=args.require_component)
            )
            return 0
        if args.command == "checkpoint" and args.checkpoint_command == "inspect":
            bundle = load_model_bundle(args.model_root, check_files=False)
            _print_json(
                {
                    "model_id": bundle.model_id,
                    "manifest_sha256": _manifest_sha256(bundle),
                    "components": sorted(bundle.components),
                    "compatibility": bundle.compatibility,
                }
            )
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
    except BundleValidationError as error:
        _print_json({"status": "invalid", "code": "model_bundle_invalid", "message": str(error)})
        return 2
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
