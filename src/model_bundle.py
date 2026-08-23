"""Versioned STRUM model-bundle manifests and checkpoint resolution.

Bundles make a trained set of checkpoints portable without changing the
working-directory assumptions of the existing pipeline.  A bundle directory
contains ``strum-model-bundle.json`` and any checkpoint/configuration files it
references.  Paths in a manifest are always relative to that directory.

When no bundle is selected, :func:`get_active_bundle` exposes the repository's
historic ``checkpoints/`` layout as a virtual legacy bundle.  This is
intentional: existing commands and installations continue to work unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import PROJECT_ROOT, __version__
from src.source_provenance import source_revision_identity

MANIFEST_FILENAME = "strum-model-bundle.json"
MANIFEST_SCHEMA_VERSION = 1
PROFILE_COMPOSITION_FORMAT = "strum-profile-composition/v1"
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]*$")


class BundleValidationError(ValueError):
    """Raised when a model bundle manifest is invalid or incompatible."""


def _graph_ancestors(stage_id: str, stages: dict[str, ProfileGraphStage]) -> set[str]:
    """Return transitive dependencies without recursing through malformed cycles."""
    ancestors: set[str] = set()
    pending = list(stages[stage_id].depends_on)
    while pending:
        dependency = pending.pop()
        if dependency in ancestors or dependency not in stages:
            continue
        ancestors.add(dependency)
        pending.extend(stages[dependency].depends_on)
    return ancestors


def _graph_has_cycle(stages: dict[str, ProfileGraphStage]) -> bool:
    """Check directed stage dependencies with a small deterministic DFS."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> bool:
        if stage_id in visiting:
            return True
        if stage_id in visited:
            return False
        visiting.add(stage_id)
        for dependency in stages[stage_id].depends_on:
            if dependency in stages and visit(dependency):
                return True
        visiting.remove(stage_id)
        visited.add(stage_id)
        return False

    return any(visit(stage_id) for stage_id in stages)


@dataclass(frozen=True)
class ModelComponent:
    """One named checkpoint-bearing component within a model bundle."""

    name: str
    root: Path
    checkpoint: Path | None = None
    config: Path | None = None
    sha256: str | None = None
    byte_length: int | None = None
    config_sha256: str | None = None
    config_byte_length: int | None = None
    required: bool = True
    architecture: str | None = None
    preprocessing: str | None = None


@dataclass(frozen=True)
class RuntimeCompanion:
    """A versioned non-checkpoint dependency required by a profile graph.

    A companion is intentionally declarative.  The manifest records the
    runtime dependency STRUM must later verify; merely declaring one never
    claims that the current worker can load or execute it.
    """

    companion_id: str
    kind: str
    version: str

    def as_json(self) -> dict[str, str]:
        return {"id": self.companion_id, "kind": self.kind, "version": self.version}


@dataclass(frozen=True)
class ProfileGraphStage:
    """One typed node in a composed auto-chart profile graph."""

    stage_id: str
    kind: str
    instrument: str | None
    required: bool
    component_ids: tuple[str, ...]
    companion_ids: tuple[str, ...]
    depends_on: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    difficulty: str | None = None
    difficulty_policies: tuple[str, ...] = ()

    def as_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.stage_id,
            "kind": self.kind,
            "required": self.required,
            "component_ids": list(self.component_ids),
            "companion_ids": list(self.companion_ids),
            "depends_on": list(self.depends_on),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }
        if self.instrument is not None:
            payload["instrument"] = self.instrument
        if self.difficulty is not None:
            payload["difficulty"] = self.difficulty
        if self.difficulty_policies:
            payload["difficulty_policies"] = list(self.difficulty_policies)
        return payload


@dataclass(frozen=True)
class ProfileGraphOutput:
    """A terminal chart artifact declared by a profile graph."""

    instrument: str
    stage_id: str
    artifact_id: str
    difficulty: str

    def as_json(self) -> dict[str, str]:
        return {
            "instrument": self.instrument,
            "stage_id": self.stage_id,
            "artifact_id": self.artifact_id,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class ProfileGraph:
    """A path-free composition plan for a complete profile."""

    stages: tuple[ProfileGraphStage, ...]
    outputs: tuple[ProfileGraphOutput, ...]

    def as_json(self) -> dict[str, object]:
        return {
            "format": PROFILE_COMPOSITION_FORMAT,
            "stages": [stage.as_json() for stage in self.stages],
            "outputs": [output.as_json() for output in self.outputs],
        }


@dataclass(frozen=True)
class InferenceProfile:
    """A complete, deployable auto-chart capability within a model bundle."""

    profile_id: str
    capability: str
    instruments: tuple[str, ...]
    required_components: tuple[str, ...]
    required_companions: tuple[str, ...]
    difficulty_policies: tuple[str, ...]
    configuration: Path | None = None
    configuration_sha256: str | None = None
    configuration_byte_length: int | None = None
    graph: ProfileGraph | None = None


@dataclass(frozen=True)
class ModelBundle:
    """A parsed manifest with paths resolved relative to its bundle root."""

    root: Path
    model_id: str
    schema_version: int
    compatibility: dict[str, Any]
    components: dict[str, ModelComponent]
    companions: dict[str, RuntimeCompanion]
    profiles: dict[str, InferenceProfile]
    manifest_path: Path | None = None
    legacy: bool = False

    def component(self, name: str) -> ModelComponent | None:
        """Return a declared component, or ``None`` for an optional override."""
        return self.components.get(name)

    def profile(self, profile_id: str) -> InferenceProfile | None:
        """Return a declared inference profile without loading its weights."""
        return self.profiles.get(profile_id)

    def checkpoint(self, name: str, default: Path | None = None) -> Path | None:
        """Resolve a component checkpoint, falling back to ``default`` when absent."""
        component = self.component(name)
        return component.checkpoint if component and component.checkpoint else default

    def config(self, name: str, default: Path | None = None) -> Path | None:
        """Resolve a component configuration, falling back to ``default`` when absent."""
        component = self.component(name)
        return component.config if component and component.config else default

    def validate(self, *, check_files: bool = False, verify_hashes: bool = False) -> list[str]:
        """Return human-readable validation errors without loading any ML code."""
        errors: list[str] = []
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            errors.append(
                f"unsupported manifest schema {self.schema_version}; "
                f"expected {MANIFEST_SCHEMA_VERSION}"
            )
        required_schema = self.compatibility.get("manifest_schema")
        if required_schema is not None and required_schema != MANIFEST_SCHEMA_VERSION:
            errors.append(
                "bundle requires manifest schema "
                f"{required_schema}, but STRUM supports {MANIFEST_SCHEMA_VERSION}"
            )
        required_strum = self.compatibility.get("strum_version")
        if required_strum is not None and not _version_is_compatible(__version__, required_strum):
            errors.append(f"bundle requires STRUM {required_strum}; running {__version__}")
        required_revision = self.compatibility.get("strum_revision")
        if required_revision is not None:
            if source_revision_identity(required_revision) is None:
                errors.append("bundle strum_revision must be a safe Git revision identity")
            else:
                runtime_revision, runtime_revision_invalid = _runtime_revision_configuration()
                if runtime_revision_invalid:
                    errors.append("STRUM source revision configuration is invalid")
                elif runtime_revision is not None and runtime_revision != required_revision:
                    errors.append(
                        f"bundle requires STRUM source revision {required_revision}; "
                        f"running {runtime_revision}"
                    )

        if check_files:
            for component in self.components.values():
                for label, path in (
                    ("checkpoint", component.checkpoint),
                    ("config", component.config),
                ):
                    if path is not None and not path.is_file():
                        errors.append(f"{component.name}: {label} not found: {path}")
                if (
                    verify_hashes
                    and component.sha256
                    and component.checkpoint
                    and component.checkpoint.is_file()
                ):
                    actual = _sha256(component.checkpoint)
                    if actual != component.sha256:
                        errors.append(
                            f"{component.name}: checkpoint sha256 mismatch "
                            f"(expected {component.sha256}, got {actual})"
                        )
                if (
                    component.byte_length is not None
                    and component.checkpoint
                    and component.checkpoint.is_file()
                    and component.checkpoint.stat().st_size != component.byte_length
                ):
                    errors.append(
                        f"{component.name}: checkpoint byte length mismatch "
                        f"(expected {component.byte_length}, got {component.checkpoint.stat().st_size})"
                    )
                if (
                    verify_hashes
                    and component.config_sha256
                    and component.config
                    and component.config.is_file()
                    and _sha256(component.config) != component.config_sha256
                ):
                    errors.append(f"{component.name}: config sha256 mismatch")
                if (
                    component.config_byte_length is not None
                    and component.config
                    and component.config.is_file()
                    and component.config.stat().st_size != component.config_byte_length
                ):
                    errors.append(f"{component.name}: config byte length mismatch")
        for profile in self.profiles.values():
            for component_name in profile.required_components:
                if component_name not in self.components:
                    errors.append(
                        f"profile {profile.profile_id} requires undeclared component {component_name}"
                    )
            for companion_name in profile.required_companions:
                if companion_name not in self.companions:
                    errors.append(
                        f"profile {profile.profile_id} requires undeclared companion {companion_name}"
                    )
            if profile.graph is not None:
                errors.extend(self._validate_profile_graph(profile))
            if profile.configuration is not None:
                if not profile.configuration.is_file():
                    errors.append(
                        f"profile {profile.profile_id}: configuration not found: {profile.configuration}"
                    )
                elif (
                    verify_hashes and _sha256(profile.configuration) != profile.configuration_sha256
                ):
                    errors.append(f"profile {profile.profile_id}: configuration sha256 mismatch")
                elif profile.configuration.stat().st_size != profile.configuration_byte_length:
                    errors.append(
                        f"profile {profile.profile_id}: configuration byte length mismatch"
                    )
        return errors

    def _validate_profile_graph(self, profile: InferenceProfile) -> list[str]:
        """Validate composition semantics after manifest references are resolved."""
        assert profile.graph is not None
        graph = profile.graph
        errors: list[str] = []
        stages = {stage.stage_id: stage for stage in graph.stages}
        graph_components = {
            component_id for stage in graph.stages for component_id in stage.component_ids
        }
        graph_companions = {
            companion_id for stage in graph.stages for companion_id in stage.companion_ids
        }
        missing_graph_components = set(profile.required_components) - graph_components
        if missing_graph_components:
            errors.append(
                f"profile {profile.profile_id} required_components are absent from graph: "
                f"{', '.join(sorted(missing_graph_components))}"
            )
        missing_graph_companions = set(profile.required_companions) - graph_companions
        if missing_graph_companions:
            errors.append(
                f"profile {profile.profile_id} required_companions are absent from graph: "
                f"{', '.join(sorted(missing_graph_companions))}"
            )
        produced_by = {
            artifact_id: stage.stage_id for stage in graph.stages for artifact_id in stage.outputs
        }
        if len(produced_by) != sum(len(stage.outputs) for stage in graph.stages):
            errors.append(f"profile {profile.profile_id} graph artifact outputs must be unique")
        graph_instruments = {output.instrument for output in graph.outputs}
        if graph_instruments != set(profile.instruments):
            errors.append(
                f"profile {profile.profile_id} graph outputs must cover exactly its profile instruments"
            )
        for stage in graph.stages:
            if stage.instrument is not None and stage.instrument not in profile.instruments:
                errors.append(
                    f"profile {profile.profile_id} graph stage {stage.stage_id} names an unsupported instrument"
                )
            for dependency in stage.depends_on:
                if dependency not in stages:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} depends on unknown stage {dependency}"
                    )
            if stage.stage_id in stage.depends_on:
                errors.append(
                    f"profile {profile.profile_id} graph stage {stage.stage_id} cannot depend on itself"
                )
            for component_name in stage.component_ids:
                if component_name not in self.components:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} requires undeclared component {component_name}"
                    )
                elif stage.required and component_name not in profile.required_components:
                    errors.append(
                        f"profile {profile.profile_id} graph required stage {stage.stage_id} component {component_name} is not in required_components"
                    )
            for companion_name in stage.companion_ids:
                if companion_name not in self.companions:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} requires undeclared companion {companion_name}"
                    )
                elif stage.required and companion_name not in profile.required_companions:
                    errors.append(
                        f"profile {profile.profile_id} graph required stage {stage.stage_id} companion {companion_name} is not in required_companions"
                    )
            for policy in stage.difficulty_policies:
                if policy not in profile.difficulty_policies:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} names unsupported difficulty policy {policy}"
                    )
            for artifact_id in stage.outputs:
                if not artifact_id.startswith(("artifact.", "chart.")):
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} output {artifact_id} must be an artifact or chart identity"
                    )
            ancestors = _graph_ancestors(stage.stage_id, stages)
            for artifact_id in stage.inputs:
                if artifact_id.startswith("source."):
                    continue
                producer = produced_by.get(artifact_id)
                if producer is None:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} input {artifact_id} has no producer"
                    )
                elif producer not in ancestors:
                    errors.append(
                        f"profile {profile.profile_id} graph stage {stage.stage_id} input {artifact_id} is not supplied by a dependency"
                    )
        for output in graph.outputs:
            stage = stages.get(output.stage_id)
            if not output.artifact_id.startswith("chart."):
                errors.append(
                    f"profile {profile.profile_id} graph terminal output {output.artifact_id} must be a chart identity"
                )
            if stage is None:
                errors.append(
                    f"profile {profile.profile_id} graph output names unknown stage {output.stage_id}"
                )
            elif output.artifact_id not in stage.outputs:
                errors.append(
                    f"profile {profile.profile_id} graph output {output.artifact_id} is not emitted by stage {output.stage_id}"
                )
            elif stage.instrument not in {None, output.instrument}:
                errors.append(
                    f"profile {profile.profile_id} graph output instrument does not match stage {output.stage_id}"
                )
        if _graph_has_cycle(stages):
            errors.append(f"profile {profile.profile_id} graph must be acyclic")
        return errors

    def profile_summary(self, profile: InferenceProfile) -> dict[str, object]:
        """Return path-free profile discovery data for editor integrations."""
        payload: dict[str, object] = {
            "profile_id": profile.profile_id,
            "capability": profile.capability,
            "instruments": list(profile.instruments),
            "required_components": list(profile.required_components),
            "required_companions": [
                self.companions[companion].as_json() for companion in profile.required_companions
            ],
            "difficulty_policies": list(profile.difficulty_policies),
        }
        if profile.graph is not None:
            payload["composition"] = profile.graph.as_json()
        return payload

    def compatibility_status(self) -> list[str]:
        """Describe compatibility declarations that cannot be verified locally."""
        source_dirty = self.compatibility.get("strum_source_dirty")
        source_dirty_status: list[str] = []
        if "strum_source_dirty" in self.compatibility:
            if source_dirty is True:
                source_dirty_status.append(
                    "STRUM source tree was dirty when this artifact was built"
                )
            elif source_dirty is None:
                source_dirty_status.append(
                    "STRUM source tree state was unknown when this artifact was built"
                )
        required_revision = self.compatibility.get("strum_revision")
        if not isinstance(required_revision, str) or not required_revision.strip():
            return source_dirty_status
        runtime_revision, runtime_revision_invalid = _runtime_revision_configuration()
        if runtime_revision_invalid:
            return [
                *source_dirty_status,
                "STRUM source revision configuration is invalid; "
                "declared revision cannot be verified",
            ]
        if runtime_revision is None:
            return [
                *source_dirty_status,
                f"STRUM source revision {required_revision}: declared, unverified "
                "(set STRUM_SOURCE_REVISION to verify)",
            ]
        return [*source_dirty_status, f"STRUM source revision {required_revision}: verified"]


def _version_is_compatible(current: str, requirement: object) -> bool:
    """Check a simple exact or comma-separated semantic-version requirement.

    This intentionally supports the small subset needed by portable manifests
    without making model discovery depend on ``packaging`` at runtime.
    """
    if not isinstance(requirement, str) or not requirement.strip():
        return False
    if requirement == current:
        return True

    current_parts = _version_tuple(current)
    for term in (part.strip() for part in requirement.split(",")):
        if term.startswith(">="):
            if current_parts < _version_tuple(term[2:]):
                return False
        elif term.startswith(">"):
            if current_parts <= _version_tuple(term[1:]):
                return False
        elif term.startswith("<="):
            if current_parts > _version_tuple(term[2:]):
                return False
        elif term.startswith("<"):
            if current_parts >= _version_tuple(term[1:]):
                return False
        elif term.startswith("=="):
            if current != term[2:]:
                return False
        else:
            return False
    return True


def get_runtime_revision() -> str | None:
    """Return an explicitly supplied STRUM source revision, if one is known.

    Wheels and copied source trees do not reliably contain Git metadata.  The
    caller that pins STRUM (for example, an editor integration) can therefore
    set ``STRUM_SOURCE_REVISION`` to make a bundle's revision requirement
    enforceable.  Absence means "declared but unverified", not incompatibility.
    Invalid configured values intentionally remain indistinguishable from
    absence to callers that only need a safe display identity; validation uses
    :func:`_runtime_revision_configuration` to fail closed for that case.
    """
    revision, _invalid = _runtime_revision_configuration()
    return revision


def _runtime_revision_configuration() -> tuple[str | None, bool]:
    """Return the safe configured revision and whether configuration is invalid.

    ``None`` has two externally redacted meanings: an unset runtime revision
    and an unsafe value that cannot leave the host.  Bundle compatibility must
    distinguish them internally: an unset value leaves a manifest-pinned
    revision unverified, whereas an unsafe configured value is an explicit
    failed attestation.  The boolean carries only that fact and never exposes
    the original environment value.
    """
    configured = os.environ.get("STRUM_SOURCE_REVISION")
    if configured is None:
        return None, False
    revision = source_revision_identity(configured)
    return revision, revision is None


def _version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.strip().split("."))
    except ValueError as error:
        raise BundleValidationError(f"invalid STRUM version requirement: {value!r}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative_path(root: Path, value: object, field: str, component: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"{component}.{field} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute():
        raise BundleValidationError(f"{component}.{field} must be relative to the bundle root")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise BundleValidationError(f"{component}.{field} escapes the bundle root") from error
    return resolved


def _parse_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise BundleValidationError(f"{field} must be a lowercase stable identifier")
    return value


def _parse_identifier_list(
    value: object, field: str, *, allow_empty: bool = True
) -> tuple[str, ...]:
    if not isinstance(value, list) or (not allow_empty and not value):
        raise BundleValidationError(
            f"{field} must be a {'non-empty ' if not allow_empty else ''}list"
        )
    values = tuple(_parse_identifier(item, f"{field} item") for item in value)
    if len(set(values)) != len(values):
        raise BundleValidationError(f"{field} must not contain duplicates")
    return values


def _parse_capability(value: object, field: str) -> str:
    """Parse a public capability identity without accepting a filesystem path.

    Capabilities intentionally have one or more namespace/version segments
    (for example ``guitar.neural-v1-expert/v1``), unlike opaque local paths.
    Keep every segment in the same stable identifier alphabet used by bundle
    IDs, components, and profiles before it can reach a host renderer.
    """
    if not isinstance(value, str) or not value:
        raise BundleValidationError(f"{field} must be a versioned capability identifier")
    segments = value.split("/")
    if len(segments) < 2 or any(not _IDENTIFIER_PATTERN.fullmatch(segment) for segment in segments):
        raise BundleValidationError(f"{field} must be a versioned capability identifier")
    return value


def _parse_companion(name: str, value: object) -> RuntimeCompanion:
    if not isinstance(value, dict):
        raise BundleValidationError(f"companions.{name} must be an object")
    unknown = set(value) - {"kind", "version"}
    if unknown:
        raise BundleValidationError(
            f"companions.{name} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    kind = value.get("kind")
    version = value.get("version")
    if kind not in {"runtime", "model_service"}:
        raise BundleValidationError(f"companions.{name}.kind must be runtime or model_service")
    if not isinstance(version, str) or not version.strip():
        raise BundleValidationError(f"companions.{name}.version must be a non-empty string")
    return RuntimeCompanion(companion_id=name, kind=kind, version=version)


def _parse_profile_graph_stage(value: object, profile_name: str) -> ProfileGraphStage:
    if not isinstance(value, dict):
        raise BundleValidationError(f"profiles.{profile_name}.graph.stages items must be objects")
    allowed = {
        "id",
        "kind",
        "instrument",
        "required",
        "component_ids",
        "companion_ids",
        "depends_on",
        "inputs",
        "outputs",
        "difficulty",
        "difficulty_policies",
    }
    required = {
        "id",
        "kind",
        "required",
        "component_ids",
        "companion_ids",
        "depends_on",
        "inputs",
        "outputs",
    }
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages has unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages missing field(s): {', '.join(sorted(missing))}"
        )
    stage_id = _parse_identifier(value["id"], f"profiles.{profile_name}.graph.stages.id")
    kind = _parse_identifier(value["kind"], f"profiles.{profile_name}.graph.stages.{stage_id}.kind")
    instrument = value.get("instrument")
    if instrument is not None:
        instrument = _parse_identifier(
            instrument, f"profiles.{profile_name}.graph.stages.{stage_id}.instrument"
        )
    stage_required = value["required"]
    if not isinstance(stage_required, bool):
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages.{stage_id}.required must be a boolean"
        )
    difficulty = value.get("difficulty")
    if difficulty is not None and (not isinstance(difficulty, str) or not difficulty.strip()):
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages.{stage_id}.difficulty must be a non-empty string"
        )
    policies_value = value.get("difficulty_policies", [])
    if (
        not isinstance(policies_value, list)
        or not all(isinstance(policy, str) and policy for policy in policies_value)
        or len(set(policies_value)) != len(policies_value)
    ):
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages.{stage_id}.difficulty_policies must be a unique string list"
        )
    return ProfileGraphStage(
        stage_id=stage_id,
        kind=kind,
        instrument=instrument,
        required=stage_required,
        component_ids=_parse_identifier_list(
            value["component_ids"], f"profiles.{profile_name}.graph.stages.{stage_id}.component_ids"
        ),
        companion_ids=_parse_identifier_list(
            value["companion_ids"], f"profiles.{profile_name}.graph.stages.{stage_id}.companion_ids"
        ),
        depends_on=_parse_identifier_list(
            value["depends_on"], f"profiles.{profile_name}.graph.stages.{stage_id}.depends_on"
        ),
        inputs=_parse_identifier_list(
            value["inputs"], f"profiles.{profile_name}.graph.stages.{stage_id}.inputs"
        ),
        outputs=_parse_identifier_list(
            value["outputs"], f"profiles.{profile_name}.graph.stages.{stage_id}.outputs"
        ),
        difficulty=difficulty,
        difficulty_policies=tuple(policies_value),
    )


def _parse_profile_graph(value: object, profile_name: str) -> ProfileGraph:
    if not isinstance(value, dict):
        raise BundleValidationError(f"profiles.{profile_name}.graph must be an object")
    unknown = set(value) - {"stages", "outputs"}
    required = {"stages", "outputs"}
    missing = required - set(value)
    if unknown:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph has unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph missing field(s): {', '.join(sorted(missing))}"
        )
    if not isinstance(value["stages"], list) or not value["stages"]:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.stages must be a non-empty list"
        )
    stages = tuple(_parse_profile_graph_stage(stage, profile_name) for stage in value["stages"])
    stage_ids = {stage.stage_id for stage in stages}
    if len(stage_ids) != len(stages):
        raise BundleValidationError(f"profiles.{profile_name}.graph.stages must have unique ids")
    if not isinstance(value["outputs"], list) or not value["outputs"]:
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.outputs must be a non-empty list"
        )
    outputs: list[ProfileGraphOutput] = []
    for output in value["outputs"]:
        if not isinstance(output, dict):
            raise BundleValidationError(
                f"profiles.{profile_name}.graph.outputs items must be objects"
            )
        unknown_output = set(output) - {"instrument", "stage_id", "artifact_id", "difficulty"}
        required_output = {"instrument", "stage_id", "artifact_id", "difficulty"}
        missing_output = required_output - set(output)
        if unknown_output or missing_output:
            details = unknown_output or missing_output
            raise BundleValidationError(
                f"profiles.{profile_name}.graph.outputs has invalid field(s): {', '.join(sorted(details))}"
            )
        instrument = _parse_identifier(
            output["instrument"], f"profiles.{profile_name}.graph.outputs.instrument"
        )
        stage_id = _parse_identifier(
            output["stage_id"], f"profiles.{profile_name}.graph.outputs.stage_id"
        )
        artifact_id = _parse_identifier(
            output["artifact_id"], f"profiles.{profile_name}.graph.outputs.artifact_id"
        )
        difficulty = output["difficulty"]
        if not isinstance(difficulty, str) or not difficulty.strip():
            raise BundleValidationError(
                f"profiles.{profile_name}.graph.outputs.difficulty must be a non-empty string"
            )
        outputs.append(
            ProfileGraphOutput(
                instrument=instrument,
                stage_id=stage_id,
                artifact_id=artifact_id,
                difficulty=difficulty,
            )
        )
    if len({output.instrument for output in outputs}) != len(outputs):
        raise BundleValidationError(
            f"profiles.{profile_name}.graph.outputs must have one terminal output per instrument"
        )
    return ProfileGraph(stages=stages, outputs=tuple(outputs))


def _parse_component(root: Path, name: str, value: object) -> ModelComponent:
    if not isinstance(value, dict):
        raise BundleValidationError(f"components.{name} must be an object")
    allowed = {
        "checkpoint",
        "config",
        "sha256",
        "byte_length",
        "config_sha256",
        "config_byte_length",
        "required",
        "architecture",
        "preprocessing",
    }
    unknown = set(value) - allowed
    if unknown:
        raise BundleValidationError(
            f"components.{name} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    sha256 = value.get("sha256")
    if sha256 is not None and (
        not isinstance(sha256, str)
        or len(sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in sha256)
    ):
        raise BundleValidationError(
            f"components.{name}.sha256 must be a lowercase SHA-256 hex digest"
        )
    if sha256 is not None and value.get("checkpoint") is None:
        raise BundleValidationError(f"components.{name}.sha256 requires a checkpoint path")
    byte_length = value.get("byte_length")
    if byte_length is not None and (
        not isinstance(byte_length, int) or isinstance(byte_length, bool) or byte_length < 0
    ):
        raise BundleValidationError(f"components.{name}.byte_length must be a non-negative integer")
    if byte_length is not None and value.get("checkpoint") is None:
        raise BundleValidationError(f"components.{name}.byte_length requires a checkpoint path")
    config_sha256 = value.get("config_sha256")
    if config_sha256 is not None and (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in config_sha256)
        or value.get("config") is None
    ):
        raise BundleValidationError(
            f"components.{name}.config_sha256 requires a config path and SHA-256"
        )
    config_byte_length = value.get("config_byte_length")
    if config_byte_length is not None and (
        not isinstance(config_byte_length, int)
        or isinstance(config_byte_length, bool)
        or config_byte_length < 0
        or value.get("config") is None
    ):
        raise BundleValidationError(
            f"components.{name}.config_byte_length requires a config path and non-negative integer"
        )
    required = value.get("required", True)
    if not isinstance(required, bool):
        raise BundleValidationError(f"components.{name}.required must be a boolean")
    architecture = value.get("architecture")
    if architecture is not None and (not isinstance(architecture, str) or not architecture.strip()):
        raise BundleValidationError(f"components.{name}.architecture must be a non-empty string")
    preprocessing = value.get("preprocessing")
    if preprocessing is not None and (
        not isinstance(preprocessing, str) or not preprocessing.strip()
    ):
        raise BundleValidationError(f"components.{name}.preprocessing must be a non-empty string")
    return ModelComponent(
        name=name,
        root=root,
        checkpoint=_resolve_relative_path(root, value.get("checkpoint"), "checkpoint", name),
        config=_resolve_relative_path(root, value.get("config"), "config", name),
        sha256=sha256,
        byte_length=byte_length,
        config_sha256=config_sha256,
        config_byte_length=config_byte_length,
        required=required,
        architecture=architecture,
        preprocessing=preprocessing,
    )


def _parse_profile(root: Path, name: str, value: object) -> InferenceProfile:
    if not isinstance(value, dict):
        raise BundleValidationError(f"profiles.{name} must be an object")
    allowed = {
        "capability",
        "instruments",
        "required_components",
        "required_companions",
        "difficulty_policies",
        "configuration",
        "configuration_sha256",
        "configuration_byte_length",
        "graph",
    }
    unknown = set(value) - allowed
    required = {"capability", "instruments", "required_components", "difficulty_policies"}
    missing = required - set(value)
    if unknown:
        raise BundleValidationError(
            f"profiles.{name} has unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise BundleValidationError(
            f"profiles.{name} missing field(s): {', '.join(sorted(missing))}"
        )
    capability = _parse_capability(value["capability"], f"profiles.{name}.capability")
    instruments = value["instruments"]
    required_components = value["required_components"]
    difficulty_policies = value["difficulty_policies"]
    required_companions = value.get("required_companions", [])
    for field, candidate in (("difficulty_policies", difficulty_policies),):
        if (
            not isinstance(candidate, list)
            or not candidate
            or not all(isinstance(item, str) and item.strip() for item in candidate)
            or len(set(candidate)) != len(candidate)
        ):
            raise BundleValidationError(
                f"profiles.{name}.{field} must be a unique non-empty string list"
            )
    if any(
        policy not in {"expert_only", "deterministic-v1", "evaluation_only"}
        and not policy.startswith("learned:")
        for policy in difficulty_policies
    ):
        raise BundleValidationError(
            f"profiles.{name}.difficulty_policies must use expert_only, deterministic-v1, or learned:<id>"
        )
    instrument_ids = _parse_identifier_list(
        instruments, f"profiles.{name}.instruments", allow_empty=False
    )
    required_component_ids = _parse_identifier_list(
        required_components, f"profiles.{name}.required_components", allow_empty=False
    )
    required_companion_ids = _parse_identifier_list(
        required_companions, f"profiles.{name}.required_companions"
    )
    configuration = _resolve_relative_path(
        root, value.get("configuration"), "configuration", f"profiles.{name}"
    )
    configuration_sha256 = value.get("configuration_sha256")
    configuration_byte_length = value.get("configuration_byte_length")
    if configuration is not None and (
        not isinstance(configuration_sha256, str)
        or len(configuration_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in configuration_sha256)
        or not isinstance(configuration_byte_length, int)
        or configuration_byte_length < 0
    ):
        raise BundleValidationError(
            f"profiles.{name}.configuration requires sha256 and non-negative byte_length"
        )
    if configuration is None and (
        configuration_sha256 is not None or configuration_byte_length is not None
    ):
        raise BundleValidationError(
            f"profiles.{name}.configuration_sha256 and configuration_byte_length require configuration"
        )
    graph = _parse_profile_graph(value["graph"], name) if "graph" in value else None
    return InferenceProfile(
        profile_id=name,
        capability=capability,
        instruments=instrument_ids,
        required_components=required_component_ids,
        required_companions=required_companion_ids,
        difficulty_policies=tuple(difficulty_policies),
        configuration=configuration,
        configuration_sha256=configuration_sha256,
        configuration_byte_length=configuration_byte_length,
        graph=graph,
    )


def load_model_bundle(path: str | Path, *, check_files: bool = False) -> ModelBundle:
    """Load a manifest file or a directory containing one.

    ``check_files`` verifies declared paths but does not deserialize model
    weights, so it is safe for installation and editor discovery flows.
    """
    candidate = Path(path).expanduser().resolve()
    manifest_path = (
        candidate if candidate.name == MANIFEST_FILENAME else candidate / MANIFEST_FILENAME
    )
    if not manifest_path.is_file():
        raise BundleValidationError(f"model bundle manifest not found: {manifest_path}")
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise BundleValidationError(f"invalid JSON in {manifest_path}: {error}") from error
    if not isinstance(raw, dict):
        raise BundleValidationError("bundle manifest must be a JSON object")

    required = {"schema_version", "model_id", "compatibility", "components"}
    allowed = {*required, "companions", "profiles"}
    unknown = set(raw) - allowed
    missing = required - set(raw)
    if unknown:
        raise BundleValidationError(
            f"bundle manifest has unknown field(s): {', '.join(sorted(unknown))}"
        )
    if missing:
        raise BundleValidationError(
            f"bundle manifest missing field(s): {', '.join(sorted(missing))}"
        )
    if not isinstance(raw["schema_version"], int):
        raise BundleValidationError("schema_version must be an integer")
    model_id = _parse_identifier(raw["model_id"], "model_id")
    if not isinstance(raw["compatibility"], dict):
        raise BundleValidationError("compatibility must be an object")
    if (
        "strum_revision" in raw["compatibility"]
        and source_revision_identity(raw["compatibility"]["strum_revision"]) is None
    ):
        raise BundleValidationError(
            "compatibility.strum_revision must be a safe Git revision identity"
        )
    if (
        "strum_source_dirty" in raw["compatibility"]
        and raw["compatibility"]["strum_source_dirty"] is not None
        and not isinstance(raw["compatibility"]["strum_source_dirty"], bool)
    ):
        raise BundleValidationError("compatibility.strum_source_dirty must be a boolean or null")
    if not isinstance(raw["components"], dict) or not raw["components"]:
        raise BundleValidationError("components must be a non-empty object")
    if "profiles" in raw and not isinstance(raw["profiles"], dict):
        raise BundleValidationError("profiles must be an object")
    if "companions" in raw and not isinstance(raw["companions"], dict):
        raise BundleValidationError("companions must be an object")

    root = manifest_path.parent.resolve()
    components = {
        _parse_identifier(name, "components key"): _parse_component(
            root, _parse_identifier(name, "components key"), value
        )
        for name, value in raw["components"].items()
    }
    companions = {
        _parse_identifier(name, "companions key"): _parse_companion(
            _parse_identifier(name, "companions key"), value
        )
        for name, value in raw.get("companions", {}).items()
    }
    profiles = {
        _parse_identifier(name, "profiles key"): _parse_profile(
            root, _parse_identifier(name, "profiles key"), value
        )
        for name, value in raw.get("profiles", {}).items()
    }
    bundle = ModelBundle(
        root=root,
        model_id=model_id,
        schema_version=raw["schema_version"],
        compatibility=raw["compatibility"],
        components=components,
        companions=companions,
        profiles=profiles,
        manifest_path=manifest_path,
    )
    errors = bundle.validate(check_files=check_files)
    if errors:
        raise BundleValidationError("; ".join(errors))
    return bundle


def legacy_model_bundle(root: Path = PROJECT_ROOT) -> ModelBundle:
    """Expose the established repository layout through the new resolver API."""
    root = root.resolve()
    component_paths = {
        "drums.v14_onset": {"checkpoint": "checkpoints/drums_v14/best.pt"},
        "drums.ensemble.v2": {
            "checkpoint": "checkpoints/onset_classifier/best_f1.pt",
            "config": "configs/onset_classifier.yaml",
        },
        "drums.ensemble.v4": {
            "checkpoint": "checkpoints/onset_classifier_v4/best_f1.pt",
            "config": "configs/onset_classifier_v4.yaml",
        },
        "drums.ensemble.v6": {
            "checkpoint": "checkpoints/onset_classifier_v6/best_f1.pt",
            "config": "configs/onset_classifier_v6.yaml",
        },
        "drums.ensemble.v12c": {
            "checkpoint": "checkpoints/onset_classifier_v12_clean/best_f1.pt",
            "config": "configs/onset_classifier_v12_clean.yaml",
        },
        "drums.ensemble.v15": {
            "checkpoint": "checkpoints/onset_classifier_v15/best_f1.pt",
            "config": "configs/onset_classifier_v15.yaml",
        },
        "drums.ensemble.v16": {
            "checkpoint": "checkpoints/onset_classifier_v16/best_f1.pt",
            "config": "configs/onset_classifier_v16.yaml",
        },
        "drums.ensemble.v17": {
            "checkpoint": "checkpoints/onset_classifier_v17/best_f1.pt",
            "config": "configs/onset_classifier_v17.yaml",
        },
        "guitar.onset": {"checkpoint": "checkpoints/guitar_v2/guitar_v2_onset/best.pt"},
    }
    components = {
        name: _parse_component(root, name, value) for name, value in component_paths.items()
    }
    return ModelBundle(
        root=root,
        model_id="legacy-repository-layout",
        schema_version=MANIFEST_SCHEMA_VERSION,
        compatibility={
            "manifest_schema": MANIFEST_SCHEMA_VERSION,
            "strum_version": f">={__version__}",
        },
        components=components,
        companions={},
        profiles={},
        legacy=True,
    )


def get_active_bundle() -> ModelBundle:
    """Return the explicitly selected bundle or the backwards-compatible default.

    Set ``STRUM_MODEL_BUNDLE`` to either a bundle directory or its manifest
    file.  An explicit invalid selection fails early instead of silently using
    a different model set.
    """
    selected = os.environ.get("STRUM_MODEL_BUNDLE")
    return load_model_bundle(selected) if selected else legacy_model_bundle()


def discover_model_bundles(root: str | Path) -> list[ModelBundle]:
    """Find and validate manifests below a user-selected model directory.

    Invalid manifests are intentionally omitted: callers can independently
    call :func:`load_model_bundle` to display their validation error.
    """
    base = Path(root).expanduser()
    manifests: Iterable[Path]
    if base.name == MANIFEST_FILENAME:
        manifests = (base,)
    else:
        manifests = base.glob(f"*/{MANIFEST_FILENAME}")
    bundles: list[ModelBundle] = []
    for manifest in sorted(manifests):
        try:
            bundles.append(load_model_bundle(manifest))
        except BundleValidationError:
            continue
    return bundles


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect STRUM model bundles without loading weights."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="validate one bundle manifest")
    validate.add_argument("path", help="bundle directory or manifest path")
    validate.add_argument(
        "--check-files", action="store_true", help="require declared checkpoint/config files"
    )
    validate.add_argument(
        "--verify-hashes", action="store_true", help="SHA-256 declared checkpoint files"
    )
    listing = commands.add_parser("list", help="list valid child bundles in a directory")
    listing.add_argument("path", help="directory containing bundle directories")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            check_files = args.check_files or args.verify_hashes
            bundle = load_model_bundle(args.path, check_files=check_files)
            errors = bundle.validate(check_files=check_files, verify_hashes=args.verify_hashes)
            if errors:
                raise BundleValidationError("; ".join(errors))
            print(f"valid: {bundle.model_id} ({bundle.manifest_path})")
            for status in bundle.compatibility_status():
                print(f"status: {status}")
        else:
            for bundle in discover_model_bundles(args.path):
                print(f"{bundle.model_id}\t{bundle.manifest_path}")
    except BundleValidationError as error:
        print(f"invalid: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
