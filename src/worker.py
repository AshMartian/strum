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
from typing import Any

from src import PROJECT_ROOT, __version__
from src.catalog_chart_pairs import CatalogChartPairOptions, prepare_catalog_chart_pairs
from src.catalog_drums_manifest import build_drums_manifest, write_drums_manifest
from src.catalog_guitar_manifest import build_guitar_manifest, write_guitar_manifest
from src.catalog_task_manifest import (
    PIPELINE_IDS as CATALOG_TASK_PIPELINES,
)
from src.catalog_task_manifest import (
    build_catalog_task_manifest,
    write_catalog_task_manifest,
)
from src.model_bundle import BundleValidationError, InferenceProfile, ModelBundle, load_model_bundle
from src.song_source_catalog import CatalogValidationError, load_catalog

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
    preparation_status: str
    training_status: str

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
        status="catalog_ready",
        preparation_status="available",
        training_status="script_only",
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
        checkpoint_outputs=("chart_transform",),
        inference_capability="difficulty.transform/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="available",
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
        checkpoint_outputs=("drums_onset",),
        inference_capability="drums.audio_to_chart/v1",
        status="catalog_ready",
        preparation_status="available",
        training_status="script_only",
    ),
    *(
        PipelineDescriptor(
            id=pipeline_id,
            display_name=task_kind.replace("_", " ").title(),
            kind="derived_labels"
            if task_kind.startswith(("fret_mapper", "section_"))
            else "chart_to_chart",
            version=1,
            catalog_requirements={
                "instrument": task_kind.replace("fret_mapper_", "").replace("section_", ""),
                "difficulties": ["expert"],
                "audio_policy": "task-specific managed role with mix fallback",
            },
            checkpoint_outputs=(task_kind,),
            inference_capability=None,
            status="catalog_ready",
            preparation_status="available",
            training_status="planned",
        )
        for task_kind, pipeline_id in sorted(CATALOG_TASK_PIPELINES.items())
    ),
)


class WorkerRequestError(ValueError):
    """Raised for an invalid host-to-worker request before any task is written."""


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
    capabilities = [
        "pipeline_discovery",
        "catalog_inspect",
        "dataset_prepare",
        "model_bundle_preflight",
        "checkpoint_inspect",
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
        "pipelines": [
            pipeline.id for pipeline in PIPELINES if pipeline.preparation_status == "available"
        ],
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


def _pipeline_by_id(pipeline_id: str) -> PipelineDescriptor:
    for pipeline in PIPELINES:
        if pipeline.id == pipeline_id:
            return pipeline
    raise WorkerRequestError("unknown pipeline_id")


def inspect_catalog(catalog_root: str | Path, pipeline_id: str | None = None) -> dict[str, object]:
    """Validate a catalog and return a path-free capability summary for OCTAVE."""
    if pipeline_id is not None:
        _pipeline_by_id(pipeline_id)
    catalog = load_catalog(catalog_root)
    allowed = sum(record.training_use == "allowed" for record in catalog.records)
    return {
        "status": "ready",
        "catalog_id": catalog.catalog_id,
        "record_count": len(catalog.records),
        "allowed_record_count": allowed,
        "pipeline_id": pipeline_id,
    }


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
        permitted = {"audio_role", "fallback_audio_role", "required_difficulty"}
        if set(options) - permitted:
            raise WorkerRequestError("unsupported Guitar preparation option")
        manifest = build_guitar_manifest(catalog_root, **options)
        written = write_guitar_manifest(output, manifest)
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
            "validation_fraction",
            "dataset_id",
            "overwrite",
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
    expected = {"pipeline_id", "task_view", "output", "options"}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise WorkerRequestError("training request has unsupported fields")
    if not all(
        isinstance(raw[key], str) and raw[key] for key in ("pipeline_id", "task_view", "output")
    ):
        raise WorkerRequestError(
            "training request locations and pipeline_id must be non-empty strings"
        )
    if not isinstance(raw["options"], dict):
        raise WorkerRequestError("training options must be an object")
    return raw


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
        "seed",
        "validation_fraction",
        "lane_count",
        "alignment_tolerance_ms",
        "hidden_dim",
        "learning_rate",
        "epochs",
        "device",
        "audio_feature_mode",
        "audio_sample_rate",
        "audio_window_ms",
        "audio_max_duration_seconds",
        "strum_revision",
    }
    if set(options) - permitted or not isinstance(options.get("model_id"), str):
        raise WorkerRequestError("invalid chart-transform training options")
    try:
        dataset = json.loads(Path(request["task_view"]).read_text(encoding="utf-8"))
        config = TrainingConfig.from_mapping(
            {
                "dataset_manifest": request["task_view"],
                "output_dir": request["output"],
                "model_id": options["model_id"],
                "source_difficulty": dataset["source_difficulty"],
                "target_difficulty": dataset["target_difficulty"],
                **{key: value for key, value in options.items() if key != "model_id"},
            }
        )
        result = train(config)
        preflight = preflight_bundle(result["bundle_dir"])
    except (KeyError, OSError, TypeError, ValueError, DatasetValidationError) as error:
        raise WorkerRequestError("chart-transform training request failed validation") from error
    return {
        "status": "completed",
        "pipeline_id": pipeline_id,
        "model_id": preflight["model_id"],
        "bundle_name": Path(result["bundle_dir"]).name,
        "manifest_sha256": preflight["manifest_sha256"],
        "components": preflight["components"],
        "validation": result["metrics"]["validation"],
    }


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
    catalog_inspect.add_argument("--json", action="store_true")
    dataset = commands.add_parser("dataset", help="prepare STRUM task views")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    prepare = dataset_commands.add_parser("prepare", help="materialize one catalog task view")
    prepare.add_argument("--request", type=Path, required=True)
    prepare.add_argument("--json", action="store_true")
    training = commands.add_parser("train", help="run worker-managed training")
    training_commands = training.add_subparsers(dest="training_command", required=True)
    training_run = training_commands.add_parser("run", help="run one synchronous training job")
    training_run.add_argument("--request", type=Path, required=True)
    training_run.add_argument("--json", action="store_true")
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
        if args.command == "catalog" and args.catalog_command == "inspect":
            _print_json(inspect_catalog(args.catalog_root, args.pipeline))
            return 0
        if args.command == "dataset" and args.dataset_command == "prepare":
            _print_json(prepare_dataset_request(args.request))
            return 0
        if args.command == "train" and args.training_command == "run":
            _print_json(run_training_request(args.request))
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
