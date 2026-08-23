"""Package verified catalog-worker Drums V2 experiments for evaluation.

The result is a portable, hash-checked bundle containing exactly one Stage-2
classifier.  It deliberately declares an evaluation-only capability: V14 owns
the direct onset/velocity-to-chart execution contract.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from src import __version__
from src.drums_onset_training import EXPERIMENT_FORMAT
from src.inference.drums_onset_classifier_profile import (
    ARCHITECTURE,
    CAPABILITY,
    CLASS_NAMES,
    COARSE_MEL_SHAPE,
    CONTEXT_SHAPE,
    EXECUTION_SCOPE,
    FINE_MEL_SHAPE,
    MODEL_CONFIG_FORMAT,
    PREPROCESSING,
    PROFILE_FORMAT,
    normalize_v2_model_parameters,
)
from src.model_bundle import MANIFEST_FILENAME, BundleValidationError, load_model_bundle

PACKAGE_LINEAGE_FORMAT = "strum-drums-onset-evaluation-package-lineage/v1"
PROFILE_ID = "drums-onset-classifier-evaluation"
COMPONENT_ID = "drums.onset_classifier.v2"


class DrumsProfilePackagingError(ValueError):
    """Raised when a worker experiment cannot safely become an evaluator bundle."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, error_message: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DrumsProfilePackagingError(error_message) from error
    if not isinstance(value, dict):
        raise DrumsProfilePackagingError(error_message)
    return value


def _safe_child(root: Path, name: object, label: str) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise DrumsProfilePackagingError(f"Drums experiment {label} is not a safe relative filename")
    child = (root / name).resolve()
    try:
        child.relative_to(root)
    except ValueError as error:
        raise DrumsProfilePackagingError(
            f"Drums experiment {label} escapes its experiment root"
        ) from error
    return child


def _load_experiment(experiment_root: Path) -> tuple[dict[str, Any], Path, Path, dict[str, object]]:
    ledger_path = experiment_root / "experiment.json"
    ledger = _read_json(ledger_path, "Drums experiment ledger is unreadable")
    required = {"format", "pipeline", "model", "task_view", "preprocessing", "training", "checkpoint", "metrics"}
    if set(ledger) != required or ledger.get("format") != EXPERIMENT_FORMAT:
        raise DrumsProfilePackagingError("Drums experiment ledger has unsupported fields")
    pipeline, model, training, checkpoint, metrics = (
        ledger.get("pipeline"), ledger.get("model"), ledger.get("training"), ledger.get("checkpoint"), ledger.get("metrics")
    )
    if (
        pipeline != {"id": "drums.onset-classifier/v1", "version": 1}
        or not isinstance(model, dict)
        or model.get("profile") != "onset_classifier_v2"
        or model.get("architecture") != ARCHITECTURE
        or not isinstance(model.get("id"), str)
        or not model["id"]
        or not isinstance(training, dict)
        or not isinstance(checkpoint, dict)
        or not isinstance(metrics, dict)
    ):
        raise DrumsProfilePackagingError("Drums experiment ledger is not a supported V2 worker result")
    config_path = _safe_child(experiment_root, training.get("config_name"), "config_name")
    checkpoint_path = _safe_child(experiment_root, checkpoint.get("name"), "checkpoint.name")
    if (
        not config_path.is_file()
        or training.get("config_sha256") != _sha256(config_path)
        or not checkpoint_path.is_file()
        or checkpoint.get("sha256") != _sha256(checkpoint_path)
        or checkpoint.get("byte_length") != checkpoint_path.stat().st_size
        or checkpoint.get("format") != "torch-training-checkpoint/v1"
        or checkpoint.get("deployment_status") != "requires_profile_packaging"
    ):
        raise DrumsProfilePackagingError("Drums experiment assets do not match their ledger")
    try:
        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    except Exception as error:
        raise DrumsProfilePackagingError("Drums experiment model configuration is unreadable") from error
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise DrumsProfilePackagingError("Drums experiment model configuration has no model settings")
    parameters = normalize_v2_model_parameters(dict(config["model"]))
    return ledger, config_path, checkpoint_path, parameters


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def package_drums_onset_experiment(
    experiment_root: str | Path, output_dir: str | Path
) -> dict[str, object]:
    """Turn one complete catalog-worker V2 experiment into a safe evaluator bundle.

    ``output_dir`` must not already exist.  The atomic rename means callers do
    not discover a partially copied checkpoint as a selectable artifact.
    """
    source = Path(experiment_root).resolve()
    output = Path(output_dir).resolve()
    if not source.is_dir():
        raise DrumsProfilePackagingError("Drums experiment root is unavailable")
    if output.exists():
        raise DrumsProfilePackagingError("Drums evaluation bundle output already exists")
    ledger, _config_path, checkpoint_path, parameters = _load_experiment(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        checkpoint_target = temporary / "weights" / "onset-classifier-v2.pt"
        checkpoint_target.parent.mkdir(parents=True)
        shutil.copy2(checkpoint_path, checkpoint_target)
        model_config = {
            "schema_version": 1,
            "format": MODEL_CONFIG_FORMAT,
            "model_architecture": ARCHITECTURE,
            "preprocessing": PREPROCESSING,
            "model_parameters": parameters,
        }
        model_config_path = temporary / "configs" / "onset-classifier-v2.json"
        _write_json(model_config_path, model_config)
        profile_config = {
            "schema_version": 1,
            "format": PROFILE_FORMAT,
            "model_architecture": ARCHITECTURE,
            "preprocessing": PREPROCESSING,
            "execution_scope": EXECUTION_SCOPE,
            "class_names": list(CLASS_NAMES),
            "fine_mel_shape": list(FINE_MEL_SHAPE),
            "coarse_mel_shape": list(COARSE_MEL_SHAPE),
            "context_shape": list(CONTEXT_SHAPE),
            "output_contract": "sigmoid_8_class_probabilities/v1",
            "auto_chart_status": "not_supported",
        }
        profile_config_path = temporary / "profiles" / "drums-onset-classifier-evaluation.json"
        _write_json(profile_config_path, profile_config)
        lineage = {
            "format": PACKAGE_LINEAGE_FORMAT,
            "source_experiment_sha256": _sha256(source / "experiment.json"),
            "pipeline": ledger["pipeline"],
            "model": ledger["model"],
            "task_view": ledger["task_view"],
            "preprocessing": ledger["preprocessing"],
            "training": ledger["training"],
            "checkpoint": {
                "sha256": ledger["checkpoint"]["sha256"],
                "byte_length": ledger["checkpoint"]["byte_length"],
            },
            "metrics": ledger["metrics"],
            "deployment_status": "evaluation_only_not_auto_chart_deployable",
        }
        _write_json(temporary / "lineage.json", lineage)
        manifest = {
            "schema_version": 1,
            "model_id": ledger["model"]["id"],
            "compatibility": {"manifest_schema": 1, "strum_version": f">={__version__}"},
            "components": {
                COMPONENT_ID: {
                    "checkpoint": "weights/onset-classifier-v2.pt",
                    "sha256": _sha256(checkpoint_target),
                    "byte_length": checkpoint_target.stat().st_size,
                    "config": "configs/onset-classifier-v2.json",
                    "config_sha256": _sha256(model_config_path),
                    "config_byte_length": model_config_path.stat().st_size,
                    "architecture": ARCHITECTURE,
                    "preprocessing": PREPROCESSING,
                }
            },
            "profiles": {
                PROFILE_ID: {
                    "capability": CAPABILITY,
                    "instruments": ["drums"],
                    "required_components": [COMPONENT_ID],
                    "difficulty_policies": ["evaluation_only"],
                    "configuration": "profiles/drums-onset-classifier-evaluation.json",
                    "configuration_sha256": _sha256(profile_config_path),
                    "configuration_byte_length": profile_config_path.stat().st_size,
                }
            },
        }
        _write_json(temporary / MANIFEST_FILENAME, manifest)
        bundle = load_model_bundle(temporary, check_files=True)
        errors = bundle.validate(check_files=True, verify_hashes=True)
        if errors:
            raise BundleValidationError("; ".join(errors))
        os.replace(temporary, output)
    except (OSError, BundleValidationError) as error:
        raise DrumsProfilePackagingError("unable to package Drums evaluation bundle") from error
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return {
        "status": "packaged",
        "model_id": ledger["model"]["id"],
        "bundle_name": output.name,
        "profile_id": PROFILE_ID,
        "capability": CAPABILITY,
        "execution_scope": EXECUTION_SCOPE,
        "deployment_status": "evaluation_only_not_auto_chart_deployable",
        "manifest_sha256": _sha256(output / MANIFEST_FILENAME),
        "metrics": ledger["metrics"],
    }
