from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from src.worker import PIPELINES, prepare_dataset_request, run_training_request


def _asset(root: Path, content: bytes, filename: str) -> dict[str, object]:
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


def _catalog_with_train_and_val(root: Path) -> None:
    records: list[dict[str, object]] = []
    # The catalog adapter's source-ID split is deterministic. This sample is
    # deliberately larger than a two-split fixture so the worker exercises its
    # genuine non-empty train/validation precondition.
    for index in range(32):
        source_id = f"octave-src-{index:08x}"
        records.append(
            {
                "source_id": source_id,
                "import": {"kind": "sng", "adapter_version": "octave-sng/1", "warnings": []},
                "rights": {
                    "training_use": "allowed",
                    "provenance": "Reviewed",
                    "license": "test-only",
                },
                "metadata": {"name": f"Fixture {index}"},
                "chart": {
                    "notes_midi": _asset(root, f"midi-{index}".encode(), "notes.mid"),
                    "instruments": {
                        "guitar": {
                            "status": "present",
                            "difficulties": ["expert"],
                            "track_names": ["PART GUITAR"],
                        }
                    },
                },
                "audio": {"guitar": _asset(root, f"audio-{index}".encode(), "guitar.ogg")},
            }
        )
    (root / "records.jsonl").write_text("\n".join(json.dumps(record) for record in records) + "\n")
    (root / "catalog.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "octave-song-source-catalog/v1",
                "catalog_id": "guitar-training-fixture",
                "records": "records.jsonl",
            }
        )
    )


def test_guitar_worker_trains_from_catalog_task_view_and_packages_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    _catalog_with_train_and_val(tmp_path)
    task_view = tmp_path / "views" / "guitar.json"
    prepare_request = tmp_path / "prepare.json"
    prepare_request.write_text(
        json.dumps(
            {
                "catalog_root": str(tmp_path),
                "pipeline_id": "guitar.onset-fret/v1",
                "output": str(task_view),
                "options": {},
            }
        )
    )
    prepare_dataset_request(prepare_request)

    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> None:
        commands.append(command)
        if command[1].endswith("train_guitar_v1.py"):
            config = yaml.safe_load(Path(command[command.index("--config") + 1]).read_text())
            checkpoint_dir = Path(config["paths"]["checkpoint_dir"])
            for subdir in ("guitar_v1_onset", "guitar_v1_fret"):
                stage = checkpoint_dir / subdir
                stage.mkdir(parents=True, exist_ok=True)
                (stage / "best.pt").write_bytes(f"{subdir}-weights".encode())
                (stage / "history.json").write_text(
                    json.dumps([{"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_f1": 0.8}])
                )

    monkeypatch.setattr("src.guitar_worker_training._run_script", fake_run)
    output = tmp_path / "experiments" / "guitar-v1"
    train_request = tmp_path / "train.json"
    train_request.write_text(
        json.dumps(
            {
                "pipeline_id": "guitar.onset-fret/v1",
                "task_view": str(task_view),
                "output": str(output),
                "options": {
                    "model_id": "catalog-guitar-v1",
                    "catalog_root": str(tmp_path),
                    "epochs": 1,
                    "batch_size": 2,
                    "device": "cpu",
                },
            }
        )
    )

    result = run_training_request(train_request)

    assert result["status"] == "completed"
    assert result["pipeline_id"] == "guitar.onset-fret/v1"
    assert result["model_id"] == "catalog-guitar-v1"
    assert [component["id"] for component in result["components"]] == [
        "guitar.onset",
        "guitar.fret",
    ]
    assert "preprocess_guitar_windows.py" in commands[0][1]
    assert "--catalog-root" in commands[0]
    assert "train_guitar_v1.py" in commands[1][1]
    assert "both" in commands[1]
    bundle = output / "bundle"
    manifest = (bundle / "strum-model-bundle.json").read_text()
    experiment = (output / "experiment.json").read_text()
    assert str(tmp_path) not in manifest
    assert str(tmp_path) not in experiment
    assert json.loads(experiment)["task_view"]["catalog_id"] == "guitar-training-fixture"


def test_guitar_pipeline_advertises_a_strict_worker_training_schema() -> None:
    descriptor = next(item for item in PIPELINES if item.id == "guitar.onset-fret/v1")

    assert descriptor.training_status == "available"
    assert descriptor.train_schema is not None
    assert descriptor.train_schema["required"] == ["model_id", "catalog_root"]
    assert descriptor.checkpoint_outputs == ("guitar.onset", "guitar.fret")
