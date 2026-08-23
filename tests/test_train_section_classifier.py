from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from scripts.train_section_classifier import main


def _write_split(cache_dir: Path, split: str, labels: list[int]) -> None:
    # Distinct values keep batch-norm training finite while remaining tiny.
    features = np.stack(
        [
            np.full((128, 87), float(label + index + 1), dtype=np.float32)
            for index, label in enumerate(labels)
        ]
    )
    np.save(cache_dir / f"{split}_section_mel.npy", features)
    np.save(cache_dir / f"{split}_section_label.npy", np.asarray(labels, dtype=np.int8))


def test_section_trainer_reports_best_checkpoint_test_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_split(cache_dir, "train", [0, 1, 2, 3, 4, 5])
    _write_split(cache_dir, "val", [0, 1])
    _write_split(cache_dir, "test", [2, 3])
    checkpoint_dir = tmp_path / "checkpoints"
    metrics = tmp_path / "metrics.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_section_classifier.py",
            "--cache-dir",
            str(cache_dir),
            "--ckpt-dir",
            str(checkpoint_dir),
            "--epochs",
            "1",
            "--batch-size",
            "6",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--evaluate-split",
            "test",
            "--metrics-out",
            str(metrics),
        ],
    )

    assert main() == 0
    result = json.loads(metrics.read_text(encoding="utf-8"))
    assert result["held_out_evaluation"]["status"] == "completed"
    assert result["held_out_evaluation"]["split"] == "test"
    assert result["held_out_evaluation"]["record_count"] == 2
