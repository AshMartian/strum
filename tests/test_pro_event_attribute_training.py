from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def _trainer_module():
    path = Path(__file__).parents[1] / "scripts" / "train_pro_event_attributes.py"
    spec = importlib.util.spec_from_file_location("train_pro_event_attributes", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_cache(root: Path, task_kind: str) -> None:
    root.mkdir()
    for split, value in (("train", 1.0), ("val", 2.0)):
        np.save(root / f"{split}_logmel.npy", np.full((2, 128, 22), value, dtype=np.float16))
        if task_kind == "pro_keys":
            row = {
                "split": split,
                "target_language": "pitch_channel_range_shift/v1",
                "event_tick": 12,
                "range_shifts": [{"tick": 4, "anchor": "C"}],
                "events": [{"pitch": 60, "channel": 1, "tick": 12, "duration_ticks": 120}],
            }
        else:
            row = {
                "split": split,
                "target_language": "string_fret_technique/v1",
                "track_variant": "22_fret",
                "events": [
                    {
                        "string": 2,
                        "fret": 22,
                        "technique": "normal",
                        "tick": 12,
                        "duration_ticks": 120,
                    }
                ],
            }
        (root / f"{split}_targets.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for _ in range(2)), encoding="utf-8"
        )


@pytest.mark.parametrize("task_kind", ["pro_guitar", "pro_bass", "pro_keys"])
def test_exact_pro_candidate_trains_only_known_event_attributes(
    tmp_path: Path, task_kind: str
) -> None:
    module = _trainer_module()
    cache = tmp_path / "cache"
    _write_cache(cache, task_kind)
    checkpoints = tmp_path / "checkpoints"

    metrics = module.train_pro_event_attributes(
        module.TrainingSettings(
            cache_dir=cache,
            checkpoint_dir=checkpoints,
            task_kind=task_kind,
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            device="cpu",
            max_train_batches=1,
            max_val_batches=1,
            seed=3,
            channels=4,
        )
    )

    assert set(metrics) == {
        "train_loss",
        "val_loss",
        "val_known_event_token_f1",
        "val_known_event_state_accuracy",
        "val_known_event_exact_accuracy",
    }
    assert (checkpoints / "best.pt").is_file()
    assert (checkpoints / "history.json").is_file()


def test_string_candidate_rejects_standard_track_with_22_fret_target(tmp_path: Path) -> None:
    module = _trainer_module()
    cache = tmp_path / "cache"
    _write_cache(cache, "pro_guitar")
    label = json.loads((cache / "train_targets.jsonl").read_text().splitlines()[0])
    label["track_variant"] = "standard"
    (cache / "train_targets.jsonl").write_text(
        json.dumps(label) + "\n" + json.dumps(label) + "\n", encoding="utf-8"
    )

    with pytest.raises(module.ProEventAttributeError, match="standard Pro string"):
        module.ProEventDataset(cache, "train", "pro_guitar")
