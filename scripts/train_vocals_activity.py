#!/usr/bin/env python3
"""Train the bounded STRUM lead-vocal activity + pitch experiment.

The model predicts two deliberately narrow frame-level targets produced by
``preprocess_vocals_frames.py``: whether a pitched lead vocal is active, and
its MIDI pitch in the Clone Hero vocal range.  It does not predict words,
syllable alignment, talkies, harmony tracks, or a playable vocal chart.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.preprocess_vocals_frames import N_MELS, PITCH_CLASS_COUNT, SEGMENT_FRAMES  # noqa: E402


class VocalFrameDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Memory-map Vocal worker cache files without retaining catalog paths."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        self.mel = np.load(cache_dir / f"{split}_mel.npy", mmap_mode="r")
        self.activity = np.load(cache_dir / f"{split}_activity.npy", mmap_mode="r")
        self.pitch = np.load(cache_dir / f"{split}_pitch.npy", mmap_mode="r")
        if (
            self.mel.ndim != 3
            or self.mel.shape[1:] != (N_MELS, SEGMENT_FRAMES)
            or self.activity.shape != self.pitch.shape
            or self.activity.shape != (self.mel.shape[0], SEGMENT_FRAMES)
        ):
            raise ValueError("Vocal cache has an unsupported shape")

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.mel[index].astype(np.float32)).unsqueeze(0),
            torch.from_numpy(self.activity[index].astype(np.float32)),
            torch.from_numpy(self.pitch[index].astype(np.int64)),
        )


class VocalFrameCNN(nn.Module):
    """Compact time-preserving convolutional network for bounded evaluation."""

    def __init__(self, *, channels: int = 48) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.activity = nn.Conv1d(channels, 1, kernel_size=1)
        self.pitch = nn.Conv1d(channels, PITCH_CLASS_COUNT, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(mel).mean(dim=2)
        return self.activity(features).squeeze(1), self.pitch(features)


@dataclass(frozen=True)
class TrainingSettings:
    cache_dir: Path
    checkpoint_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    max_train_batches: int
    max_val_batches: int
    seed: int


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _metrics(
    model: VocalFrameCNN, loader: DataLoader, device: str, max_batches: int
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = true_positive = false_positive = false_negative = 0
    pitch_correct = pitch_total = 0
    with torch.no_grad():
        for mel, activity, pitch in loader:
            mel, activity, pitch = mel.to(device), activity.to(device), pitch.to(device)
            activity_logits, pitch_logits = model(mel)
            activity_loss = functional.binary_cross_entropy_with_logits(activity_logits, activity)
            pitch_loss = functional.cross_entropy(pitch_logits, pitch, ignore_index=0)
            total_loss += float((activity_loss + pitch_loss).item())
            predicted_activity = torch.sigmoid(activity_logits) >= 0.5
            true_active = activity >= 0.5
            true_positive += int((predicted_activity & true_active).sum().item())
            false_positive += int((predicted_activity & ~true_active).sum().item())
            false_negative += int((~predicted_activity & true_active).sum().item())
            pitch_prediction = pitch_logits.argmax(dim=1)
            active_mask = pitch > 0
            pitch_correct += int((pitch_prediction[active_mask] == pitch[active_mask]).sum().item())
            pitch_total += int(active_mask.sum().item())
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return {
        "loss": total_loss / max(batches, 1),
        "activity_f1": _f1(true_positive, false_positive, false_negative),
        "pitch_accuracy": pitch_correct / max(pitch_total, 1),
    }


def train_vocals_activity(settings: TrainingSettings) -> dict[str, float]:
    """Train from a complete cache and write a tensor-only best checkpoint."""
    if settings.epochs < 1 or settings.batch_size < 1 or settings.learning_rate <= 0:
        raise ValueError("Vocal training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train_dataset = VocalFrameDataset(settings.cache_dir, "train")
    val_dataset = VocalFrameDataset(settings.cache_dir, "val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Vocal training needs non-empty train and val caches")
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size)
    model = VocalFrameCNN().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for mel, activity, pitch in train_loader:
            mel, activity, pitch = (
                mel.to(settings.device),
                activity.to(settings.device),
                pitch.to(settings.device),
            )
            activity_logits, pitch_logits = model(mel)
            loss = functional.binary_cross_entropy_with_logits(
                activity_logits, activity
            ) + functional.cross_entropy(pitch_logits, pitch, ignore_index=0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if settings.max_train_batches and batches >= settings.max_train_batches:
                break
        val = _metrics(model, val_loader, settings.device, settings.max_val_batches)
        item: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_loss": val["loss"],
            "val_activity_f1": val["activity_f1"],
            "val_pitch_accuracy": val["pitch_accuracy"],
        }
        history.append(item)
        torch.save({"state_dict": model.state_dict()}, settings.checkpoint_dir / "last.pt")
        if val["activity_f1"] > best_f1:
            best_f1 = val["activity_f1"]
            torch.save({"state_dict": model.state_dict()}, settings.checkpoint_dir / "best.pt")
    (settings.checkpoint_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {key: float(value) for key, value in history[-1].items() if key != "epoch"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    args = parser.parse_args()
    try:
        train_vocals_activity(
            TrainingSettings(
                cache_dir=args.cache_dir,
                checkpoint_dir=args.checkpoint_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
                seed=args.seed,
            )
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
