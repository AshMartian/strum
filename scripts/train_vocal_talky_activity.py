#!/usr/bin/env python3
"""Train only observed lead-Vocal pitchless/talky activity.

This emits no pitch, lyrics, phrase boundary, harmony, or playable-chart
decision.  Training fails closed when either train or validation has no source
note-96 span, because an all-negative split cannot evaluate this component.
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

from scripts.preprocess_vocals_frames import N_MELS, SEGMENT_FRAMES  # noqa: E402


class VocalTalkyDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, cache_dir: Path, split: str) -> None:
        self.mel = np.load(cache_dir / f"{split}_mel.npy", mmap_mode="r")
        self.talky = np.load(cache_dir / f"{split}_talky.npy", mmap_mode="r")
        if (
            self.mel.ndim != 3
            or self.mel.shape[1:] != (N_MELS, SEGMENT_FRAMES)
            or self.talky.shape != (self.mel.shape[0], SEGMENT_FRAMES)
        ):
            raise ValueError("Vocal talky cache has an unsupported shape")

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.mel[index].astype(np.float32)).unsqueeze(0),
            torch.from_numpy(self.talky[index].astype(np.float32)),
        )


class VocalTalkyCNN(nn.Module):
    """Small time-preserving binary target model."""

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

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.activity(self.features(mel).mean(dim=2)).squeeze(1)


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


def _f1(tp: int, fp: int, fn: int) -> float:
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _score(
    model: VocalTalkyCNN,
    loader: DataLoader,
    device: str,
    max_batches: int,
    pos_weight: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    batches = tp = fp = fn = 0
    with torch.no_grad():
        for mel, labels in loader:
            logits = model(mel.to(device))
            labels = labels.to(device)
            loss_total += float(
                functional.binary_cross_entropy_with_logits(
                    logits, labels, pos_weight=pos_weight
                ).item()
            )
            predicted, observed = torch.sigmoid(logits) >= 0.5, labels >= 0.5
            tp += int((predicted & observed).sum().item())
            fp += int((predicted & ~observed).sum().item())
            fn += int((~predicted & observed).sum().item())
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return {"loss": loss_total / max(batches, 1), "talky_activity_f1": _f1(tp, fp, fn)}


def train_vocal_talky_activity(settings: TrainingSettings) -> dict[str, float]:
    if settings.epochs < 1 or settings.batch_size < 1 or settings.learning_rate <= 0:
        raise ValueError("Vocal talky training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train, val = (
        VocalTalkyDataset(settings.cache_dir, "train"),
        VocalTalkyDataset(settings.cache_dir, "val"),
    )
    if not len(train) or not len(val):
        raise ValueError("Vocal talky training needs non-empty train and val caches")
    positives_train, positives_val = (
        int(np.count_nonzero(train.talky)),
        int(np.count_nonzero(val.talky)),
    )
    if not positives_train or not positives_val:
        raise ValueError(
            "Vocal talky training requires observed note-96 spans in both train and val splits"
        )
    pos_weight = torch.tensor(
        [(train.talky.size - positives_train) / positives_train], device=settings.device
    )
    model = VocalTalkyCNN().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    train_loader, val_loader = (
        DataLoader(train, batch_size=settings.batch_size, shuffle=True),
        DataLoader(val, batch_size=settings.batch_size),
    )
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for mel, labels in train_loader:
            logits = model(mel.to(settings.device))
            labels = labels.to(settings.device)
            loss = functional.binary_cross_entropy_with_logits(
                logits, labels, pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if settings.max_train_batches and batches >= settings.max_train_batches:
                break
        val_metrics = _score(
            model, val_loader, settings.device, settings.max_val_batches, pos_weight
        )
        item: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_loss": val_metrics["loss"],
            "val_talky_activity_f1": val_metrics["talky_activity_f1"],
        }
        history.append(item)
        torch.save({"state_dict": model.state_dict()}, settings.checkpoint_dir / "last.pt")
        if val_metrics["talky_activity_f1"] > best_f1:
            best_f1 = val_metrics["talky_activity_f1"]
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
        train_vocal_talky_activity(TrainingSettings(**vars(args)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
