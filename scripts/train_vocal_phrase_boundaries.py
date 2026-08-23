#!/usr/bin/env python3
"""Train STRUM's bounded lead-Vocal phrase-boundary experiment.

The model predicts only start/end neighborhoods derived from ``PART VOCALS``
phrase markers.  It does not supply lyrics, talkies, harmony tracks, or MIDI
chart assembly and therefore is never a standalone chart profile.
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


class VocalPhraseDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Memory-map local phrase caches without retaining catalog locations."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        self.mel = np.load(cache_dir / f"{split}_mel.npy", mmap_mode="r")
        self.start = np.load(cache_dir / f"{split}_phrase_start.npy", mmap_mode="r")
        self.end = np.load(cache_dir / f"{split}_phrase_end.npy", mmap_mode="r")
        if (
            self.mel.ndim != 3
            or self.mel.shape[1:] != (N_MELS, SEGMENT_FRAMES)
            or self.start.shape != self.end.shape
            or self.start.shape != (self.mel.shape[0], SEGMENT_FRAMES)
        ):
            raise ValueError("Vocal phrase cache has an unsupported shape")

    def __len__(self) -> int:
        return self.mel.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.mel[index].astype(np.float32)).unsqueeze(0),
            torch.from_numpy(self.start[index].astype(np.float32)),
            torch.from_numpy(self.end[index].astype(np.float32)),
        )


class VocalPhraseBoundaryCNN(nn.Module):
    """Small time-preserving audio model for observed phrase boundaries."""

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
        self.boundaries = nn.Conv1d(channels, 2, kernel_size=1)

    def forward(self, mel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(mel).mean(dim=2)
        logits = self.boundaries(features)
        return logits[:, 0, :], logits[:, 1, :]


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


def _score(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int]:
    predicted = torch.sigmoid(logits) >= 0.5
    observed = targets >= 0.5
    return (
        int((predicted & observed).sum().item()),
        int((predicted & ~observed).sum().item()),
        int((~predicted & observed).sum().item()),
    )


def _metrics(
    model: VocalPhraseBoundaryCNN, loader: DataLoader, device: str, max_batches: int
) -> dict[str, float]:
    model.eval()
    loss_total = 0.0
    batches = 0
    totals = {"start": [0, 0, 0], "end": [0, 0, 0]}
    with torch.no_grad():
        for mel, starts, ends in loader:
            mel, starts, ends = mel.to(device), starts.to(device), ends.to(device)
            start_logits, end_logits = model(mel)
            loss_total += float(
                (
                    functional.binary_cross_entropy_with_logits(start_logits, starts)
                    + functional.binary_cross_entropy_with_logits(end_logits, ends)
                ).item()
            )
            for name, logits, targets in (
                ("start", start_logits, starts),
                ("end", end_logits, ends),
            ):
                score = _score(logits, targets)
                totals[name] = [
                    value + increment for value, increment in zip(totals[name], score, strict=True)
                ]
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return {
        "loss": loss_total / max(batches, 1),
        "phrase_start_f1": _f1(*totals["start"]),
        "phrase_end_f1": _f1(*totals["end"]),
    }


def train_vocal_phrase_boundaries(settings: TrainingSettings) -> dict[str, float]:
    """Train from complete caches and write tensor-only checkpoints."""
    if settings.epochs < 1 or settings.batch_size < 1 or settings.learning_rate <= 0:
        raise ValueError("Vocal phrase training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train_dataset = VocalPhraseDataset(settings.cache_dir, "train")
    val_dataset = VocalPhraseDataset(settings.cache_dir, "val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Vocal phrase training needs non-empty train and val caches")
    train_loader = DataLoader(train_dataset, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size)
    model = VocalPhraseBoundaryCNN().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_score = -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_loss = 0.0
        batches = 0
        for mel, starts, ends in train_loader:
            mel, starts, ends = (
                mel.to(settings.device),
                starts.to(settings.device),
                ends.to(settings.device),
            )
            start_logits, end_logits = model(mel)
            loss = functional.binary_cross_entropy_with_logits(
                start_logits, starts
            ) + functional.binary_cross_entropy_with_logits(end_logits, ends)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
            batches += 1
            if settings.max_train_batches and batches >= settings.max_train_batches:
                break
        validation = _metrics(model, val_loader, settings.device, settings.max_val_batches)
        item: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss / max(batches, 1),
            "val_loss": validation["loss"],
            "val_phrase_start_f1": validation["phrase_start_f1"],
            "val_phrase_end_f1": validation["phrase_end_f1"],
        }
        history.append(item)
        torch.save({"state_dict": model.state_dict()}, settings.checkpoint_dir / "last.pt")
        score = (validation["phrase_start_f1"] + validation["phrase_end_f1"]) / 2
        if score > best_score:
            best_score = score
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
        train_vocal_phrase_boundaries(
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
