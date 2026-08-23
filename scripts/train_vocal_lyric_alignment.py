#!/usr/bin/env python3
"""Train STRUM's bounded observed-lyric CTC component.

This model turns catalog-approved audio windows into character-token logits. It
does not decide phrase boundaries, pitchless/talky encoding, harmony tracks,
or write a playable MIDI chart.
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
from torch.utils.data import DataLoader, Dataset

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.preprocess_vocal_lyric_alignment import VOCABULARY  # noqa: E402
from scripts.preprocess_vocals_frames import N_MELS, SEGMENT_FRAMES  # noqa: E402


class VocalLyricDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Memory-map local CTC caches without retaining catalog locations."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        self.mel = np.load(cache_dir / f"{split}_mel.npy", mmap_mode="r")
        self.targets = np.load(cache_dir / f"{split}_targets.npy", mmap_mode="r")
        self.lengths = np.load(cache_dir / f"{split}_target_lengths.npy", mmap_mode="r")
        if (
            self.mel.ndim != 3
            or self.mel.shape[1:] != (N_MELS, SEGMENT_FRAMES)
            or self.targets.ndim != 2
            or self.lengths.shape != (self.mel.shape[0],)
            or self.targets.shape[0] != self.mel.shape[0]
            or np.any(self.lengths < 1)
            or np.any(self.lengths > self.targets.shape[1])
        ):
            raise ValueError("Vocal lyric cache has an unsupported shape")

    def __len__(self) -> int:
        return int(self.mel.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        length = int(self.lengths[index])
        return (
            torch.from_numpy(self.mel[index].astype(np.float32)).unsqueeze(0),
            torch.from_numpy(self.targets[index, :length].astype(np.int64)),
            torch.tensor(length, dtype=torch.long),
        )


def _collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mel, targets, lengths = zip(*batch, strict=True)
    return torch.stack(mel), torch.cat(targets), torch.stack(lengths)


class VocalLyricCTC(nn.Module):
    """Small time-preserving acoustic encoder for observed chart text."""

    def __init__(self, *, channels: int = 64, hidden_size: int = 96) -> None:
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
        self.encoder = nn.GRU(channels, hidden_size, batch_first=True, bidirectional=True)
        self.tokens = nn.Linear(hidden_size * 2, len(VOCABULARY))

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        features = self.features(mel).mean(dim=2).transpose(1, 2)
        encoded, _state = self.encoder(features)
        return self.tokens(encoded).transpose(0, 1)


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


def _ctc_loss(
    logits: torch.Tensor, targets: torch.Tensor, target_lengths: torch.Tensor
) -> torch.Tensor:
    input_lengths = torch.full((logits.shape[1],), logits.shape[0], dtype=torch.long, device="cpu")
    return nn.functional.ctc_loss(
        logits.log_softmax(2),
        targets,
        input_lengths,
        target_lengths.cpu(),
        blank=0,
        zero_infinity=True,
    )


def _validation(model: VocalLyricCTC, loader: DataLoader, device: str, max_batches: int) -> float:
    model.eval()
    total = 0.0
    batches = 0
    with torch.no_grad():
        for mel, targets, lengths in loader:
            total += float(_ctc_loss(model(mel.to(device)), targets.to(device), lengths).item())
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return total / max(batches, 1)


def train_vocal_lyric_alignment(settings: TrainingSettings) -> dict[str, float]:
    """Train CTC from complete local caches and write tensor-only weights."""
    if settings.epochs < 1 or settings.batch_size < 1 or settings.learning_rate <= 0:
        raise ValueError("Vocal lyric training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train_dataset = VocalLyricDataset(settings.cache_dir, "train")
    val_dataset = VocalLyricDataset(settings.cache_dir, "val")
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Vocal lyric training needs non-empty train and val caches")
    train_loader = DataLoader(
        train_dataset, batch_size=settings.batch_size, shuffle=True, collate_fn=_collate
    )
    val_loader = DataLoader(val_dataset, batch_size=settings.batch_size, collate_fn=_collate)
    model = VocalLyricCTC().to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, settings.epochs + 1):
        model.train()
        train_total = 0.0
        batches = 0
        for mel, targets, lengths in train_loader:
            loss = _ctc_loss(model(mel.to(settings.device)), targets.to(settings.device), lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_total += float(loss.item())
            batches += 1
            if settings.max_train_batches and batches >= settings.max_train_batches:
                break
        val_loss = _validation(model, val_loader, settings.device, settings.max_val_batches)
        item: dict[str, float | int] = {
            "epoch": epoch,
            "train_ctc_loss": train_total / max(batches, 1),
            "val_ctc_loss": val_loss,
        }
        history.append(item)
        torch.save({"state_dict": model.state_dict()}, settings.checkpoint_dir / "last.pt")
        if val_loss < best_loss:
            best_loss = val_loss
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
        train_vocal_lyric_alignment(
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
