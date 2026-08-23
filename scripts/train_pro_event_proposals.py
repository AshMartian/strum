#!/usr/bin/env python3
"""Train a raw audio-only Pro event-proposal candidate from a private cache.

This classifier scores a supplied offline log-mel window as event/non-event.
It is deliberately limited to proposal scores: no code in this script accepts
MIDI at inference, predicts Pro attributes, decodes a temporal sequence, or
writes a chart.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset


class ProEventProposalError(ValueError):
    """Raised when a proposal cache cannot establish audio-only labels."""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventProposalError("Pro proposal target cache is unreadable") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProEventProposalError("Pro proposal target cache has no valid rows")
    return rows


class ProEventProposalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Private cache reader that refuses labels beyond a binary onset target."""

    def __init__(self, cache_dir: Path, split: str) -> None:
        features = np.load(cache_dir / f"{split}_logmel.npy", mmap_mode="r")
        rows = _read_jsonl(cache_dir / f"{split}_proposal_targets.jsonl")
        if features.ndim != 3 or features.shape[0] != len(rows) or not features.shape[0]:
            raise ProEventProposalError("Pro proposal cache features and labels do not match")
        targets: list[float] = []
        for row in rows:
            if row.get("split") != split or not isinstance(row.get("is_event"), bool):
                raise ProEventProposalError("Pro proposal cache split or label is invalid")
            if not isinstance(row.get("source_id"), str) or not isinstance(
                row.get("center_frame"), int
            ):
                raise ProEventProposalError("Pro proposal cache source identity is invalid")
            targets.append(float(row["is_event"]))
        if not 0 < sum(targets) < len(targets):
            raise ProEventProposalError("Pro proposal cache requires positive and negative labels")
        self.features = features
        self.targets = np.asarray(targets, dtype=np.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index].astype(np.float32)).unsqueeze(0),
            torch.tensor(self.targets[index], dtype=torch.float32),
        )


class ProEventProposalCNN(nn.Module):
    """Bounded offline window scorer; it contains no sequence or MIDI head."""

    def __init__(self, *, channels: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        self.proposal = nn.Linear(channels, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proposal(self.features(features).mean(dim=(2, 3))).squeeze(1)


@dataclass(frozen=True)
class TrainingSettings:
    cache_dir: Path
    checkpoint_dir: Path
    task_kind: str
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    max_train_batches: int
    max_val_batches: int
    seed: int
    channels: int


def _metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int, int]:
    predicted = torch.sigmoid(logits) >= 0.5
    observed = targets >= 0.5
    return (
        int((predicted & observed).sum().item()),
        int((predicted & ~observed).sum().item()),
        int((~predicted & observed).sum().item()),
        int((~predicted & ~observed).sum().item()),
    )


def _score(
    model: ProEventProposalCNN, loader: DataLoader, device: str, max_batches: int
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = tp = fp = fn = tn = 0
    with torch.no_grad():
        for features, targets in loader:
            logits = model(features.to(device))
            observed = targets.to(device)
            total_loss += float(
                functional.binary_cross_entropy_with_logits(logits, observed).item()
            )
            current_tp, current_fp, current_fn, current_tn = _metrics(logits, observed)
            tp += current_tp
            fp += current_fp
            fn += current_fn
            tn += current_tn
            batches += 1
            if max_batches and batches >= max_batches:
                break
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "loss": total_loss / max(batches, 1),
        "proposal_precision": precision,
        "proposal_recall": recall,
        "proposal_f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "proposal_balanced_accuracy": 0.5 * (recall + tn / max(tn + fp, 1)),
    }


def train_pro_event_proposals(settings: TrainingSettings) -> dict[str, float]:
    if settings.task_kind not in {"pro_guitar", "pro_bass", "pro_keys"}:
        raise ProEventProposalError("Pro proposal task kind is invalid")
    if (
        min(settings.epochs, settings.batch_size, settings.channels) < 1
        or settings.learning_rate <= 0
    ):
        raise ProEventProposalError("Pro proposal training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train = ProEventProposalDataset(settings.cache_dir, "train")
    val = ProEventProposalDataset(settings.cache_dir, "val")
    model = ProEventProposalCNN(channels=settings.channels).to(settings.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)
    train_loader = DataLoader(train, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=settings.batch_size)
    settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
    for epoch in range(1, settings.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for features, targets in train_loader:
            logits = model(features.to(settings.device))
            loss = functional.binary_cross_entropy_with_logits(logits, targets.to(settings.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
            if settings.max_train_batches and batches >= settings.max_train_batches:
                break
        metrics = _score(model, val_loader, settings.device, settings.max_val_batches)
        item: dict[str, float | int] = {"epoch": epoch, "train_loss": total_loss / max(batches, 1)}
        item.update({f"val_{key}": value for key, value in metrics.items()})
        history.append(item)
        payload = {
            "format": "strum-pro-event-proposal-candidate-checkpoint/v1",
            "task_kind": settings.task_kind,
            "channels": settings.channels,
            "state_dict": model.state_dict(),
        }
        torch.save(payload, settings.checkpoint_dir / "last.pt")
        if metrics["proposal_f1"] > best_f1:
            best_f1 = metrics["proposal_f1"]
            torch.save(payload, settings.checkpoint_dir / "best.pt")
    (settings.checkpoint_dir / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return {key: float(value) for key, value in history[-1].items() if key != "epoch"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--task-kind", choices=["pro_guitar", "pro_bass", "pro_keys"], required=True
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", choices=["cuda", "mps", "cpu"], required=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--channels", type=int, default=48)
    args = parser.parse_args()
    try:
        metrics = train_pro_event_proposals(
            TrainingSettings(
                cache_dir=args.cache_dir,
                checkpoint_dir=args.checkpoint_dir,
                task_kind=args.task_kind,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                device=args.device,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
                seed=args.seed,
                channels=args.channels,
            )
        )
    except ProEventProposalError as error:
        parser.error(str(error))
    print(json.dumps(metrics, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
