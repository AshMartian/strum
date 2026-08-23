#!/usr/bin/env python3
"""Train an exact-Pro *known-event* attribute candidate from a private cache.

This is intentionally narrower than a playable Pro charter.  Each input is
centered on an authored reference event supplied by the held-out catalog label.
The model learns only the attributes at that already-known event (string/fret/
technique/track variant or chromatic pitches/range state).  It cannot propose
event times, decode a free-running sequence, or write MIDI.
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

STRING_TECHNIQUES = (
    "normal",
    "arpeggio_form",
    "bent",
    "muted",
    "tapped",
    "harmonic",
    "pinch_harmonic",
)
STRING_COUNT = 6
MAX_FRET = 22
KEYS_MIN_PITCH = 48
KEYS_MAX_PITCH = 72
RANGE_ANCHORS = ("none", "C", "D", "E", "F", "G", "A")


class ProEventAttributeError(ValueError):
    """Raised when a cache cannot prove exact-Pro candidate semantics."""


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as error:
        raise ProEventAttributeError("Pro event target cache is unreadable") from error
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ProEventAttributeError("Pro event target cache has no valid rows")
    return rows


def _current_range(shifts: object, tick: object) -> int:
    if not isinstance(shifts, list) or not isinstance(tick, int):
        raise ProEventAttributeError("Pro Keys candidate target has invalid range shifts")
    latest = "none"
    for shift in shifts:
        if not isinstance(shift, dict) or not isinstance(shift.get("tick"), int):
            raise ProEventAttributeError("Pro Keys candidate range shift is invalid")
        if shift["tick"] <= tick:
            anchor = shift.get("anchor")
            if anchor not in RANGE_ANCHORS[1:]:
                raise ProEventAttributeError("Pro Keys candidate range anchor is invalid")
            latest = anchor
    return RANGE_ANCHORS.index(latest)


def _string_vector(row: dict[str, object]) -> tuple[np.ndarray, int]:
    if row.get("target_language") != "string_fret_technique/v1":
        raise ProEventAttributeError("Pro string cache target language is invalid")
    variant = row.get("track_variant")
    if variant not in {"standard", "22_fret"}:
        raise ProEventAttributeError("Pro string cache track variant is invalid")
    events = row.get("events")
    if not isinstance(events, list) or not events:
        raise ProEventAttributeError("Pro string cache has no event targets")
    target = np.zeros(STRING_COUNT * (MAX_FRET + 1) * len(STRING_TECHNIQUES), dtype=np.float32)
    for event in events:
        if not isinstance(event, dict):
            raise ProEventAttributeError("Pro string cache event is invalid")
        string, fret, technique = event.get("string"), event.get("fret"), event.get("technique")
        if (
            not isinstance(string, int)
            or not isinstance(fret, int)
            or not 0 <= string < STRING_COUNT
            or not 0 <= fret <= MAX_FRET
            or technique not in STRING_TECHNIQUES
        ):
            raise ProEventAttributeError("Pro string cache event semantics are invalid")
        if variant == "standard" and fret > 17:
            raise ProEventAttributeError("standard Pro string cache contains a 22-fret target")
        index = (
            (string * (MAX_FRET + 1) + fret) * len(STRING_TECHNIQUES)
        ) + STRING_TECHNIQUES.index(technique)
        target[index] = 1.0
    return target, int(variant == "22_fret")


def _keys_vector(row: dict[str, object]) -> tuple[np.ndarray, int]:
    if row.get("target_language") != "pitch_channel_range_shift/v1":
        raise ProEventAttributeError("Pro Keys cache target language is invalid")
    events = row.get("events")
    if not isinstance(events, list) or not events:
        raise ProEventAttributeError("Pro Keys cache has no event targets")
    target = np.zeros(KEYS_MAX_PITCH - KEYS_MIN_PITCH + 1, dtype=np.float32)
    for event in events:
        if not isinstance(event, dict) or not isinstance(event.get("pitch"), int):
            raise ProEventAttributeError("Pro Keys cache event is invalid")
        pitch = event["pitch"]
        if not KEYS_MIN_PITCH <= pitch <= KEYS_MAX_PITCH:
            raise ProEventAttributeError("Pro Keys cache pitch is invalid")
        target[pitch - KEYS_MIN_PITCH] = 1.0
    return target, _current_range(row.get("range_shifts"), row.get("event_tick"))


class ProEventDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, cache_dir: Path, split: str, task_kind: str) -> None:
        features = np.load(cache_dir / f"{split}_logmel.npy", mmap_mode="r")
        rows = _read_jsonl(cache_dir / f"{split}_targets.jsonl")
        if features.ndim != 3 or features.shape[0] != len(rows) or not features.shape[0]:
            raise ProEventAttributeError("Pro event cache feature and target rows do not match")
        self.features = features
        vectors: list[np.ndarray] = []
        states: list[int] = []
        for row in rows:
            if row.get("split") != split:
                raise ProEventAttributeError("Pro event cache split is inconsistent")
            vector, state = _keys_vector(row) if task_kind == "pro_keys" else _string_vector(row)
            vectors.append(vector)
            states.append(state)
        self.targets = np.stack(vectors)
        self.states = np.asarray(states, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(self.features[index].astype(np.float32)).unsqueeze(0),
            torch.from_numpy(self.targets[index]),
            torch.tensor(self.states[index], dtype=torch.long),
        )


class ProEventAttributeCNN(nn.Module):
    """A bounded event-window encoder; it has no temporal event proposal head."""

    def __init__(self, *, target_dim: int, state_dim: int, channels: int) -> None:
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
        self.target = nn.Linear(channels, target_dim)
        self.state = nn.Linear(channels, state_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.features(features).mean(dim=(2, 3))
        return self.target(pooled), self.state(pooled)


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


def _f1(tp: int, fp: int, fn: int) -> float:
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def _score(
    model: ProEventAttributeCNN, loader: DataLoader, device: str, max_batches: int
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    batches = tp = fp = fn = correct_state = exact = events = 0
    with torch.no_grad():
        for features, targets, states in loader:
            target_logits, state_logits = model(features.to(device))
            targets, states = targets.to(device), states.to(device)
            loss = functional.binary_cross_entropy_with_logits(target_logits, targets)
            loss += functional.cross_entropy(state_logits, states)
            predicted = torch.sigmoid(target_logits) >= 0.5
            observed = targets >= 0.5
            tp += int((predicted & observed).sum().item())
            fp += int((predicted & ~observed).sum().item())
            fn += int((~predicted & observed).sum().item())
            correct_state += int((state_logits.argmax(dim=1) == states).sum().item())
            exact += int(
                ((predicted == observed).all(dim=1) & (state_logits.argmax(dim=1) == states))
                .sum()
                .item()
            )
            events += len(states)
            total_loss += float(loss.item())
            batches += 1
            if max_batches and batches >= max_batches:
                break
    return {
        "loss": total_loss / max(batches, 1),
        "known_event_token_f1": _f1(tp, fp, fn),
        "known_event_state_accuracy": correct_state / max(events, 1),
        "known_event_exact_accuracy": exact / max(events, 1),
    }


def train_pro_event_attributes(settings: TrainingSettings) -> dict[str, float]:
    if settings.task_kind not in {"pro_guitar", "pro_bass", "pro_keys"}:
        raise ProEventAttributeError("Pro event candidate task kind is invalid")
    if (
        min(settings.epochs, settings.batch_size, settings.channels) < 1
        or settings.learning_rate <= 0
    ):
        raise ProEventAttributeError("Pro event candidate training settings are invalid")
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    train = ProEventDataset(settings.cache_dir, "train", settings.task_kind)
    val = ProEventDataset(settings.cache_dir, "val", settings.task_kind)
    target_dim = train.targets.shape[1]
    state_dim = len(RANGE_ANCHORS) if settings.task_kind == "pro_keys" else 2
    if val.targets.shape[1] != target_dim or not len(train) or not len(val):
        raise ProEventAttributeError(
            "Pro event candidate requires non-empty compatible train and val caches"
        )
    model = ProEventAttributeCNN(
        target_dim=target_dim, state_dim=state_dim, channels=settings.channels
    ).to(settings.device)
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
        for features, targets, states in train_loader:
            target_logits, state_logits = model(features.to(settings.device))
            loss = functional.binary_cross_entropy_with_logits(
                target_logits, targets.to(settings.device)
            )
            loss += functional.cross_entropy(state_logits, states.to(settings.device))
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
            "format": "strum-pro-event-attribute-candidate-checkpoint/v1",
            "task_kind": settings.task_kind,
            "target_dimension": target_dim,
            "state_dimension": state_dim,
            "channels": settings.channels,
            "state_dict": model.state_dict(),
        }
        torch.save(payload, settings.checkpoint_dir / "last.pt")
        if metrics["known_event_token_f1"] > best_f1:
            best_f1 = metrics["known_event_token_f1"]
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
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], required=True)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--channels", type=int, default=48)
    args = parser.parse_args()
    try:
        train_pro_event_attributes(TrainingSettings(**vars(args)))
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI delegation.
    raise SystemExit(main())
