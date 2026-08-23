"""Shared, path-free schema for Pro event-proposal training configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class ProEventProposalTrainingOptionsError(ValueError):
    """Raised when a proposal training configuration is not portable."""


@dataclass(frozen=True)
class ProEventProposalTrainingOptions:
    """The exact configuration schema persisted into a raw candidate bundle."""

    model_id: str
    epochs: int = 25
    batch_size: int = 32
    learning_rate: float = 0.0003
    device: str = "auto"
    limit_songs: int = 0
    max_train_batches: int = 0
    max_val_batches: int = 0
    seed: int = 20260822
    channels: int = 48
    negative_ratio: int = 4
    negative_exclusion_ms: int = 80

    @classmethod
    def from_mapping(cls, raw: Mapping[str, object]) -> ProEventProposalTrainingOptions:
        permitted = {
            "model_id",
            "epochs",
            "batch_size",
            "learning_rate",
            "device",
            "limit_songs",
            "max_train_batches",
            "max_val_batches",
            "seed",
            "channels",
            "negative_ratio",
            "negative_exclusion_ms",
        }
        requested = dict(raw)
        if set(requested) - permitted:
            raise ProEventProposalTrainingOptionsError(
                "unsupported Pro event proposal training option"
            )
        model_id = requested.get("model_id")
        if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
            raise ProEventProposalTrainingOptionsError("Pro event proposal model_id is invalid")
        values: dict[str, Any] = {"model_id": model_id}
        for key, default, minimum in (
            ("epochs", 25, 1),
            ("batch_size", 32, 1),
            ("limit_songs", 0, 0),
            ("max_train_batches", 0, 0),
            ("max_val_batches", 0, 0),
            ("seed", 20260822, 0),
            ("channels", 48, 1),
            ("negative_ratio", 4, 1),
            ("negative_exclusion_ms", 80, 0),
        ):
            value = requested.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ProEventProposalTrainingOptionsError(f"Pro event proposal {key} is invalid")
            if key == "negative_ratio" and value > 32:
                raise ProEventProposalTrainingOptionsError(
                    "Pro event proposal negative_ratio is invalid"
                )
            if key == "negative_exclusion_ms" and value > 5000:
                raise ProEventProposalTrainingOptionsError(
                    "Pro event proposal negative_exclusion_ms is invalid"
                )
            values[key] = value
        learning_rate = requested.get("learning_rate", 0.0003)
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or learning_rate <= 0
        ):
            raise ProEventProposalTrainingOptionsError(
                "Pro event proposal learning_rate is invalid"
            )
        values["learning_rate"] = float(learning_rate)
        device = requested.get("device", "auto")
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ProEventProposalTrainingOptionsError("Pro event proposal device is invalid")
        values["device"] = device
        return cls(**values)

    def portable(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "limit_songs": self.limit_songs,
            "max_train_batches": self.max_train_batches,
            "max_val_batches": self.max_val_batches,
            "seed": self.seed,
            "channels": self.channels,
            "negative_ratio": self.negative_ratio,
            "negative_exclusion_ms": self.negative_exclusion_ms,
        }
