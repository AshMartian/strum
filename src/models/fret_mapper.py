"""Typed MLP used by the catalog-trained five-lane fret mapper.

The model intentionally lives outside the training script so a future,
explicit profile loader can instantiate exactly the architecture that produced
the component.  Importing this module does not select the component for chart
execution.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FretMapperMLP(nn.Module):
    """Map one Basic Pitch onset feature vector to five lane logits."""

    def __init__(
        self,
        in_dim: int = 95,
        hidden: int = 256,
        out_dim: int = 5,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return unnormalized five-lane logits."""
        return self.net(features)
