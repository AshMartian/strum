#!/usr/bin/env python3
"""Convert a reviewed legacy V14 checkpoint into a tensor-only runtime asset.

The converter is deliberately narrow. It accepts the historical V14 checkpoint
layout only after PyTorch's static unsafe-global scan identifies exactly the
two NumPy scalar helpers it needs. The emitted file contains model tensors and
small primitive metadata only, so normal runtime loading can use
``weights_only=True`` without an allowlist.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

ALLOWED_UNSAFE_GLOBALS = {"numpy.dtype", "numpy._core.multiarray.scalar"}
EXPECTED_KEYS = {
    "architecture",
    "best_f1",
    "config",
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "val_loss",
    "version",
}


def package_checkpoint(source: Path, output: Path) -> dict[str, object]:
    """Write a tensor-only V14 checkpoint after a narrow safe-global audit."""
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(source))
    if unsafe != ALLOWED_UNSAFE_GLOBALS:
        raise ValueError("legacy checkpoint has unsupported serialized globals")
    if output.exists():
        raise ValueError("refusing to overwrite an existing packaged checkpoint")
    allowed_classes = [np.dtype, np._core.multiarray.scalar, type(np.dtype(np.float64))]
    with torch.serialization.safe_globals(allowed_classes):
        raw = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict) or set(raw) != EXPECTED_KEYS:
        raise ValueError("legacy checkpoint has an unsupported V14 metadata layout")
    state_dict = raw["model_state_dict"]
    if (
        raw["architecture"] != "TwoStageDrumsCRNN"
        or raw["version"] != "v14"
        or not isinstance(state_dict, dict)
        or not state_dict
        or not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state_dict.items())
    ):
        raise ValueError("legacy checkpoint is not a V14 tensor state dictionary")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "strum-drums-v14-state-dict/v1",
        "architecture": "TwoStageDrumsCRNN/v14",
        "model_state_dict": state_dict,
    }
    torch.save(payload, output)
    return {"tensor_count": len(state_dict), "byte_length": output.stat().st_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(package_checkpoint(args.source, args.output))


if __name__ == "__main__":
    main()
