"""Regression coverage for the learned guitar fret decoder source module."""

from __future__ import annotations

import numpy as np

from scripts.viterbi_fret_decode import STATE_VECS, STATES, viterbi_decode


def test_viterbi_fret_decode_imports_and_returns_one_state_per_onset() -> None:
    probabilities = np.array(
        [
            [0.95, 0.05, 0.05, 0.05, 0.05],
            [0.05, 0.95, 0.05, 0.05, 0.05],
            [0.05, 0.05, 0.95, 0.05, 0.05],
        ],
        dtype=np.float32,
    )

    decoded = viterbi_decode(probabilities)

    assert len(decoded) == len(probabilities)
    assert all(state in STATES for state in decoded)
    assert STATE_VECS.shape == (len(STATES), 5)
