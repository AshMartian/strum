"""Canonical, host-independent quality policy for Expert profile promotion."""

from __future__ import annotations

import hashlib
import json
from typing import Final

PROFILE_POLICY_ID: Final = "strum-five-lane-expert-profile/v1"
_POLICY_JSON: Final = json.dumps({
    "policy_id": PROFILE_POLICY_ID,
    "minimum_onset_f1": 0.5,
    "minimum_fret_f1": 0.5,
    "onset_threshold": 0.4,
    "fret_thresholds": [0.5, 0.5, 0.5, 0.5, 0.5],
    "note_duration_ms": 100.0,
}, sort_keys=True, separators=(",", ":"))


def profile_quality_policy() -> dict[str, object]:
    return json.loads(_POLICY_JSON)


def profile_quality_policy_sha256() -> str:
    return hashlib.sha256(_POLICY_JSON.encode("utf-8")).hexdigest()
