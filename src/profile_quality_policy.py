"""Canonical, host-independent quality policy for Expert profile promotion."""

from __future__ import annotations

import hashlib
import json
from typing import Final

PROFILE_POLICY_ID: Final = "strum-five-lane-expert-profile/v1"
_POLICY: Final = {
    "policy_id": PROFILE_POLICY_ID,
    "minimum_onset_f1": 0.5,
    "minimum_fret_f1": 0.5,
    "onset_threshold": None,
    "fret_thresholds": None,
}


def profile_quality_policy() -> dict[str, object]:
    return dict(_POLICY)


def profile_quality_policy_sha256() -> str:
    encoded = json.dumps(_POLICY, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
