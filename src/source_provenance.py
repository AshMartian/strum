"""Small, path-safe source-build provenance primitives.

Source revision values leave a worker in runtime and artifact metadata.  They
are identifiers, not user-facing labels: accepting a free-form environment
value here would let a host path (or another private build detail) escape in a
portable bundle.  A Git object name is deliberately the only supported
identity.  Short Git IDs are accepted for callers that pin an abbreviated
revision, while generated artifacts preserve the exact supplied identity.
"""

from __future__ import annotations

import re

# Git accepts a short object name of at least seven hexadecimal characters.
# Cap the value at a SHA-256 object ID rather than accepting arbitrary long
# strings.  Lowercase is required to keep artifact identity canonical.
SOURCE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{7,64}\Z")


def source_revision_identity(value: object) -> str | None:
    """Return a canonical safe Git revision identity, or ``None``.

    The function intentionally does not normalize case or strip whitespace:
    callers should reject or redact ambiguous configuration instead of silently
    changing the declared build identity.
    """
    if not isinstance(value, str) or not SOURCE_REVISION_PATTERN.fullmatch(value):
        return None
    return value
