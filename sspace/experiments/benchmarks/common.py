"""Shared strict helpers for benchmark source parsing."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any


def embedded_image(value: Any) -> tuple[bytes, str]:
    """Return embedded image bytes and SHA-256 from a parquet image field.

    Raises:
        ValueError: The field is not exactly a mapping containing bytes.
    """
    if not isinstance(value, Mapping) or not isinstance(value.get("bytes"), bytes):
        raise ValueError("Image field must contain embedded bytes")
    image_bytes = value["bytes"]
    return image_bytes, hashlib.sha256(image_bytes).hexdigest()
