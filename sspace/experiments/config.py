"""Validate metadata shared by all experiment launch configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


LAUNCH_FIELDS = {"runner", "suites"}


def validate_launch_metadata(
    values: Mapping[str, Any], expected_runner: str
) -> tuple[str, ...]:
    """Validate the explicit evaluator route and non-empty suite memberships.

    Args:
        values: Parsed experiment configuration.
        expected_runner: Evaluator identifier accepted by the config loader.

    Returns:
        Ordered suite names declared by the configuration.

    Raises:
        ValueError: The runner differs, suites are malformed, or a suite repeats.

    Side effects:
        None.
    """
    if values.get("runner") != expected_runner:
        raise ValueError(f"runner must be {expected_runner!r}")
    suites = values.get("suites")
    if not isinstance(suites, list) or not suites:
        raise ValueError("suites must be a non-empty JSON string array")
    if any(not isinstance(value, str) or not value.strip() for value in suites):
        raise ValueError("Every suite name must be a non-empty string")
    if len(suites) != len(set(suites)):
        raise ValueError("Suite names must be unique")
    return tuple(suites)
