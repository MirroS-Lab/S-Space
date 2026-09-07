"""Configure process requirements for deterministic CUDA execution."""

from __future__ import annotations

import os
from collections.abc import Mapping


CUBLAS_WORKSPACE_CONFIG = ":4096:8"


def _validate_workspace(configured: str | None) -> None:
    if configured not in {None, CUBLAS_WORKSPACE_CONFIG}:
        raise ValueError(
            "CUBLAS_WORKSPACE_CONFIG must be unset or equal to the frozen "
            f"{CUBLAS_WORKSPACE_CONFIG}"
        )


def deterministic_cuda_environment(
    base: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a subprocess environment with the frozen cuBLAS workspace."""
    environment = dict(os.environ if base is None else base)
    _validate_workspace(environment.get("CUBLAS_WORKSPACE_CONFIG"))
    environment["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    return environment


def configure_deterministic_cuda() -> None:
    """Set the frozen cuBLAS workspace before deterministic CUDA operations."""
    _validate_workspace(os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
