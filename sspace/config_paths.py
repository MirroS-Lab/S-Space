"""Resolve portable paths declared by release configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def resolve_project_path(value: Any, project_root: Path, field: str) -> Path:
    """Resolve a nonempty project-relative path without following symlinks."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty path string")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(
            f"{field} must be project-relative and cannot escape the project"
        )
    return project_root / candidate


def as_project_relative(path: Path, project_root: Path, field: str) -> str:
    """Serialize one lexical in-project path for a generated configuration."""
    root = project_root.absolute()
    candidate = path.absolute() if path.is_absolute() else (root / path).absolute()
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"{field} must remain inside the project") from error
