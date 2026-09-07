"""Safely extract downloaded dataset archives."""

from __future__ import annotations

import zipfile
from pathlib import Path


def _matches_member(
    path: Path, archive: zipfile.ZipFile, member: zipfile.ZipInfo
) -> bool:
    if not path.is_file() or path.stat().st_size != member.file_size:
        return False
    with path.open("rb") as current, archive.open(member) as expected:
        return all(
            left == right
            for left, right in zip(
                iter(lambda: current.read(1024 * 1024), b""),
                iter(lambda: expected.read(1024 * 1024), b""),
                strict=True,
            )
        )


def extract_zip_once(archive: Path, destination: Path) -> None:
    """Extract one ZIP after rejecting paths outside the destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        targets: list[tuple[Path, zipfile.ZipInfo]] = []
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(
                    f"Archive member escapes destination: {member.filename}"
                )
            if not member.is_dir():
                targets.append((target, member))
        existing = [target.exists() for target, _ in targets]
        if targets and all(existing):
            changed = [
                target
                for target, member in targets
                if not _matches_member(target, source, member)
            ]
            if changed:
                raise ValueError(f"Extracted archive member changed: {changed[0]}")
            return
        if any(existing):
            raise FileExistsError(f"Archive is only partially extracted: {archive}")
        source.extractall(destination)
