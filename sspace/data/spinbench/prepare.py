"""Extract and verify the pinned SpinBench snapshot."""

from __future__ import annotations

from pathlib import Path

from sspace.data.archive import extract_zip_once


def prepare_spinbench(source_dir: Path) -> None:
    """Extract images and require the annotation/image layout used by configs."""
    archive = source_dir / "images.zip"
    if not archive.is_file():
        raise FileNotFoundError(archive)
    extract_zip_once(archive, source_dir)
    if (
        not (source_dir / "test.jsonl").is_file()
        or not (source_dir / "images").is_dir()
    ):
        raise ValueError("SpinBench snapshot is incomplete after extraction")
