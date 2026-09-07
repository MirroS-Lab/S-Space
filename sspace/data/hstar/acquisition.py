"""Extract and verify the pinned H* HOS source snapshot."""

from __future__ import annotations

from pathlib import Path

from sspace.data.archive import extract_zip_once
from sspace.run_records import file_sha256


def verify_hstar_archive(archive: Path, expected_sha256: str) -> str:
    """Return the actual HOS archive digest only when it matches the pin."""
    if not archive.is_file():
        raise FileNotFoundError(archive)
    actual = file_sha256(archive)
    if actual != expected_sha256:
        raise ValueError(f"H* archive SHA-256 {actual} != {expected_sha256}")
    return actual


def prepare_hstar_source(source_dir: Path) -> Path:
    """Extract HOS and return the directory consumed by the H* builder."""
    archive = source_dir / "hos_train.zip"
    from .pairs import SOURCE_ARCHIVE_SHA256

    verify_hstar_archive(archive, SOURCE_ARCHIVE_SHA256)
    extract_zip_once(archive, source_dir)
    extracted = source_dir / "hos_train"
    if not extracted.is_dir():
        raise ValueError("H* HOS snapshot is incomplete after extraction")
    return extracted
