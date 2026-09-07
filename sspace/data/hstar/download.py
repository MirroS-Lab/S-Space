"""Pinned H* HOS source declaration and preparation."""

from pathlib import Path

from sspace.data.hub import HubAsset

from .acquisition import prepare_hstar_source


ASSETS = (
    HubAsset(
        "hstar",
        "humanoid-vstar/hvs_rl",
        "030e612b7d140b659c024464ab50508379c2dcfa",
        "dataset",
        "datasets/hstar",
    ),
)


def prepare(asset_root: Path) -> None:
    """Extract and validate the downloaded H* HOS snapshot."""
    prepare_hstar_source(asset_root / ASSETS[0].relative_dir)
