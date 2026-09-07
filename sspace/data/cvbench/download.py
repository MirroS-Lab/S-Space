"""Pinned CV-Bench source declaration."""

from pathlib import Path

from sspace.data.hub import HubAsset


ASSETS = (
    HubAsset(
        "cvbench",
        "nyu-visionx/CV-Bench",
        "bc284db50d036958861cb60cdd7b77612052ce0d",
        "dataset",
        "datasets/cvbench",
    ),
)


def prepare(asset_root: Path) -> None:
    """Require the two parquet files consumed by the CV-Bench adapter."""
    root = asset_root / ASSETS[0].relative_dir
    for name in ("test_2d.parquet", "test_3d.parquet"):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
