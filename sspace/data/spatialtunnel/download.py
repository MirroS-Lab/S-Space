"""Pinned SpatialTunnel source declaration."""

from pathlib import Path

from sspace.data.hub import HubAsset


ASSETS = (
    HubAsset(
        "spatialtunnel",
        "cubec/spatialtunnel",
        "96f48f3e6ab8738e87a151e7c8fc29954fa10fd6",
        "dataset",
        "datasets/spatialtunnel",
    ),
)


def prepare(asset_root: Path) -> None:
    """Require the exact file consumed by SpatialTunnel configs."""
    path = asset_root / ASSETS[0].relative_dir / "contrastive_probing.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
