"""Pinned SpinBench source declaration and preparation."""

from pathlib import Path

from sspace.data.hub import HubAsset

from .prepare import prepare_spinbench


ASSETS = (
    HubAsset(
        "spinbench",
        "YuyouZhang/SpinBench",
        "291aaab6aa820f9975eff8a193914a240f5a60fe",
        "dataset",
        "datasets/spinbench",
    ),
)


def prepare(asset_root: Path) -> None:
    """Extract and validate the downloaded SpinBench snapshot."""
    prepare_spinbench(asset_root / ASSETS[0].relative_dir)
