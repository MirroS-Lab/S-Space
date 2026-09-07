"""Pinned EmbSpatial source declaration."""

from pathlib import Path

from sspace.data.hub import HubAsset


ASSETS = (
    HubAsset(
        "embspatial",
        "ch-min/EmbSpatial-Bench-tsv",
        "c953faef1693576727fe6af1910e7c92082b246c",
        "dataset",
        "datasets/embspatial",
    ),
    HubAsset(
        "embspatial_annotations",
        "FlagEval/EmbSpatial-Bench",
        "3c0e6b34b632de666a51091727c0128d07a54a6a",
        "dataset",
        "datasets/embspatial_annotations",
    ),
)


def prepare(asset_root: Path) -> None:
    """Require the exact benchmark and Spatial Causality inputs."""
    required = (
        asset_root / ASSETS[0].relative_dir / "EmbSpatial-Bench.tsv",
        asset_root / ASSETS[1].relative_dir / "data/test-00000-of-00001.parquet",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
