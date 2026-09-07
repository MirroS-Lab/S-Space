"""Shared primitives for downloading pinned Hugging Face snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from sspace.run_records import atomic_json


HUB_ASSET_MARKER = ".sspace_hub_asset.json"


@dataclass(frozen=True)
class HubAsset:
    """Declare one immutable Hugging Face repository snapshot."""

    key: str
    repo_id: str
    revision: str
    repo_type: str
    relative_dir: str


def download_hub_assets(
    asset_root: Path,
    assets: Iterable[HubAsset],
    *,
    dry_run: bool = False,
) -> list[dict[str, str]]:
    """Download pinned snapshots to stable asset-root directories."""
    selected = tuple(assets)
    if dry_run:
        for asset in selected:
            destination = asset_root / asset.relative_dir
            print(
                f"{asset.repo_type}:{asset.repo_id}@{asset.revision} -> {destination}"
            )
        return [asdict(asset) for asset in selected]

    from huggingface_hub import snapshot_download

    records = []
    cache_dir = asset_root / ".hub_cache"
    for index, asset in enumerate(selected, start=1):
        destination = asset_root / asset.relative_dir
        print(f"asset {index}/{len(selected)} {asset.key}", flush=True)
        snapshot_download(
            repo_id=asset.repo_id,
            repo_type=asset.repo_type,
            revision=asset.revision,
            local_dir=destination,
            cache_dir=cache_dir,
        )
        if not destination.is_dir():
            raise FileNotFoundError(destination)
        atomic_json(
            destination / HUB_ASSET_MARKER,
            {
                "schema_version": "1.0.0",
                "repo_id": asset.repo_id,
                "repo_type": asset.repo_type,
                "revision": asset.revision,
            },
        )
        records.append({**asdict(asset), "path": str(destination.resolve())})
    return records
