"""Resolve and acquire the complete pinned Release asset set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from sspace.run_records import atomic_json

from .cvbench.download import ASSETS as CVBENCH_ASSETS
from .embspatial.download import ASSETS as EMBSPATIAL_ASSETS
from .hstar.download import ASSETS as HSTAR_ASSETS
from .hub import HubAsset, download_hub_assets
from .spatialtunnel.download import ASSETS as SPATIALTUNNEL_ASSETS
from .spinbench.download import ASSETS as SPINBENCH_ASSETS


MODEL_ASSETS = (
    HubAsset(
        "molmo2_er",
        "allenai/Molmo2-ER",
        "dab22564403d2607855bb1fffb0721285b445081",
        "model",
        "models/molmo2_er",
    ),
    HubAsset(
        "molmoact2",
        "allenai/MolmoAct2",
        "e432d85f6e039edca44afb93c262f3084ab72a9c",
        "model",
        "models/molmoact2",
    ),
    HubAsset(
        "molmoact2_pretrain",
        "allenai/MolmoAct2-Pretrain",
        "a05effca9ba36c1177359b42a9d5d7a4568dbe3c",
        "model",
        "models/molmoact2_pretrain",
    ),
    HubAsset(
        "qwen35_4b",
        "Qwen/Qwen3.5-4B",
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "model",
        "models/qwen35_4b",
    ),
    HubAsset(
        "qwen36_27b",
        "Qwen/Qwen3.6-27B",
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "model",
        "models/qwen36_27b",
    ),
)

OBJECT_LENS_MODEL_ASSETS = (
    HubAsset(
        "grounding_dino",
        "IDEA-Research/grounding-dino-tiny",
        "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "model",
        "models/grounding_dino",
    ),
    HubAsset(
        "sam_vit_base",
        "facebook/sam-vit-base",
        "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f",
        "model",
        "models/sam_vit_base",
    ),
)
ACTION_SUPERVISION_MODEL_ASSETS = (
    OBJECT_LENS_MODEL_ASSETS[0],
    HubAsset(
        "slimsam_50_uniform",
        "nielsr/slimsam-50-uniform",
        "2a2699253149cf2eaf4097f9ff925ce41df4a2b9",
        "model",
        "models/slimsam_50_uniform",
    ),
    HubAsset(
        "depth_anything_v2_small_hf",
        "depth-anything/Depth-Anything-V2-Small-hf",
        "5426e4f0f36572d16453bbda7a8389317b1bef99",
        "model",
        "models/depth_anything_v2_small_hf",
    ),
)
AUXILIARY_MODEL_ASSETS = tuple(
    {
        asset.key: asset
        for asset in (*OBJECT_LENS_MODEL_ASSETS, *ACTION_SUPERVISION_MODEL_ASSETS)
    }.values()
)
ALL_MODEL_ASSETS = MODEL_ASSETS + AUXILIARY_MODEL_ASSETS

DATASET_ASSETS = {
    "coco": (),
    "spatialtunnel": SPATIALTUNNEL_ASSETS,
    "embspatial": EMBSPATIAL_ASSETS,
    "cvbench": CVBENCH_ASSETS,
    "spinbench": SPINBENCH_ASSETS,
    "hstar": HSTAR_ASSETS,
}

GROUPS = {
    "models": MODEL_ASSETS,
    "object_lens": OBJECT_LENS_MODEL_ASSETS,
    "action_supervision": ACTION_SUPERVISION_MODEL_ASSETS,
    "datasets": tuple(asset for assets in DATASET_ASSETS.values() for asset in assets),
}


def bind_asset_root(project_root: Path, asset_root: Path) -> Path:
    """Bind ``PROJECT/.cache/assets`` to a user-selected persistent root."""
    project = project_root.resolve()
    requested = asset_root.expanduser().resolve()
    requested.mkdir(parents=True, exist_ok=True)
    stable = project / ".cache/assets"
    stable.parent.mkdir(parents=True, exist_ok=True)
    if stable.is_symlink():
        if stable.resolve() != requested:
            raise ValueError(f"Asset link {stable} already targets {stable.resolve()}")
    elif stable.exists():
        if stable.resolve() != requested:
            raise ValueError(
                f"Asset directory already exists and is not {requested}: {stable}"
            )
    elif stable != requested:
        stable.symlink_to(requested, target_is_directory=True)
    return requested


def acquire_hub_assets(
    asset_root: Path,
    assets: Iterable[HubAsset],
    *,
    dry_run: bool = False,
) -> dict[str, object]:
    """Download selected Hub assets and update their canonical manifest."""
    records = download_hub_assets(asset_root, assets, dry_run=dry_run)
    if dry_run:
        return {"dry_run": True, "assets": records}
    manifest_path = asset_root / "hub_assets.json"
    merged = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        merged = {row["key"]: row for row in previous.get("assets", [])}
    merged.update({row["key"]: row for row in records})
    manifest = {
        "schema_version": "1.0.0",
        "assets": [merged[key] for key in sorted(merged)],
    }
    atomic_json(manifest_path, manifest)
    return manifest


def selected_assets(
    groups: Iterable[str], datasets: Iterable[str] = ()
) -> tuple[HubAsset, ...]:
    """Expand explicit asset groups or dataset names without duplicates."""
    names = tuple(groups)
    dataset_names = tuple(datasets)
    assets = []
    if dataset_names:
        unknown = set(dataset_names) - set(DATASET_ASSETS)
        if unknown:
            raise ValueError(f"Unknown datasets: {sorted(unknown)}")
        assets.extend(asset for name in dataset_names for asset in DATASET_ASSETS[name])
    if (not names and not dataset_names) or "all" in names:
        names = tuple(GROUPS)
    unknown = set(names) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown asset groups: {sorted(unknown)}")
    assets.extend(asset for name in names for asset in GROUPS[name])
    by_key = {asset.key: asset for asset in assets}
    return tuple(by_key[key] for key in sorted(by_key))
