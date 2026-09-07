"""Download pinned Release models and public experiment datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from .assets import (
    ALL_MODEL_ASSETS,
    DATASET_ASSETS,
    GROUPS,
    acquire_hub_assets,
    bind_asset_root,
    selected_assets,
)
from .coco.assets import prepare_coco_assets
from .cvbench.download import prepare as prepare_cvbench
from .embspatial.download import prepare as prepare_embspatial
from .hstar.download import prepare as prepare_hstar
from .spinbench.download import prepare as prepare_spinbench
from .spatialtunnel.download import prepare as prepare_spatialtunnel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--group",
        action="append",
        choices=("all", *GROUPS),
    )
    selection.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_ASSETS),
        help="Acquire one dataset instead of a complete asset group.",
    )
    model_assets = {asset.key: asset for asset in ALL_MODEL_ASSETS}
    selection.add_argument(
        "--model",
        choices=tuple(model_assets),
        help="Acquire one pinned model; this option cannot be repeated.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    root = args.asset_root.expanduser().resolve()
    if not args.dry_run:
        root = bind_asset_root(project_root, root)
    groups = tuple(args.group or ())
    datasets = tuple(
        args.dataset
        or (tuple(DATASET_ASSETS) if "all" in groups or "datasets" in groups else ())
    )
    assets = (
        (model_assets[args.model],)
        if args.model is not None
        else selected_assets(groups, datasets)
    )
    acquire_hub_assets(root, assets, dry_run=args.dry_run)
    if "coco" in datasets:
        if args.dry_run:
            print(f"coco:2017 frozen 6000+1800 images -> {root / 'datasets/coco2017'}")
        else:
            prepare_coco_assets(root, args.workers)
    if "spinbench" in datasets and not args.dry_run:
        prepare_spinbench(root)
    if "spatialtunnel" in datasets and not args.dry_run:
        prepare_spatialtunnel(root)
    if "embspatial" in datasets and not args.dry_run:
        prepare_embspatial(root)
    if "cvbench" in datasets and not args.dry_run:
        prepare_cvbench(root)
    if "hstar" in datasets and not args.dry_run:
        prepare_hstar(root)


if __name__ == "__main__":
    main()
