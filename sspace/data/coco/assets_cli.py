"""Download the pinned COCO construction assets to an explicit location."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import prepare_coco_assets


def main() -> None:
    """Run PRE-00 and print the verified asset manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    manifest = prepare_coco_assets(args.asset_root, args.workers)
    print(
        json.dumps(
            {
                "asset_root": str(args.asset_root.expanduser().resolve()),
                "train_image_count": manifest["train_image_count"],
                "validation_image_count": manifest["validation_image_count"],
                "manifest": str(
                    args.asset_root.expanduser().resolve() / "manifest.json"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
