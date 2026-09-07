"""Publish the exact separate COCO-6000 and COCO-1800 datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .frozen import publish_frozen_coco_datasets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    train, validation = publish_frozen_coco_datasets(args.asset_root, project_root)
    print(
        json.dumps(
            {
                "train": train["dataset_fingerprint"],
                "validation": validation["dataset_fingerprint"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
