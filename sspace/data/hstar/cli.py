"""Build the frozen H* HOS panorama multi-view dataset (PRE-06)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .pairs import (
    PUBLISHED_PAIRS_SHA256,
    enumerate_hos_candidates,
    select_balanced_hos_pairs,
    write_hos_dataset,
)


def main() -> None:
    """Build, balance, render, and serialize PRE-06."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        from sspace.experiments.multi_view.hstar.hstar_multiview import load_hstar_multiview
        from sspace.run_records import file_sha256

        if file_sha256(args.output_dir / "pairs.jsonl") != PUBLISHED_PAIRS_SHA256:
            raise ValueError("Existing H* data differs from the released dataset")
        load_hstar_multiview(args.output_dir)
        print(f"Reusing validated H* data: {args.output_dir}")
        return
    candidates = enumerate_hos_candidates(args.source_dir)
    print(
        f"candidates={len(candidates)} labels="
        f"{dict(Counter(row.label for row in candidates))}",
        flush=True,
    )
    selected = select_balanced_hos_pairs(candidates)
    manifest = write_hos_dataset(
        selected,
        args.output_dir,
        source_archive=args.source_dir.parent / "hos_train.zip",
    )
    print(manifest, flush=True)


if __name__ == "__main__":
    main()
