"""Finalize visual review for one unchanged H* artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pairs import finalize_hos_review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--reviewed-by", required=True)
    parser.add_argument("--expected-dataset-fingerprint", required=True)
    args = parser.parse_args()
    print(
        finalize_hos_review(
            args.dataset_dir, args.reviewed_by, args.expected_dataset_fingerprint
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
