"""Prepare the frozen EmbSpatial-1200 benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from .prepare import prepare_embspatial1200


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tsv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_embspatial1200(args.source_tsv, args.output)


if __name__ == "__main__":
    main()
