#!/usr/bin/env python3
"""Build an exhaustive MMSI-Bench manifest for eight geographic directions.

Selection is deliberately independent of model output and annotated reasoning:
keep every source QA whose *gold option text* can be normalized to one of
north/south/east/west/northeast/northwest/southeast/southwest.  Question type
is recorded but not used as a filter.  Purely egocentric answers such as
front/back/left/right are excluded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[5]
SOURCE_REPOSITORY = "RunsenXu/MMSI-Bench"
SOURCE_REVISION = "ec7c92bfaf7728fcca1d61e3e224e190af309436"
SOURCE_SHA256 = "72a9c94699ca88f3eba79f330ab28965ee537bccd7cbf563265982a6ff8b74b4"
REFERENCE_MANIFEST = PROJECT_ROOT / "sspace/data/contextual_cases/mmsi_eightway/manifest.jsonl"
OPTION_RE = re.compile(
    r"(?:^|[,\n]\s*|\s)([A-D]):\s*(.*?)(?=(?:[,\n]\s*|\s)[A-D]:|$)",
    re.DOTALL,
)

COMPOUNDS = {
    "northeast": re.compile(
        r"\bnorth(?:\s*[-/]?\s*)east(?:ern)?\b|\bnortheastern?\b", re.IGNORECASE
    ),
    "northwest": re.compile(
        r"\bnorth(?:\s*[-/]?\s*)west(?:ern)?\b|\bnorthwestern?\b", re.IGNORECASE
    ),
    "southeast": re.compile(
        r"\bsouth(?:\s*[-/]?\s*)east(?:ern)?\b|\bsoutheastern?\b", re.IGNORECASE
    ),
    "southwest": re.compile(
        r"\bsouth(?:\s*[-/]?\s*)west(?:ern)?\b|\bsouthwestern?\b", re.IGNORECASE
    ),
}
BASE_RE = re.compile(r"\b(north|south|east|west)(?:ern)?\b", re.IGNORECASE)
ORDER = (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)


def source_parquet() -> Path:
    """Acquire the immutable source, including the original image bytes."""
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache/huggingface"))
    os.environ.setdefault("HF_XET_CACHE", str(PROJECT_ROOT / ".cache/huggingface/xet"))
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(
        repo_id=SOURCE_REPOSITORY,
        repo_type="dataset",
        revision=SOURCE_REVISION,
        filename="MMSI_Bench.parquet",
        local_dir=PROJECT_ROOT / ".cache/assets/datasets/mmsi_bench",
        cache_dir=PROJECT_ROOT / ".cache/assets/.hub_cache",
    ))
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("MMSI source parquet differs from the pinned SHA-256")
    return path


def validate_reference(manifest: list[dict[str, Any]], *, skip_images: bool) -> None:
    """Require the article's exact questions, labels, order and image bytes."""
    expected = [json.loads(line) for line in REFERENCE_MANIFEST.read_text().splitlines()]
    if skip_images:
        expected = [{**row, "image_sha256": [""] * len(row["image_paths"])} for row in expected]
    if manifest != expected:
        raise ValueError("MMSI selection differs from the released 191-question manifest")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answer_text(question: str, letter: str) -> str:
    marker = question.find("Options:")
    if marker < 0:
        return ""
    options = {
        key: value.strip() for key, value in OPTION_RE.findall(question[marker + 8 :])
    }
    return options.get(letter.strip().upper(), "")


def normalize_geographic_direction(text: str) -> str:
    """Return one canonical eight-way direction, or an empty string.

    Explicit compounds are preferred.  Phrases such as "west wall, north
    wall" are interpreted as northwest when they contain exactly one vertical
    and one horizontal geographic component.
    """

    explicit = [name for name, pattern in COMPOUNDS.items() if pattern.search(text)]
    if len(explicit) == 1:
        return explicit[0]
    if len(explicit) > 1:
        return ""
    words = {match.group(1).lower() for match in BASE_RE.finditer(text)}
    vertical = [value for value in ("north", "south") if value in words]
    horizontal = [value for value in ("east", "west") if value in words]
    if len(vertical) > 1 or len(horizontal) > 1:
        return ""
    if vertical and horizontal:
        return vertical[0] + horizontal[0]
    if vertical:
        return vertical[0]
    if horizontal:
        return horizontal[0]
    return ""


def contains_geographic_direction(text: str) -> bool:
    """Return whether text contains a cardinal or intercardinal direction."""

    return bool(
        BASE_RE.search(text)
        or any(pattern.search(text) for pattern in COMPOUNDS.values())
    )


def normalize_difficulty(value: Any) -> int:
    labels = {"easy": 0, "medium": 1, "hard": 2}
    text = str(value).strip().lower()
    return labels[text] if text in labels else int(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection",
        choices=("gold-answer", "geographic-context"),
        default="gold-answer",
        help=(
            "gold-answer keeps QAs whose correct option is one geographic direction; "
            "geographic-context keeps QAs whose question/options or official thought "
            "contains a north/south/east/west term"
        ),
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        help="Optional local MMSI_Bench.parquet; default downloads the pinned snapshot.",
    )
    parser.add_argument("--skip-images", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = args.output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    import pandas as pd

    parquet = args.parquet if args.parquet is not None else source_parquet()
    rows = pd.read_parquet(parquet).to_dict(orient="records")
    if len(rows) != 1000:
        raise ValueError(f"Expected 1000 rows, received {len(rows)}")

    selected: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: int(value["id"])):
        gold_text = answer_text(str(row["question"]), str(row["answer"]))
        direction = normalize_geographic_direction(gold_text)
        has_geographic_context = contains_geographic_direction(
            str(row["question"])
        ) or contains_geographic_direction(str(row["thought"]))
        if args.selection == "gold-answer" and not direction:
            continue
        if args.selection == "geographic-context" and not has_geographic_context:
            continue
        selected.append({**row, "answer_text": gold_text, "gold_direction": direction})

    manifest: list[dict[str, Any]] = []
    for row in selected:
        image_paths: list[str] = []
        image_hashes: list[str] = []
        for image_index, image in enumerate(row["images"]):
            path = image_dir / f"mmsi_{int(row['id']):04d}_{image_index}.jpg"
            if not args.skip_images:
                payload = image["bytes"] if isinstance(image, dict) else image
                path.write_bytes(bytes(payload))
            image_paths.append(
                str(path.relative_to(args.output_dir)).replace("\\", "/")
            )
            image_hashes.append(sha256_file(path) if not args.skip_images else "")
        manifest.append(
            {
                "mmsi_id": int(row["id"]),
                "question_type": str(row["question_type"]),
                "difficulty": normalize_difficulty(row["difficulty"]),
                "answer_letter": str(row["answer"]).upper(),
                "answer_text": row["answer_text"],
                "gold_direction": row["gold_direction"],
                "image_paths": image_paths,
                "image_sha256": image_hashes,
                "prompt": f"Question: {row['question']}",
                "official_thought": str(row["thought"]),
            }
        )

    if args.selection == "gold-answer":
        validate_reference(manifest, skip_images=args.skip_images)
    (args.output_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest),
        encoding="utf-8",
    )
    by_type: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        by_type[row["question_type"]][row["gold_direction"]] += 1
    selection_description = (
        "gold option normalizes to one of eight geographic directions; no question-type or thought filter"
        if args.selection == "gold-answer"
        else "question/options or official thought contains a geographic direction token; no question-type or gold-answer filter"
    )
    config = {
        "dataset": SOURCE_REPOSITORY,
        "source_revision": SOURCE_REVISION if args.parquet is None else None,
        "source_sha256": sha256_file(parquet),
        "split": "test",
        "source_rows": len(rows),
        "selected_rows": len(manifest),
        "selection": selection_description,
        "excluded_egocentric_only": ["front", "back", "left", "right"],
        "direction_order": list(ORDER),
        "direction_counts": dict(Counter(row["gold_direction"] for row in manifest)),
        "question_type_counts": dict(Counter(row["question_type"] for row in manifest)),
        "question_type_by_direction": {
            key: dict(value) for key, value in sorted(by_type.items())
        },
        "images_downloaded": not args.skip_images,
        "source_parquet": str(parquet),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
