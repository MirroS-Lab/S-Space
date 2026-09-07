"""Download the frozen InstructPart candidate subset from Dataset Viewer."""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sspace.run_records import atomic_json, file_sha256


VIEWER_ROWS = "https://datasets-server.huggingface.co/rows"
DATASET = "IffYuan/InstructPart"
REVISION = "bcd06969e32582ceeba3e1841183106027b379fb"
CONFIG = "default"
SPLIT = "train"
EXPECTED_ROWS = 1773
EXPECTED_SELECTION_SHA256 = (
    "cb6d668645f7e5bbf40bb34f0ee00af95940141c89fbfee6e9dacf9c13d08c71"
)


def _open(request: str, timeout: int) -> Any:
    """Open one Dataset Viewer URL with bounded rate-limit retries."""
    headers = {"User-Agent": "sspace-instructpart-reproduction/1.0"}
    for attempt in range(5):
        try:
            return urllib.request.urlopen(
                urllib.request.Request(request, headers=headers), timeout=timeout
            )
        except urllib.error.HTTPError as error:
            if error.code not in {429, 502, 503, 504} or attempt == 4:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = min(30.0, float(retry_after)) if retry_after else 2.0**attempt
            print(f"source request retry {attempt + 1}/4 in {delay:g}s", flush=True)
            time.sleep(delay)
    raise AssertionError("Unreachable retry state")


def _viewer_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while offset < EXPECTED_ROWS:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": CONFIG,
                "split": SPLIT,
                "offset": offset,
                "length": 100,
            }
        )
        with _open(f"{VIEWER_ROWS}?{query}", timeout=60) as response:
            value = json.load(response)
        if int(value["num_rows_total"]) != EXPECTED_ROWS:
            raise ValueError("InstructPart row count differs from the frozen source")
        page = value["rows"]
        if not page:
            raise ValueError(f"Dataset Viewer returned an empty page at {offset}")
        rows.extend(page)
        offset += len(page)
        print(f"metadata {len(rows)}/{EXPECTED_ROWS}", flush=True)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("Dataset Viewer returned too many InstructPart rows")
    return rows


def _instructions(value: Any) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("InstructPart instruction must be a non-empty list")
    first = value[0]
    if isinstance(first, str) and first.startswith("["):
        parsed = ast.literal_eval(first)
        if not isinstance(parsed, list) or not parsed:
            raise ValueError("Nested InstructPart instruction list is invalid")
        return [str(item) for item in parsed]
    return [str(item) for item in value]


def _stable_key(question_id: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def select_rows(
    rows: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Select a group-balanced deterministic prefix of InstructPart rows."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for wrapped in rows:
        row = wrapped["row"]
        group = tuple(
            str(row[field]).strip().lower()
            for field in ("object", "part", "affordance")
        )
        groups[group].append(wrapped)
    for values in groups.values():
        values.sort(key=lambda item: _stable_key(int(item["row"]["question_id"]), seed))
    group_order = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{seed}:{'|'.join(group)}".encode()
        ).hexdigest(),
    )
    selected: list[dict[str, Any]] = []
    round_index = 0
    while len(selected) < count:
        added = 0
        for group in group_order:
            values = groups[group]
            if round_index < len(values):
                selected.append(values[round_index])
                added += 1
                if len(selected) == count:
                    break
        if not added:
            raise ValueError(f"InstructPart has fewer than {count} selectable rows")
        round_index += 1
    return selected


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _source_url(value: Any) -> str:
    url = str(value).split("?", maxsplit=1)[0]
    expected = f"/IffYuan/InstructPart/--/{REVISION}/--/default/train/"
    if expected not in url:
        raise ValueError(f"InstructPart media URL is not revision-pinned: {url}")
    return url


def _download(url: str) -> bytes:
    with _open(url, timeout=120) as response:
        return response.read()


def _save_image(image: Image.Image, path: Path, image_format: str, **kwargs: Any) -> None:
    temporary = path.with_name(path.name + ".part")
    image.save(temporary, format=image_format, **kwargs)
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_sample(
    showcase_id: int,
    wrapped: dict[str, Any],
    assets_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = wrapped["row"]
    object_name = str(row["object"]).strip().lower()
    part_name = str(row["part"]).strip().lower()
    stem = f"{showcase_id:04d}_{_slug(object_name)}_{_slug(part_name)}"
    image_name = f"{stem}_image.jpg"
    mask_name = f"{stem}_mask.png"
    image_path = assets_dir / image_name
    mask_path = assets_dir / mask_name
    image_url = _source_url(row["image"]["src"])
    mask_url = _source_url(row["mask"]["src"])
    write_assets = not (image_path.is_file() and mask_path.is_file())
    if not write_assets:
        image = Image.open(image_path).convert("RGB")
        mask_source = Image.open(mask_path).convert("L")
    else:
        image = Image.open(io.BytesIO(_download(image_url))).convert("RGB")
        mask_source = Image.open(io.BytesIO(_download(mask_url))).convert("L")
    if image.size != mask_source.size:
        raise ValueError(f"Image/mask shape differs for question {row['question_id']}")
    mask = np.asarray(mask_source) > 127
    if not mask.any():
        raise ValueError(f"Empty part mask for question {row['question_id']}")
    if write_assets:
        _save_image(image, image_path, "JPEG", quality=95)
        _save_image(Image.fromarray(mask.astype(np.uint8) * 255), mask_path, "PNG")
    instructions = _instructions(row["instruction"])
    record = {
        "showcase_id": showcase_id,
        "source_row_index": int(wrapped["row_idx"]),
        "mirror_question_id": int(row["question_id"]),
        "object": object_name,
        "part": part_name,
        "affordance": str(row["affordance"]).strip().lower(),
        "action": str(row["action"]).strip().lower(),
        "instructions": instructions,
        "action_context": instructions[0].strip().rstrip(" ?!."),
        "image": image_name,
        "mask": mask_name,
        "width": image.width,
        "height": image.height,
        "source_mirror": DATASET,
        "source_image_url": image_url,
        "source_mask_url": mask_url,
    }
    assets = {
        "showcase_id": showcase_id,
        "image_sha256": file_sha256(image_path),
        "mask_sha256": file_sha256(mask_path),
    }
    return record, assets


def _validate_existing(output_dir: Path, count: int) -> bool:
    selection_path = output_dir / "selection.json"
    manifest_path = output_dir / "source_manifest.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        return False
    if file_sha256(selection_path) != EXPECTED_SELECTION_SHA256:
        raise ValueError("Existing InstructPart selection checksum differs")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        len(selection) != count
        or manifest.get("repo_id") != DATASET
        or manifest.get("revision") != REVISION
        or manifest.get("selection_sha256") != EXPECTED_SELECTION_SHA256
    ):
        raise ValueError("Existing InstructPart source manifest differs")
    expected_assets = {int(row["showcase_id"]): row for row in manifest["assets"]}
    if set(expected_assets) != {int(row["showcase_id"]) for row in selection}:
        raise ValueError("InstructPart asset manifest IDs differ")
    assets_dir = output_dir / "assets"
    for row in selection:
        expected = expected_assets[int(row["showcase_id"])]
        if file_sha256(assets_dir / row["image"]) != expected["image_sha256"]:
            raise ValueError(f"Image checksum differs for {row['showcase_id']}")
        if file_sha256(assets_dir / row["mask"]) != expected["mask_sha256"]:
            raise ValueError(f"Mask checksum differs for {row['showcase_id']}")
    return True


def prepare(output_dir: Path, count: int, seed: int, workers: int) -> None:
    """Download and verify the exact 360 historical candidate rows."""
    if count != 360 or seed != 42 or workers <= 0:
        raise ValueError("Frozen InstructPart acquisition requires count=360 and seed=42")
    output_dir.mkdir(parents=True, exist_ok=True)
    if _validate_existing(output_dir, count):
        print("verified existing InstructPart source subset", flush=True)
        return
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(exist_ok=True)
    selected = select_rows(_viewer_rows(), count, seed)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_write_sample, index, row, assets_dir)
            for index, row in enumerate(selected, start=1)
        ]
        results = []
        for index, future in enumerate(futures, start=1):
            results.append(future.result())
            print(f"assets {index}/{count}", flush=True)
    records = [result[0] for result in results]
    asset_records = [result[1] for result in results]
    atomic_json(output_dir / "selection.json", records)
    if file_sha256(output_dir / "selection.json") != EXPECTED_SELECTION_SHA256:
        raise ValueError("Downloaded InstructPart selection differs from the historical run")
    atomic_json(
        output_dir / "source_manifest.json",
        {
            "schema_version": "1.0.0",
            "repo_id": DATASET,
            "revision": REVISION,
            "config": CONFIG,
            "split": SPLIT,
            "source_row_count": EXPECTED_ROWS,
            "candidate_count": count,
            "selection_protocol": "round_robin_object_part_affordance_hash_order",
            "seed": seed,
            "selection_sha256": EXPECTED_SELECTION_SHA256,
            "assets": asset_records,
        },
    )
    print(f"complete InstructPart source subset: {count} rows", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=360)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    prepare(args.output_dir.resolve(), args.count, args.seed, args.workers)


if __name__ == "__main__":
    main()
