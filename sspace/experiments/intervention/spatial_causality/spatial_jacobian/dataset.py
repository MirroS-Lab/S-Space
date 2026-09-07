"""Deterministic spatial-pair construction and offline SQLite access."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import random
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from matplotlib.colors import rgb_to_hsv
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, find_objects, label

from .constants import CATEGORY_ORDER, GROUP_MAP, OPPOSITE


LOGGER = logging.getLogger(__name__)

OBJECT_PATTERNS = [
    re.compile(r"between\s+(.+?)\s+and\s+(.+?)\s+in", re.IGNORECASE),
    re.compile(r"of\s+(.+?)\s+and\s+(.+?)\s+in", re.IGNORECASE),
    re.compile(r"positions\s+of\s+(.+?)\s+and\s+(.+?)\s+interact", re.IGNORECASE),
    re.compile(r"How\s+are\s+(.+?)\s+and\s+(.+?)\s+positioned", re.IGNORECASE),
    re.compile(r"arrangement\s+of\s+(.+?)\s+and\s+(.+?)\s+in", re.IGNORECASE),
]

SHORT_TEMPLATES = {
    "horizontal": "Is the {obj1} to the left or right of the {obj2}? Answer with only one word.",
    "vertical": "Is the {obj1} above or below the {obj2}? Answer with only one word.",
    "distance": "Compared to {obj2}, is {obj1} far or close from you? Answer with only one word.",
}


@dataclass(frozen=True)
class Sample:
    dataset: str
    sample_order: int
    index: int
    question_id: str
    category: str
    group: str
    obj1: str
    obj2: str
    original_question: str
    swapped_question: str
    original_answer: str
    swapped_answer: str
    paper_split: str | None
    query_center_y: float | None
    reference_center_y: float | None
    image_sha256: str
    image_bytes: bytes


def extract_objects(question: str) -> tuple[str, str]:
    for pattern in OBJECT_PATTERNS:
        match = pattern.search(question)
        if match:
            return match.group(1).strip(), match.group(2).strip()
    raise ValueError(f"Could not extract objects from: {question}")


def load_swap_pairs(path: Path, seed: int) -> list[dict[str, Any]]:
    """Reproduce the original row-order-dependent distance-reference sampling."""
    rng = random.Random(seed)
    frame = pd.read_csv(path, sep="\t")
    pairs: list[dict[str, Any]] = []

    def valid(value: object) -> bool:
        return bool(value) and str(value).strip().lower() not in {
            "unknown",
            "n/a",
            "",
            "nan",
        }

    for row in frame.to_dict("records"):
        category = str(row["category"]).lower()
        if category in {"under", "beneath"}:
            category = "below"
        if category not in GROUP_MAP:
            continue
        try:
            if category in {"left", "right", "above", "below"}:
                obj1, obj2 = extract_objects(str(row["question"]))
            else:
                answer_key = str(row["answer"])
                options = {key: row[key] for key in ("A", "B", "C", "D")}
                obj1 = str(options[answer_key])
                candidates = [
                    str(value)
                    for key, value in options.items()
                    if key != answer_key and valid(value)
                ]
                if not valid(obj1) or not candidates:
                    continue
                obj2 = rng.choice(candidates)
            group = GROUP_MAP[category]
            pairs.append(
                {
                    "index": int(row["index"]),
                    "question_id": str(row["question_id"]),
                    "category": category,
                    "group": group,
                    "obj1": obj1,
                    "obj2": obj2,
                    "original_question": SHORT_TEMPLATES[group].format(
                        obj1=obj1, obj2=obj2
                    ),
                    "swapped_question": SHORT_TEMPLATES[group].format(
                        obj1=obj2, obj2=obj1
                    ),
                    "original_answer": category,
                    "swapped_answer": OPPOSITE[category],
                    "image_base64": str(row["image"]),
                }
            )
        except (ValueError, KeyError) as exc:
            LOGGER.warning("Skipping index %s: %s", row.get("index"), exc)
    return pairs


def balanced_pairs(
    pairs: Sequence[Mapping[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[str(pair["category"])].append(dict(pair))
    output: list[dict[str, Any]] = []
    for category in CATEGORY_ORDER:
        values = grouped[category]
        if len(values) < limit:
            raise ValueError(f"Need {limit} {category} samples, found {len(values)}")
        if len(values) > limit:
            values = rng.sample(values, limit)
        output.extend(values)
    return output


def dataset_fingerprint(pairs: Sequence[Mapping[str, Any]]) -> str:
    identity = [
        [
            str(pair["index"]),
            str(pair["category"]),
            str(pair["original_question"]),
            str(pair["swapped_question"]),
        ]
        for pair in pairs
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _normalise_objects(value: Any) -> dict[str, Sequence[float]]:
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, Mapping):
        names, boxes = value.get("name", []), value.get("bbox", [])
        return {
            str(name): list(map(float, box))
            for name, box in zip(names, boxes, strict=True)
        }
    if isinstance(value, (list, np.ndarray)):
        output = {}
        for item in value:
            if isinstance(item, Mapping):
                output[str(item["name"])] = list(map(float, item["bbox"]))
        return output
    raise TypeError(f"Unsupported objects annotation type: {type(value)!r}")


def embspatial_centers(annotation_path: Path) -> dict[str, dict[str, float]]:
    frame = pd.read_parquet(annotation_path, columns=["question_id", "objects"])
    result: dict[str, dict[str, float]] = {}
    for row in frame.to_dict("records"):
        boxes = _normalise_objects(row["objects"])
        result[str(row["question_id"])] = {
            name: float(box[1] + box[3] / 2.0) for name, box in boxes.items()
        }
    return result


HUES = {
    "red": 0.0,
    "yellow": 1 / 6,
    "green": 1 / 3,
    "cyan": 1 / 2,
    "blue": 2 / 3,
    "magenta": 5 / 6,
}


def _components(image: Image.Image, color: str) -> list[tuple[float, float, float]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    hsv = rgb_to_hsv(rgb)
    if color == "black":
        mask = binary_opening(hsv[..., 2] < 0.23, np.ones((5, 5)))
    else:
        distance = np.abs(hsv[..., 0] - HUES[color])
        distance = np.minimum(distance, 1 - distance)
        mask = binary_opening(
            (distance < 0.075) & (hsv[..., 1] > 0.25) & (hsv[..., 2] > 0.12),
            np.ones((2, 2)),
        )
    mask = binary_closing(mask, np.ones((3, 3)))
    labels, _ = label(mask)
    output = []
    for component, slices in enumerate(find_objects(labels), 1):
        if slices is None:
            continue
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        area = int((labels[slices] == component).sum())
        if area > 80 and 8 < height < 250 and 8 < width < 250:
            center_y = (slices[0].start + slices[0].stop - 1) / 2
            output.append((area / (height * width), center_y, float(area)))
    return sorted(output, key=lambda item: item[2], reverse=True)


def spatialtunnel_center(image_bytes: bytes, object_name: str) -> float:
    color, shape = object_name.split()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    candidates = _components(image, color)
    if not candidates:
        raise ValueError(f"Could not locate {object_name}")
    if len(candidates) == 1:
        return candidates[0][1]
    chosen = (
        max(candidates, key=lambda item: item[0])
        if shape == "cube"
        else min(candidates, key=lambda item: item[0])
    )
    return chosen[1]


def add_paper_splits(
    dataset: str,
    pairs: Sequence[dict[str, Any]],
    emb_centers: Mapping[str, Mapping[str, float]] | None,
) -> None:
    for pair in pairs:
        if pair["group"] != "distance":
            pair.update(paper_split=None, query_center_y=None, reference_center_y=None)
            continue
        image_bytes = base64.b64decode(pair["image_base64"])
        if dataset == "embspatial":
            if emb_centers is None:
                raise ValueError("EmbSpatial centers are required")
            centers = emb_centers[str(pair["question_id"])]
            query_y = float(centers[str(pair["obj1"])])
            reference_y = float(centers[str(pair["obj2"])])
        else:
            query_y = spatialtunnel_center(image_bytes, str(pair["obj1"]))
            reference_y = spatialtunnel_center(image_bytes, str(pair["obj2"]))
        far_y, near_y = (
            (query_y, reference_y)
            if pair["category"] == "far"
            else (reference_y, query_y)
        )
        split = (
            "consistent" if far_y < near_y else ("counter" if far_y > near_y else "tie")
        )
        pair.update(
            paper_split=split, query_center_y=query_y, reference_center_y=reference_y
        )


SCHEMA = """
PRAGMA journal_mode=OFF;
PRAGMA synchronous=OFF;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE images (sha256 TEXT PRIMARY KEY, bytes BLOB NOT NULL);
CREATE TABLE samples (
    dataset TEXT NOT NULL,
    sample_order INTEGER NOT NULL,
    source_index INTEGER NOT NULL,
    question_id TEXT NOT NULL,
    category TEXT NOT NULL,
    group_name TEXT NOT NULL,
    obj1 TEXT NOT NULL,
    obj2 TEXT NOT NULL,
    original_question TEXT NOT NULL,
    swapped_question TEXT NOT NULL,
    original_answer TEXT NOT NULL,
    swapped_answer TEXT NOT NULL,
    paper_split TEXT,
    query_center_y REAL,
    reference_center_y REAL,
    image_sha256 TEXT NOT NULL REFERENCES images(sha256),
    PRIMARY KEY (dataset, source_index, category)
);
CREATE UNIQUE INDEX samples_order ON samples(dataset, sample_order);
CREATE TABLE historical_generation (
    dataset TEXT NOT NULL,
    source_index INTEGER NOT NULL,
    category TEXT NOT NULL,
    model_id TEXT NOT NULL,
    raw_prediction TEXT NOT NULL,
    parsed_prediction TEXT NOT NULL,
    correct INTEGER NOT NULL,
    PRIMARY KEY(dataset, source_index, category)
);
"""


def write_database(
    path: Path,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    metadata: Mapping[str, Any],
    historical: pd.DataFrame,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temp.unlink(missing_ok=True)
    connection = sqlite3.connect(temp)
    try:
        connection.executescript(SCHEMA)
        connection.executemany(
            "INSERT INTO metadata(key,value_json) VALUES (?,?)",
            [
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in metadata.items()
            ],
        )
        seen_images: set[str] = set()
        for dataset, pairs in datasets.items():
            for sample_order, pair in enumerate(pairs):
                image_bytes = base64.b64decode(str(pair["image_base64"]))
                with Image.open(io.BytesIO(image_bytes)) as image:
                    image.verify()
                image_sha = hashlib.sha256(image_bytes).hexdigest()
                if image_sha not in seen_images:
                    connection.execute(
                        "INSERT INTO images(sha256,bytes) VALUES (?,?)",
                        (image_sha, image_bytes),
                    )
                    seen_images.add(image_sha)
                connection.execute(
                    """INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        dataset,
                        sample_order,
                        int(pair["index"]),
                        str(pair["question_id"]),
                        str(pair["category"]),
                        str(pair["group"]),
                        str(pair["obj1"]),
                        str(pair["obj2"]),
                        str(pair["original_question"]),
                        str(pair["swapped_question"]),
                        str(pair["original_answer"]),
                        str(pair["swapped_answer"]),
                        pair.get("paper_split"),
                        pair.get("query_center_y"),
                        pair.get("reference_center_y"),
                        image_sha,
                    ),
                )
        connection.executemany(
            """INSERT INTO historical_generation VALUES (?,?,?,?,?,?,?)""",
            [
                (
                    "embspatial",
                    int(row["index"]),
                    str(row["category"]),
                    "allenai/MolmoAct2-Pretrain",
                    str(row["raw_prediction"]),
                    str(row["parsed_prediction"]),
                    int(bool(row["correct"])),
                )
                for row in historical.to_dict("records")
            ],
        )
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {result}")
    except BaseException:
        connection.close()
        temp.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temp, path)


def read_metadata(path: Path) -> dict[str, Any]:
    with sqlite3.connect(path) as connection:
        return {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value_json FROM metadata")
        }


def load_samples(path: Path, datasets: Iterable[str] | None = None) -> list[Sample]:
    selected = list(datasets or [])
    where, args = "", []
    if selected:
        where = "WHERE s.dataset IN (%s)" % ",".join("?" for _ in selected)
        args = selected
    query = f"""
        SELECT s.dataset,s.sample_order,s.source_index,s.question_id,s.category,s.group_name,
               s.obj1,s.obj2,s.original_question,s.swapped_question,s.original_answer,s.swapped_answer,
               s.paper_split,s.query_center_y,s.reference_center_y,s.image_sha256,i.bytes
        FROM samples s JOIN images i ON i.sha256=s.image_sha256
        {where} ORDER BY s.dataset,s.sample_order
    """
    with sqlite3.connect(path) as connection:
        return [Sample(*row) for row in connection.execute(query, args)]


def load_historical_generation(path: Path) -> pd.DataFrame:
    with sqlite3.connect(path) as connection:
        return pd.read_sql_query(
            """SELECT dataset,source_index AS 'index',category,model_id,raw_prediction,
                      parsed_prediction,CAST(correct AS INTEGER) AS correct
               FROM historical_generation ORDER BY source_index""",
            connection,
        )
