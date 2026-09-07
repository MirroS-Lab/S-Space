"""Build the frozen balanced EmbSpatial-1200 SQLite artifact."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import random
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


SEED = 42
SAMPLES_PER_ENDPOINT = 200
ENDPOINTS = ("left", "right", "above", "below", "far", "close")
GROUP = {
    "left": "horizontal",
    "right": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "far": "distance",
    "close": "distance",
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "far": "close",
    "close": "far",
}
PROMPTS = {
    "horizontal": "Is the {query} to the left or right of the {reference}? Answer with only one word.",
    "vertical": "Is the {query} above or below the {reference}? Answer with only one word.",
    "distance": "Compared to {reference}, is {query} far or close from you? Answer with only one word.",
}
OBJECT_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"between\s+(.+?)\s+and\s+(.+?)\s+in",
        r"of\s+(.+?)\s+and\s+(.+?)\s+in",
        r"positions\s+of\s+(.+?)\s+and\s+(.+?)\s+interact",
        r"How\s+are\s+(.+?)\s+and\s+(.+?)\s+positioned",
        r"arrangement\s+of\s+(.+?)\s+and\s+(.+?)\s+in",
    )
)
EXPECTED_FINGERPRINT = (
    "5216d96aed866f14428ff6ad4382f6be3ec661007489dbb33db3e3a09c541067"
)
SELECTION_PROTOCOL = "legacy_seed42_unique_evaluator_input_repair"
EXPECTED_EXCLUSION_COUNTS = {
    "duplicate_evaluator_input": 12,
    "conflicting_evaluator_label": 4,
}


def validate_embspatial1200(path: Path) -> None:
    """Validate the existing frozen EmbSpatial SQLite artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(path) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("EmbSpatial SQLite integrity check failed")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        required_tables = {
            "images",
            "metadata",
            "samples",
            "selection_exclusions",
        }
        if not required_tables.issubset(tables):
            raise ValueError("EmbSpatial SQLite schema is obsolete or incomplete")
        count = connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        endpoints = dict(
            connection.execute(
                "SELECT category, COUNT(*) FROM samples GROUP BY category"
            )
        )
        raw = connection.execute(
            "SELECT value_json FROM metadata WHERE key='dataset_fingerprints'"
        ).fetchone()
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key,value_json FROM metadata")
        }
        unique_inputs = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT image_sha256,group_name,obj1,obj2
                FROM samples
                GROUP BY image_sha256,group_name,obj1,obj2
            )
            """
        ).fetchone()[0]
        exclusion_counts = dict(
            connection.execute(
                "SELECT reason,COUNT(*) FROM selection_exclusions GROUP BY reason"
            )
        )
    if count != 1200 or endpoints != {name: 200 for name in ENDPOINTS}:
        raise ValueError("EmbSpatial row balance changed")
    if raw is None or json.loads(raw[0]) != {"embspatial": EXPECTED_FINGERPRINT}:
        raise ValueError("EmbSpatial fingerprint changed")
    if (
        metadata.get("format_version") != 2
        or metadata.get("selection_protocol") != SELECTION_PROTOCOL
    ):
        raise ValueError("EmbSpatial selection protocol changed")
    if unique_inputs != 1200:
        raise ValueError("EmbSpatial evaluator inputs must be unique")
    if exclusion_counts != EXPECTED_EXCLUSION_COUNTS:
        raise ValueError("EmbSpatial selection exclusions changed")


def _objects(question: str) -> tuple[str, str]:
    for pattern in OBJECT_PATTERNS:
        if match := pattern.search(question):
            return match.group(1).strip(), match.group(2).strip()
    raise ValueError(f"Cannot extract two objects from question: {question}")


def _valid_option(value: object) -> bool:
    return bool(value) and str(value).strip().lower() not in {"unknown", "n/a", "nan"}


def _load_pairs(path: Path) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    rows = pd.read_csv(path, sep="\t").to_dict("records")
    pairs = []
    for row in rows:
        endpoint = str(row["category"]).lower()
        endpoint = "below" if endpoint in {"under", "beneath"} else endpoint
        if endpoint not in GROUP:
            continue
        if endpoint in {"left", "right", "above", "below"}:
            query, reference = _objects(str(row["question"]))
        else:
            answer_key = str(row["answer"])
            options = {key: row[key] for key in ("A", "B", "C", "D")}
            query = str(options[answer_key])
            candidates = [
                str(value)
                for key, value in options.items()
                if key != answer_key and _valid_option(value)
            ]
            if not _valid_option(query) or not candidates:
                continue
            reference = rng.choice(candidates)
        group = GROUP[endpoint]
        encoded_image = str(row["image"])
        pairs.append(
            {
                "source_index": int(row["index"]),
                "question_id": str(row["question_id"]),
                "endpoint": endpoint,
                "group": group,
                "query": query,
                "reference": reference,
                "prompt": PROMPTS[group].format(query=query, reference=reference),
                "swapped_prompt": PROMPTS[group].format(
                    query=reference, reference=query
                ),
                "image": encoded_image,
                "image_sha256": hashlib.sha256(
                    base64.b64decode(encoded_image)
                ).hexdigest(),
            }
        )
    return pairs


def _evaluator_input_key(pair: dict[str, Any]) -> tuple[str, str, str, str]:
    """Identify the exact image and ordered relation rendered for a model."""
    return (
        str(pair["image_sha256"]),
        str(pair["group"]),
        str(pair["query"]),
        str(pair["reference"]),
    )


def _legacy_select(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reproduce the original seed-42, 200-per-endpoint row selection."""
    rng = random.Random(SEED)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        grouped[pair["endpoint"]].append(pair)
    selected = []
    for endpoint in ENDPOINTS:
        values = grouped[endpoint]
        if len(values) < SAMPLES_PER_ENDPOINT:
            raise ValueError(f"Only {len(values)} rows are available for {endpoint}")
        selected.extend(rng.sample(values, SAMPLES_PER_ENDPOINT))
    return selected


def _select(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repair only invalid legacy rows and deterministically restore balance.

    The legacy selection remains the frozen starting point. Repeated evaluator
    inputs retain their first selected occurrence. Every selected occurrence of
    an input with conflicting source labels is removed. A separate seed-42
    stream samples unique replacement inputs from rows outside the legacy set.

    Returns:
        The revised 1,200 rows and one explicit reason for every removed legacy
        row.
    """
    grouped_inputs: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
        list
    )
    for pair in pairs:
        grouped_inputs[_evaluator_input_key(pair)].append(pair)
    conflicting = {
        key
        for key, values in grouped_inputs.items()
        if len({str(value["endpoint"]) for value in values}) > 1
    }

    legacy = _legacy_select(pairs)
    legacy_sources = {int(pair["source_index"]) for pair in legacy}
    refill_rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    used_inputs: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for endpoint in ENDPOINTS:
        kept = []
        for pair in (value for value in legacy if value["endpoint"] == endpoint):
            key = _evaluator_input_key(pair)
            retained = used_inputs.get(key)
            if key in conflicting:
                reason = "conflicting_evaluator_label"
            elif retained is not None:
                reason = "duplicate_evaluator_input"
            else:
                kept.append(pair)
                used_inputs[key] = pair
                continue
            exclusions.append(
                {
                    "source_index": int(pair["source_index"]),
                    "question_id": str(pair["question_id"]),
                    "endpoint": str(pair["endpoint"]),
                    "reason": reason,
                    "retained_source_index": (
                        None if retained is None else int(retained["source_index"])
                    ),
                }
            )

        pool = []
        pool_inputs = set()
        for pair in pairs:
            key = _evaluator_input_key(pair)
            if (
                pair["endpoint"] != endpoint
                or int(pair["source_index"]) in legacy_sources
                or key in conflicting
                or key in used_inputs
                or key in pool_inputs
            ):
                continue
            pool.append(pair)
            pool_inputs.add(key)
        needed = SAMPLES_PER_ENDPOINT - len(kept)
        if len(pool) < needed:
            raise ValueError(
                f"Only {len(pool)} unique refill rows exist for {endpoint}"
            )
        replacements = refill_rng.sample(pool, needed)
        for pair in replacements:
            used_inputs[_evaluator_input_key(pair)] = pair
        selected.extend((*kept, *replacements))

    counts = Counter(str(pair["endpoint"]) for pair in selected)
    if counts != {endpoint: SAMPLES_PER_ENDPOINT for endpoint in ENDPOINTS}:
        raise ValueError(f"EmbSpatial repaired endpoint counts differ: {dict(counts)}")
    if len(used_inputs) != len(selected) or len(selected) != 1200:
        raise ValueError("EmbSpatial repaired evaluator inputs must be unique")
    return selected, exclusions


def _fingerprint(pairs: list[dict[str, Any]]) -> str:
    identity = [
        [
            str(pair["source_index"]),
            pair["endpoint"],
            pair["prompt"],
            pair["swapped_prompt"],
        ]
        for pair in pairs
    ]
    encoded = json.dumps(identity, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def prepare_embspatial1200(source_tsv: Path, output_path: Path) -> None:
    """Build and validate the exact seed-42, 200-per-endpoint dataset."""
    if output_path.exists():
        validate_embspatial1200(output_path)
        return
    selected, exclusions = _select(_load_pairs(source_tsv))
    fingerprint = _fingerprint(selected)
    if fingerprint != EXPECTED_FINGERPRINT:
        raise ValueError(f"EmbSpatial selection fingerprint changed: {fingerprint}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    temporary = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    connection = sqlite3.connect(temporary)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
            CREATE TABLE images (sha256 TEXT PRIMARY KEY, bytes BLOB NOT NULL);
            CREATE TABLE samples (
                dataset TEXT NOT NULL, sample_order INTEGER NOT NULL,
                source_index INTEGER NOT NULL, question_id TEXT NOT NULL,
                category TEXT NOT NULL, group_name TEXT NOT NULL,
                obj1 TEXT NOT NULL, obj2 TEXT NOT NULL,
                original_question TEXT NOT NULL, swapped_question TEXT NOT NULL,
                original_answer TEXT NOT NULL, swapped_answer TEXT NOT NULL,
                paper_split TEXT, query_center_y REAL, reference_center_y REAL,
                image_sha256 TEXT NOT NULL REFERENCES images(sha256),
                PRIMARY KEY (dataset, source_index, category));
            CREATE UNIQUE INDEX samples_order ON samples(dataset, sample_order);
            CREATE TABLE selection_exclusions (
                source_index INTEGER PRIMARY KEY, question_id TEXT NOT NULL,
                category TEXT NOT NULL, reason TEXT NOT NULL,
                retained_source_index INTEGER);
            """
        )
        metadata = {
            "format_version": 2,
            "seed": SEED,
            "samples_per_category": SAMPLES_PER_ENDPOINT,
            "selection_protocol": SELECTION_PROTOCOL,
            "dataset_fingerprints": {"embspatial": fingerprint},
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?,?)",
            [
                (key, json.dumps(value, ensure_ascii=False, sort_keys=True))
                for key, value in metadata.items()
            ],
        )
        connection.executemany(
            "INSERT INTO selection_exclusions VALUES (?,?,?,?,?)",
            [
                (
                    row["source_index"],
                    row["question_id"],
                    row["endpoint"],
                    row["reason"],
                    row["retained_source_index"],
                )
                for row in exclusions
            ],
        )
        seen = set()
        for order, pair in enumerate(selected):
            image = base64.b64decode(pair["image"])
            with Image.open(io.BytesIO(image)) as opened:
                opened.verify()
            digest = hashlib.sha256(image).hexdigest()
            if digest != pair["image_sha256"]:
                raise ValueError("EmbSpatial image hash changed after selection")
            if digest not in seen:
                connection.execute("INSERT INTO images VALUES (?,?)", (digest, image))
                seen.add(digest)
            connection.execute(
                "INSERT INTO samples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "embspatial",
                    order,
                    pair["source_index"],
                    pair["question_id"],
                    pair["endpoint"],
                    pair["group"],
                    pair["query"],
                    pair["reference"],
                    pair["prompt"],
                    pair["swapped_prompt"],
                    pair["endpoint"],
                    OPPOSITE[pair["endpoint"]],
                    None,
                    None,
                    None,
                    digest,
                ),
            )
        connection.commit()
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("EmbSpatial SQLite integrity check failed")
    except BaseException:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    connection.close()
    os.replace(temporary, output_path)
    validate_embspatial1200(output_path)
