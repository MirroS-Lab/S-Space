"""Load the frozen balanced EmbSpatial-1200 benchmark."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from sspace.experiments.common.pairwise.schema import PairwiseEvaluationSample

_BALANCED1200_FINGERPRINT = (
    "5216d96aed866f14428ff6ad4382f6be3ec661007489dbb33db3e3a09c541067"
)
_SELECTION_PROTOCOL = "legacy_seed42_unique_evaluator_input_repair"
_ENDPOINT_GROUP = {
    "left": "horizontal",
    "right": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "far": "distance",
    "close": "distance",
}


def load_embspatial_balanced1200(
    path: Path,
) -> tuple[PairwiseEvaluationSample, ...]:
    """Load the frozen balanced 1,200-row EmbSpatial paper subset.

    The source is the immutable SQLite artifact produced with seed 42 by
    selecting 200 rows for each of left, right, above, below, far, and close.
    The loader validates the database bytes, embedded dataset fingerprint,
    exact endpoint counts, sample order, object binding, and image checksums.

    Args:
        path: Frozen ``qa_embspatial.sqlite3`` dataset artifact.

    Returns:
        Exactly 1,200 validated pairwise samples in frozen sample order.

    Raises:
        ValueError: The database, metadata, schema, rows, or images differ.

    Side effects:
        Reads the SQLite database and embedded images without changing them.
    """

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        if connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("EmbSpatial-1200 database integrity check failed")
        metadata = {
            key: json.loads(value)
            for key, value in connection.execute("SELECT key, value_json FROM metadata")
        }
        if (
            metadata.get("format_version") != 2
            or metadata.get("seed") != 42
            or metadata.get("samples_per_category") != 200
            or metadata.get("selection_protocol") != _SELECTION_PROTOCOL
            or metadata.get("dataset_fingerprints", {}).get("embspatial")
            != _BALANCED1200_FINGERPRINT
        ):
            raise ValueError("EmbSpatial-1200 selection metadata differs")
        rows = connection.execute(
            """
            SELECT s.sample_order,s.source_index,s.question_id,s.category,
                   s.group_name,s.obj1,s.obj2,s.original_question,
                   s.original_answer,s.image_sha256,i.bytes
            FROM samples AS s
            JOIN images AS i ON i.sha256=s.image_sha256
            WHERE s.dataset='embspatial'
            ORDER BY s.sample_order
            """
        ).fetchall()
    if len(rows) != 1200 or [int(row[0]) for row in rows] != list(range(1200)):
        raise ValueError("EmbSpatial-1200 row count or order differs")
    samples = []
    for row in rows:
        (
            sample_order,
            source_index,
            question_id,
            endpoint,
            group,
            query,
            reference,
            prompt,
            answer,
            image_sha256,
            image_bytes,
        ) = row
        endpoint = str(endpoint)
        if (
            endpoint not in _ENDPOINT_GROUP
            or str(group) != _ENDPOINT_GROUP[endpoint]
            or str(answer) != endpoint
        ):
            raise ValueError("EmbSpatial-1200 endpoint semantics differ")
        encoded = bytes(image_bytes)
        if hashlib.sha256(encoded).hexdigest() != str(image_sha256):
            raise ValueError("EmbSpatial-1200 embedded image checksum differs")
        sample = PairwiseEvaluationSample(
            benchmark="embspatial_balanced1200",
            sample_id=str(question_id),
            split="test",
            group=str(group),
            query=str(query),
            reference=str(reference),
            gold_endpoint=endpoint,
            direct_prompt=str(prompt),
            image_sha256=str(image_sha256),
            image_bytes=encoded,
            metadata={
                "dataset_fingerprint": _BALANCED1200_FINGERPRINT,
                "sample_order": int(sample_order),
                "source_index": int(source_index),
                "selection_seed": 42,
            },
        )
        sample.validate()
        samples.append(sample)
    counts = Counter(sample.gold_endpoint for sample in samples)
    if counts != {endpoint: 200 for endpoint in _ENDPOINT_GROUP}:
        raise ValueError(f"EmbSpatial-1200 endpoint counts differ: {dict(counts)}")
    if len({sample.sample_id for sample in samples}) != 1200:
        raise ValueError("EmbSpatial-1200 sample IDs must be unique")
    input_keys = {
        (sample.image_sha256, sample.group, sample.query, sample.reference)
        for sample in samples
    }
    if len(input_keys) != 1200:
        raise ValueError("EmbSpatial-1200 evaluator inputs must be unique")
    return tuple(samples)
