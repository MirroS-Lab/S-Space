"""SpatialTunnel pairwise benchmark adapter."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd

from sspace.experiments.benchmarks.common import embedded_image
from sspace.experiments.common.pairwise.schema import PairwiseEvaluationSample


_GROUP = {
    "left": "horizontal",
    "right": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "far": "distance",
    "close": "distance",
}
_PATTERNS = {
    "horizontal": re.compile(
        r"^Is the (?P<query>.+?) to the left or right of the (?P<reference>.+?)\? "
        r"Answer with only one word\.$"
    ),
    "vertical": re.compile(
        r"^Is the (?P<query>.+?) above or below the (?P<reference>.+?)\? "
        r"Answer with only one word\.$"
    ),
    "distance": re.compile(
        r"^Compared to (?P<reference>.+?), is (?P<query>.+?) far or close from you\? "
        r"Answer with only one word\.$"
    ),
}


def load_spatialtunnel(path: Path) -> tuple[PairwiseEvaluationSample, ...]:
    """Load all 1,200 SpatialTunnel rows in source order (EVAL-03).

    Args:
        path: Pinned ``contrastive_probing.parquet``.

    Returns:
        Validated pairwise samples, with 200 rows per endpoint.

    Raises:
        ValueError: Schema, prompt grammar, label, image, count, or ID checks fail.

    Side effects:
        Reads one parquet file.
    """
    frame = pd.read_parquet(path)
    required = {"index", "image", "question", "answer"}
    if set(frame.columns) != required:
        raise ValueError(f"SpatialTunnel fields {set(frame.columns)} != {required}")
    samples = []
    for row in frame.to_dict("records"):
        endpoint = str(row["answer"]).lower()
        if endpoint not in _GROUP:
            raise ValueError(f"Unexpected SpatialTunnel endpoint {endpoint!r}")
        group = _GROUP[endpoint]
        question = str(row["question"])
        match = _PATTERNS[group].fullmatch(question)
        if match is None:
            raise ValueError(f"Unexpected SpatialTunnel prompt {question!r}")
        image_bytes, image_sha256 = embedded_image(row["image"])
        sample = PairwiseEvaluationSample(
            benchmark="spatialtunnel",
            sample_id=f"spatialtunnel_{int(row['index']):04d}",
            split="test",
            group=group,
            query=match.group("query"),
            reference=match.group("reference"),
            gold_endpoint=endpoint,
            direct_prompt=question,
            image_sha256=image_sha256,
            image_bytes=image_bytes,
            metadata={"source_index": int(row["index"])},
        )
        sample.validate()
        samples.append(sample)
    counts = Counter(sample.gold_endpoint for sample in samples)
    if counts != {endpoint: 200 for endpoint in _GROUP}:
        raise ValueError(f"SpatialTunnel endpoint counts changed: {dict(counts)}")
    if len({sample.sample_id for sample in samples}) != 1200:
        raise ValueError("SpatialTunnel must contain 1,200 unique sample IDs")
    return tuple(samples)
