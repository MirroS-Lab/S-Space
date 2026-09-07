"""CV-Bench two-object Relation and Depth adapter."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from sspace.experiments.benchmarks.common import embedded_image
from sspace.experiments.common.pairwise.schema import PairwiseEvaluationSample


_RELATION = re.compile(
    r"where (?:is|are) (?:the )?(?P<query>.+?) located with respect to "
    r"(?:the )?(?P<reference>.+?)\?$",
    re.IGNORECASE,
)
_DEPTH = re.compile(
    r"Which object is closer to the camera taking this photo, the "
    r"(?P<query>.+?) \(highlighted by a red box\) or the "
    r"(?P<reference>.+?) \(highlighted by a blue box\)\?$",
    re.IGNORECASE,
)
_ANNOTATED = re.compile(
    r"^(?P<name>.+?) \(annotated by the (?P<color>red|blue|green) box\)$",
    re.IGNORECASE,
)


def _answer_index(answer: str, choice_count: int) -> int:
    match = re.fullmatch(r"\(([A-Z])\)", answer)
    if match is None:
        raise ValueError(f"Invalid CV-Bench answer {answer!r}")
    index = ord(match.group(1)) - ord("A")
    if not 0 <= index < choice_count:
        raise ValueError("CV-Bench answer is outside the choice list")
    return index


def _marked_object(value: str) -> str:
    match = _ANNOTATED.fullmatch(value.strip())
    return (
        value.strip()
        if match is None
        else f"{match.group('color').lower()}-box {match.group('name')}"
    )


def _metadata(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_index": int(row["idx"]),
        "source": str(row["source"]),
        "source_dataset": str(row["source_dataset"]),
        "filename": str(row["filename"]),
    }


def _relation_sample(row: Mapping[str, Any]) -> PairwiseEvaluationSample:
    question = str(row["question"])
    match = _RELATION.search(question)
    if match is None:
        raise ValueError(f"Could not parse CV-Bench relation prompt {question!r}")
    choices = [str(value).lower() for value in row["choices"]]
    endpoint = choices[_answer_index(str(row["answer"]), len(choices))]
    groups = {
        "left": "horizontal",
        "right": "horizontal",
        "above": "vertical",
        "below": "vertical",
    }
    if endpoint not in groups:
        raise ValueError(f"Unsupported CV-Bench Relation endpoint {endpoint!r}")
    image_bytes, digest = embedded_image(row["image"])
    return PairwiseEvaluationSample(
        benchmark="cvbench_pairwise",
        sample_id=f"cvbench_2d_{int(row['idx'])}",
        split="test",
        group=groups[endpoint],
        query=_marked_object(match.group("query")),
        reference=_marked_object(match.group("reference")),
        gold_endpoint=endpoint,
        direct_prompt=question,
        image_sha256=digest,
        image_bytes=image_bytes,
        metadata=_metadata(row),
    )


def _depth_sample(row: Mapping[str, Any]) -> PairwiseEvaluationSample:
    question = str(row["question"])
    match = _DEPTH.fullmatch(question)
    if match is None:
        raise ValueError(f"Could not parse CV-Bench depth prompt {question!r}")
    choices = [str(value) for value in row["choices"]]
    parsed = [match.group("query"), match.group("reference")]
    if choices != parsed:
        raise ValueError("CV-Bench Depth prompt and choices have different objects")
    correct = _answer_index(str(row["answer"]), 2)
    image_bytes, digest = embedded_image(row["image"])
    return PairwiseEvaluationSample(
        benchmark="cvbench_pairwise",
        sample_id=f"cvbench_3d_{int(row['idx'])}",
        split="test",
        group="distance",
        query=f"red-box {choices[0]}",
        reference=f"blue-box {choices[1]}",
        gold_endpoint="close" if correct == 0 else "far",
        direct_prompt=question,
        image_sha256=digest,
        image_bytes=image_bytes,
        metadata=_metadata(row),
    )


def load_cvbench_pairwise(snapshot: Path) -> tuple[PairwiseEvaluationSample, ...]:
    """Load all 1,250 compatible two-object CV-Bench rows (EVAL-03).

    The adapter includes 2D ``Relation`` and 3D ``Depth``. It deliberately
    excludes 788 2D ``Count`` rows and 600 three-object ``Distance`` rows;
    those tasks do not reduce to one pairwise endpoint classification.
    """
    two_d = pd.read_parquet(snapshot / "test_2d.parquet")
    three_d = pd.read_parquet(snapshot / "test_3d.parquet")
    samples = [
        _relation_sample(row)
        for row in two_d[two_d["task"] == "Relation"].to_dict("records")
    ]
    samples.extend(
        _depth_sample(row)
        for row in three_d[three_d["task"] == "Depth"].to_dict("records")
    )
    if len(samples) != 1250 or len({sample.sample_id for sample in samples}) != 1250:
        raise ValueError(
            f"Expected 1,250 unique CV-Bench pairwise rows, found {len(samples)}"
        )
    for sample in samples:
        sample.validate()
    return tuple(samples)
