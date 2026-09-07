"""Define records shared by pairwise adapters and evaluators."""

from __future__ import annotations

import hashlib
import io
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from sspace.run_records import canonical_fingerprint


ENDPOINTS = {
    "horizontal": ("left", "right"),
    "vertical": ("above", "below"),
    "distance": ("far", "close"),
}


def open_pairwise_image(sample: PairwiseEvaluationSample) -> Image.Image:
    """Open one pairwise image only after verifying its declared SHA-256."""
    sample.validate()
    if sample.image_bytes is not None:
        encoded = sample.image_bytes
    else:
        assert sample.image_path is not None
        encoded = sample.image_path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != sample.image_sha256:
        raise ValueError(f"Pairwise image checksum changed: {sample.sample_id}")
    return Image.open(io.BytesIO(encoded)).convert("RGB")


def pairwise_dataset_fingerprint(
    samples: tuple[PairwiseEvaluationSample, ...],
) -> str:
    """Bind ordered pairwise semantics and declared image bytes."""
    if not samples:
        raise ValueError("Cannot fingerprint an empty pairwise dataset")
    for sample in samples:
        sample.validate()
    return canonical_fingerprint(
        {
            "protocol": "pairwise_evaluation_samples_v1",
            "samples": [
                {
                    "benchmark": sample.benchmark,
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "group": sample.group,
                    "query": sample.query,
                    "reference": sample.reference,
                    "gold_endpoint": sample.gold_endpoint,
                    "direct_prompt": sample.direct_prompt,
                    "image_sha256": sample.image_sha256,
                    "metadata": sample.metadata,
                }
                for sample in samples
            ],
        }
    )


@dataclass(frozen=True)
class PairwiseEvaluationSample:
    """One benchmark row answerable by a directed H/V/D pairwise score.

    The adapter declares entity binding and ground truth. It does not compute
    hidden states or inspect an artifact. Exactly one of ``image_path`` and
    ``image_bytes`` must be present.
    """

    benchmark: str
    sample_id: str
    split: str
    group: str
    query: str
    reference: str
    gold_endpoint: str
    direct_prompt: str
    image_sha256: str
    image_path: Path | None = None
    image_bytes: bytes | None = field(default=None, repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject ambiguous images, invalid endpoints, or missing identifiers."""
        if not self.benchmark or not self.sample_id or not self.split:
            raise ValueError("Benchmark, sample ID, and split must be non-empty")
        if self.group not in ENDPOINTS:
            raise ValueError(f"Unsupported pairwise group {self.group!r}")
        if self.gold_endpoint not in ENDPOINTS[self.group]:
            raise ValueError(
                f"Endpoint {self.gold_endpoint!r} is invalid for {self.group}"
            )
        if not self.query.strip() or not self.reference.strip():
            raise ValueError("Pairwise object names must be non-empty")
        if self.query == self.reference:
            raise ValueError("Pairwise query and reference must differ")
        if (self.image_path is None) == (self.image_bytes is None):
            raise ValueError("Exactly one image source must be declared")
        if self.image_path is not None and not self.image_path.is_file():
            raise FileNotFoundError(self.image_path)
        if len(self.image_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.image_sha256
        ):
            raise ValueError("Image SHA-256 must be lowercase hexadecimal")


@dataclass(frozen=True)
class PairwiseScoreRecord:
    """One fully traceable projection prediction for a benchmark row.

    ``predicted_endpoint`` is one declared endpoint or ``"tie"`` when the
    directed margin is exactly zero. A tie must have ``correct=False``.
    """

    benchmark: str
    sample_id: str
    split: str
    layer_id: int
    group: str
    query: str
    reference: str
    original_margin: float
    swapped_margin: float
    margin: float
    predicted_endpoint: str
    gold_endpoint: str
    correct: bool
    decision_protocol: str
    direct_prompt: str
    projection_prompt_original: str
    projection_prompt_swapped: str
    image_sha256: str
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record without changing field values."""
        return asdict(self)
