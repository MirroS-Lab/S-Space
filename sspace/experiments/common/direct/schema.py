"""Define samples and records for direct generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..pairwise.schema import ENDPOINTS


@dataclass(frozen=True)
class DirectGenerationSample:
    """One image/question row scored as a binary H/V/D endpoint."""

    benchmark: str
    sample_id: str
    split: str
    group: str
    query: str
    reference: str
    gold_endpoint: str
    prompt: str
    image_path: Path | None = None
    image_bytes: bytes | None = field(default=None, repr=False)
    image_sha256: str | None = None
    subset_tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Reject ambiguous images, invalid endpoints, or missing identities."""
        if not self.benchmark or not self.sample_id or not self.split:
            raise ValueError("Benchmark, sample ID, and split must be non-empty")
        if self.group not in ENDPOINTS:
            raise ValueError(f"Unsupported direct-generation group {self.group!r}")
        if self.gold_endpoint not in ENDPOINTS[self.group]:
            raise ValueError("Direct-generation endpoint and group differ")
        if (
            not self.query.strip()
            or not self.reference.strip()
            or not self.prompt.strip()
        ):
            raise ValueError("Objects and direct-generation prompt must be non-empty")
        if self.query == self.reference:
            raise ValueError("Direct-generation query and reference must differ")
        if (self.image_path is None) == (self.image_bytes is None):
            raise ValueError("Exactly one direct-generation image source is required")
        if self.image_path is not None and not self.image_path.is_file():
            raise FileNotFoundError(self.image_path)
        if self.image_sha256 is not None:
            digest = self.image_sha256
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("image_sha256 must be lowercase SHA-256")
        if len(set(self.subset_tags)) != len(self.subset_tags):
            raise ValueError("Direct-generation subset tags must be unique")


@dataclass(frozen=True)
class DirectGenerationRecord:
    """One complete Qwen generation, strict parse, and correctness decision."""

    evaluation_id: str
    benchmark: str
    sample_id: str
    global_sample_index: int
    shard_rank: int
    shard_count: int
    split: str
    subset_tags: tuple[str, ...]
    group: str
    query: str
    reference: str
    gold_endpoint: str
    prompt: str
    image_sha256: str
    thinking: bool
    output_tokens: int
    ended_with_eos: bool
    thinking_completed: bool | None
    reasoning_response: str
    final_response: str
    raw_response: str
    generated_token_sha256: str
    predicted_endpoint: str | None
    parse_failure: bool
    correct: bool
    generation_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without changing generated text."""
        return asdict(self)
