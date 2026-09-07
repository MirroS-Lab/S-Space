"""Validated manifest schema for contextual S-Space case studies."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from sspace.run_records import file_sha256

THEMES = ("beyond_visible", "language_supervision")


@dataclass(frozen=True)
class CaseImage:
    """One immutable image input bound by its SHA-256 digest."""

    path: Path
    sha256: str

    def validate(self) -> None:
        """Require one existing image whose bytes match the manifest."""
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if len(self.sha256) != 64 or file_sha256(self.path) != self.sha256:
            raise ValueError(f"Case image checksum changed: {self.path}")


@dataclass(frozen=True)
class CaseSample:
    """One image/prompt case with explicit named token character spans.

    ``object_spans`` uses half-open character offsets in ``prompt``. The
    manifest must name a ``target`` role and may additionally name a
    ``reference`` role. No substring search or token fallback occurs at run
    time.
    """

    case_id: str
    theme: str
    family: str
    condition: str
    images: tuple[CaseImage, ...]
    prompt: str
    object_spans: Mapping[str, tuple[int, int]]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        spans = {
            str(role): (int(value[0]), int(value[1]))
            for role, value in self.object_spans.items()
        }
        metadata = dict(self.metadata)
        object.__setattr__(self, "object_spans", MappingProxyType(spans))
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        self.validate()

    @property
    def object_order(self) -> tuple[str, ...]:
        """Return the fixed semantic role order used in saved state arrays."""
        return tuple(
            role for role in ("target", "reference") if role in self.object_spans
        )

    def validate(self) -> None:
        """Validate identity, theme, spans, images, and JSON metadata."""
        if not all(
            value.strip() for value in (self.case_id, self.family, self.condition)
        ):
            raise ValueError("Case identifiers must be non-empty")
        if self.theme not in THEMES:
            raise ValueError(f"Unknown case theme {self.theme!r}")
        if not self.images or not self.prompt.strip():
            raise ValueError("A case requires at least one image and one prompt")
        if "target" not in self.object_spans:
            raise ValueError("A case must declare a target token span")
        if set(self.object_spans) - {"target", "reference"}:
            raise ValueError("Only target and optional reference roles are supported")
        spans = []
        for role, (start, end) in self.object_spans.items():
            if not 0 <= start < end <= len(self.prompt):
                raise ValueError(f"Object span for {role!r} is outside the prompt")
            if not self.prompt[start:end].strip():
                raise ValueError(f"Object span for {role!r} is empty")
            spans.append((start, end))
        if len(set(spans)) != len(spans):
            raise ValueError("Target and reference spans must be distinct")
        json.dumps(dict(self.metadata), sort_keys=True)
        for image in self.images:
            image.validate()


def _resolve_image(record: Mapping[str, Any], image_root: Path) -> CaseImage:
    expected = {"path", "sha256"}
    if set(record) != expected:
        raise ValueError(f"Image record fields {set(record)} != {expected}")
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest image paths must stay below image_root")
    root = image_root.resolve()
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Resolved manifest image path must stay below image_root")
    return CaseImage(resolved, str(record["sha256"]))


def load_case_manifest(path: Path, image_root: Path) -> tuple[CaseSample, ...]:
    """Load ordered JSONL cases and fail closed on schema or checksum drift."""
    manifest = Path(path)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    required = {
        "case_id",
        "theme",
        "family",
        "condition",
        "images",
        "prompt",
        "object_spans",
        "metadata",
    }
    samples = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if set(record) != required:
            raise ValueError(
                f"Manifest line {line_number} fields {set(record)} != {required}"
            )
        images = tuple(_resolve_image(value, image_root) for value in record["images"])
        spans = {role: tuple(value) for role, value in record["object_spans"].items()}
        samples.append(
            CaseSample(
                case_id=str(record["case_id"]),
                theme=str(record["theme"]),
                family=str(record["family"]),
                condition=str(record["condition"]),
                images=images,
                prompt=str(record["prompt"]),
                object_spans=spans,
                metadata=dict(record["metadata"]),
            )
        )
    if not samples or len({sample.case_id for sample in samples}) != len(samples):
        raise ValueError("Case manifest must contain unique non-empty case IDs")
    return tuple(samples)
