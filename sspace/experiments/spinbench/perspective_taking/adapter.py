"""SpinBench cardinal perspective-taking adapter and coordinate rotation."""

from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


TASK_PREFIX = "infinigen_spatial_relation_transformation_"
SPINBENCH_TEST_SHA256 = (
    "0315f4e5723a33ce24138055051c6b5ceb90ded66b37e850e258ded0553059d2"
)
VIEW_ROTATIONS = {
    "front": np.asarray(((1, 0, 0), (0, 1, 0), (0, 0, 1)), dtype=np.float64),
    "left": np.asarray(((0, 0, 1), (0, 1, 0), (-1, 0, 0)), dtype=np.float64),
    "right": np.asarray(((0, 0, -1), (0, 1, 0), (1, 0, 0)), dtype=np.float64),
    "back": np.asarray(((-1, 0, 0), (0, 1, 0), (0, 0, -1)), dtype=np.float64),
}
_TRANSFORM_PATTERN = re.compile(
    r"^perspective_(?P<source>fn|lr)_(?P<view>back|left|right)$"
)
_LEFT_QUESTION_PATTERN = re.compile(
    r"which object (?:now )?appears on the left", re.IGNORECASE
)
_CLOSER_QUESTION_PATTERN = re.compile(
    r"which object is (?:now )?closer to the viewer", re.IGNORECASE
)


@dataclass(frozen=True)
class SpinBenchSample:
    """One strict perspective-taking row with explicit A/B object binding."""

    sample_id: str
    row_index: int
    premise_mode: str
    transform: str
    source_group: str
    target_view: str
    target_property: str
    object_a: str
    object_b: str
    answer: str
    image_path: Path
    direct_prompt: str
    source_premise: str | None
    metadata: dict[str, Any]

    @property
    def group(self) -> str:
        """Return the measured source relation group."""
        return self.source_group


def open_spinbench_image(sample: SpinBenchSample) -> Image.Image:
    """Decode a SpinBench image eagerly from its declared file bytes."""
    encoded = sample.image_path.read_bytes()
    return Image.open(io.BytesIO(encoded)).convert("RGB")


def _parse_transform(transform: str) -> tuple[str, str]:
    """Decode source relation and target view from official metadata.

    This parser does not map transforms to answers. It only decodes the
    benchmark's ``fn``/``lr`` source-relation abbreviation and cardinal view.
    """
    match = _TRANSFORM_PATTERN.fullmatch(transform)
    if match is None:
        raise ValueError(f"Invalid SpinBench transform {transform!r}")
    source_group = {
        "fn": "distance",
        "lr": "horizontal",
    }[match.group("source")]
    return source_group, match.group("view")


def _parse_target_property(prompt: str) -> str:
    """Read the requested coordinate component from the official question."""
    left = _LEFT_QUESTION_PATTERN.search(prompt) is not None
    closer = _CLOSER_QUESTION_PATTERN.search(prompt) is not None
    if left == closer:
        raise ValueError("SpinBench prompt must ask exactly one target property")
    return "left" if left else "closer"


def _object_name(value: str) -> str:
    return " ".join(value.replace("_", " ").split())


def _prompt_without_image(problem: str) -> str:
    if problem.count("<image>") != 1:
        raise ValueError("SpinBench perspective rows need one image placeholder")
    return problem.replace("<image>", "", 1).strip()


def _source_premise(prompt: str) -> str:
    premise, separator, _ = prompt.partition(", then when")
    if not separator or not premise.startswith("As "):
        raise ValueError("Could not isolate the SpinBench source premise")
    return premise.rstrip(". ") + "."


def load_spinbench(
    annotation_path: Path,
    image_root: Path,
    premise_mode: str,
) -> tuple[SpinBenchSample, ...]:
    """Load the complete 146/144-row SpinBench perspective subset.

    Args:
        annotation_path: Official full ``test.jsonl``.
        image_root: Extracted SpinBench root containing the declared paths.
        premise_mode: Exact ``without`` or ``with`` subset.

    Returns:
        Ordered strict samples. ``without`` contains 146 rows and ``with`` 144.

    Raises:
        ValueError: Metadata, transform, prompt, answer, image count, or subset
            count differs from the frozen benchmark.
        FileNotFoundError: A declared image is absent.

    Side effects:
        Reads the annotation file and checks image paths.
    """
    if premise_mode not in {"without", "with"}:
        raise ValueError("SpinBench premise_mode must be 'without' or 'with'")
    samples = []
    with annotation_path.open(encoding="utf-8") as stream:
        for row_index, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("metadata")
            if not isinstance(metadata, Mapping):
                raise ValueError(f"SpinBench row {row_index} lacks metadata")
            task_type = str(metadata.get("task_type", ""))
            if not task_type.startswith(TASK_PREFIX):
                continue
            row_mode = (
                "without"
                if "_wo_premise_" in task_type
                else "with"
                if "_w_premise_" in task_type
                else None
            )
            if row_mode is None:
                raise ValueError(f"Unknown SpinBench premise mode in {task_type!r}")
            if row_mode != premise_mode:
                continue
            raw_transform = str(metadata.get("augmented_from", ""))
            implicit = raw_transform.endswith("_implicit")
            transform = raw_transform.removesuffix("_implicit")
            if implicit != (row_mode == "without"):
                raise ValueError(f"Invalid SpinBench transform {raw_transform!r}")
            images = row.get("images")
            if not isinstance(images, list) or len(images) != 1:
                raise ValueError(f"SpinBench row {row_index} must contain one image")
            image_path = image_root / str(images[0])
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            answer = str(row.get("answer", "")).upper()
            if answer not in {"A", "B"}:
                raise ValueError(f"Invalid SpinBench answer {answer!r}")
            prompt = _prompt_without_image(str(row.get("problem", "")))
            source_group, target_view = _parse_transform(transform)
            target_property = _parse_target_property(prompt)
            samples.append(
                SpinBenchSample(
                    sample_id=f"spinbench_{row_index:04d}_{metadata.get('index', row_index)}",
                    row_index=row_index,
                    premise_mode=row_mode,
                    transform=transform,
                    source_group=source_group,
                    target_view=target_view,
                    target_property=target_property,
                    object_a=_object_name(str(metadata["objectA"])),
                    object_b=_object_name(str(metadata["objectB"])),
                    answer=answer,
                    image_path=image_path,
                    direct_prompt=prompt,
                    source_premise=_source_premise(prompt)
                    if row_mode == "with"
                    else None,
                    metadata={
                        "task_type": task_type,
                        "source_index": str(metadata.get("index", "")),
                        "center_a": list(metadata["centerA"]),
                        "center_b": list(metadata["centerB"]),
                    },
                )
            )
    expected = 146 if premise_mode == "without" else 144
    if (
        len(samples) != expected
        or len({sample.sample_id for sample in samples}) != expected
    ):
        raise ValueError(
            f"Expected {expected} unique SpinBench {premise_mode} rows, found {len(samples)}"
        )
    return tuple(samples)


def spinbench_fingerprint(samples: Sequence[SpinBenchSample]) -> str:
    """Bind ordered EVAL-06 samples, prompts, metadata, and image bytes.

    Args:
        samples: One complete, single-premise-mode SpinBench subset.

    Returns:
        Lowercase SHA-256 of canonical JSON evidence.

    Raises:
        ValueError: The input is empty, mixes premise modes, or repeats IDs.
        FileNotFoundError: A sample image is absent.

    Side effects:
        Reads each declared image with bounded memory.
    """
    if not samples:
        raise ValueError("Cannot fingerprint an empty SpinBench subset")
    modes = {sample.premise_mode for sample in samples}
    if len(modes) != 1 or len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("SpinBench fingerprint requires one mode and unique IDs")

    def image_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    rows = [
        {
            "sample_id": sample.sample_id,
            "row_index": sample.row_index,
            "premise_mode": sample.premise_mode,
            "transform": sample.transform,
            "source_group": sample.source_group,
            "target_view": sample.target_view,
            "target_property": sample.target_property,
            "object_a": sample.object_a,
            "object_b": sample.object_b,
            "answer": sample.answer,
            "direct_prompt": sample.direct_prompt,
            "source_premise": sample.source_premise,
            "metadata": sample.metadata,
            "image_sha256": image_sha256(sample.image_path),
        }
        for sample in samples
    ]
    encoded = json.dumps(
        {"schema_version": "1.0.0", "samples": rows},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def rotate_coordinates(
    relative_coordinates: Sequence[float], target_view: str
) -> np.ndarray:
    """Rotate an A-minus-B right/below/close vector into a cardinal view.

    Args:
        relative_coordinates: Finite H/V/D vector ``[3]`` in source camera coordinates.
        target_view: ``front``, ``left``, ``right``, or ``back``.

    Returns:
        Rotated finite H/V/D vector ``[3]``.

    Raises:
        ValueError: Shape, finite values, or target view is invalid.

    Side effects:
        None.
    """
    coordinates = np.asarray(relative_coordinates, dtype=np.float64)
    if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
        raise ValueError("SpinBench coordinates must be finite H/V/D [3]")
    if target_view not in VIEW_ROTATIONS:
        raise ValueError(f"Unknown SpinBench target view {target_view!r}")
    rotation = VIEW_ROTATIONS[target_view]
    return rotation @ coordinates


def spinbench_target_margin(
    rotated_coordinates: Sequence[float], target_property: str
) -> float:
    """Read the requested A-over-B decision margin from target coordinates.

    Args:
        rotated_coordinates: Finite A-minus-B H/V/D coordinates ``[3]`` in the
            target camera frame.
        target_property: Official question property, ``left`` or ``closer``.

    Returns:
        Positive when object A is the requested answer and negative for B.

    Raises:
        ValueError: Coordinates, property, or the resulting margin is invalid.

    Side effects:
        None.
    """
    coordinates = np.asarray(rotated_coordinates, dtype=np.float64)
    if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
        raise ValueError("SpinBench target coordinates must be finite H/V/D [3]")
    if target_property == "left":
        margin = -float(coordinates[0])
    elif target_property == "closer":
        margin = float(coordinates[2])
    else:
        raise ValueError(f"Unknown SpinBench target property {target_property!r}")
    if margin == 0.0:
        raise ValueError("SpinBench target margin is exactly zero")
    return margin


def map_spinbench_answer(
    relative_coordinates: Sequence[float], sample: SpinBenchSample
) -> str:
    """Map source coordinates to A/B after the declared camera rotation.

    The input is object A minus object B. After rotation, a negative horizontal
    component means A is left of B; a positive distance component means A is
    closer than B. A zero target component is undefined and fails explicitly.
    """
    rotated = rotate_coordinates(relative_coordinates, sample.target_view)
    margin = spinbench_target_margin(rotated, sample.target_property)
    return "A" if margin > 0 else "B"
