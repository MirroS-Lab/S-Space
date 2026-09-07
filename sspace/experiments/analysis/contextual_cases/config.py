"""Configuration parsing for contextual case-study projection runs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata


def _path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@dataclass(frozen=True)
class CaseEvaluationConfig:
    """Fully resolved selected-layer case-study configuration."""

    evaluation_id: str
    model_adapter: str
    model_path: Path
    artifact_dir: Path
    manifest_path: Path
    image_root: Path
    output_dir: Path
    selected_layer: int
    device: str
    seed: int
    checkpoint_every: int
    image_max_pixels: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible configuration record."""
        value = asdict(self)
        for key in (
            "model_path",
            "artifact_dir",
            "manifest_path",
            "image_root",
            "output_dir",
        ):
            value[key] = str(value[key])
        return value


def load_case_config(path: Path, project_root: Path) -> CaseEvaluationConfig:
    """Load exact config fields without inferred paths or layer defaults."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "evaluation_id",
        "model_adapter",
        "model_path",
        "artifact_dir",
        "manifest_path",
        "image_root",
        "output_dir",
        "selected_layer",
        "device",
        "seed",
        "checkpoint_every",
        "image_max_pixels",
    } | LAUNCH_FIELDS
    if set(raw) != required:
        raise ValueError(f"Case config fields {set(raw)} != {required}")
    validate_launch_metadata(raw, "contextual_case")
    config = CaseEvaluationConfig(
        evaluation_id=str(raw["evaluation_id"]),
        model_adapter=str(raw["model_adapter"]),
        model_path=_path(str(raw["model_path"]), project_root),
        artifact_dir=_path(str(raw["artifact_dir"]), project_root),
        manifest_path=_path(str(raw["manifest_path"]), project_root),
        image_root=_path(str(raw["image_root"]), project_root),
        output_dir=_path(str(raw["output_dir"]), project_root),
        selected_layer=int(raw["selected_layer"]),
        device=str(raw["device"]),
        seed=int(raw["seed"]),
        checkpoint_every=int(raw["checkpoint_every"]),
        image_max_pixels=(
            None if raw["image_max_pixels"] is None else int(raw["image_max_pixels"])
        ),
    )
    if not config.evaluation_id.strip() or config.selected_layer < 0:
        raise ValueError("Evaluation ID and selected layer must be explicit")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    return config
