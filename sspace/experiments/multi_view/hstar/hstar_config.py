"""Define strict frozen H* HOS-600 all-layer evaluation configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.core.models.qwen import QWEN_MODEL_SPECS
from sspace.core.models.runtime import RUNTIME_SPECS


@dataclass(frozen=True)
class HStarMultiViewConfig:
    """Define one checkpointed EVAL-13 HOS-600 evaluation."""

    schema_version: str
    evaluation_id: str
    artifact_dir: Path
    model_adapter: str
    model_path: Path
    dataset_dir: Path
    expected_dataset_fingerprint: str
    expected_sample_count: int
    output_dir: Path
    prompt_protocol: str
    projection_protocol: str
    image_max_pixels: int | None
    device: str
    checkpoint_every: int
    seed: int

    def validate(self) -> None:
        """Reject fields outside the frozen EVAL-13 protocol."""
        if self.schema_version != "1.0.0" or not self.evaluation_id.strip():
            raise ValueError("H* multi-view schema or evaluation ID is invalid")
        if self.model_adapter not in RUNTIME_SPECS:
            raise ValueError(f"Unknown model adapter {self.model_adapter!r}")
        if self.expected_sample_count != 600:
            raise ValueError("H* evaluation requires exactly 600 samples")
        digest = self.expected_dataset_fingerprint
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("expected_dataset_fingerprint must be lowercase SHA-256")
        if self.prompt_protocol != "hstar_hos600_view1_prefix_order_swap_v1":
            raise ValueError("Unknown H* prompt protocol")
        if self.projection_protocol != "layerwise_single_object_delta_v1":
            raise ValueError("Unknown H* projection protocol")
        if self.seed < 0 or self.checkpoint_every <= 0:
            raise ValueError("H* deterministic controls are invalid")
        if self.image_max_pixels is not None:
            if self.model_adapter not in QWEN_MODEL_SPECS:
                raise ValueError("image_max_pixels is supported only by Qwen")
            spec = QWEN_MODEL_SPECS[self.model_adapter]
            if (
                not spec.image_min_pixels
                <= self.image_max_pixels
                <= spec.image_max_pixels
            ):
                raise ValueError("Qwen image_max_pixels is outside pinned bounds")
        for path in (self.artifact_dir, self.model_path, self.dataset_dir):
            if not path.exists():
                raise FileNotFoundError(path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


def load_hstar_multiview_config(path: Path, project_root: Path) -> HStarMultiViewConfig:
    """Load exact EVAL-13 fields and resolve project-relative paths."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = set(HStarMultiViewConfig.__dataclass_fields__) | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"H* config fields {set(values)} != {expected}")
    validate_launch_metadata(values, "hstar_multiview")

    image_max_pixels = values["image_max_pixels"]
    if image_max_pixels is not None and (
        isinstance(image_max_pixels, bool) or not isinstance(image_max_pixels, int)
    ):
        raise ValueError("image_max_pixels must be a JSON integer or null")
    config = HStarMultiViewConfig(
        schema_version=str(values["schema_version"]),
        evaluation_id=str(values["evaluation_id"]),
        artifact_dir=resolve_project_path(
            values["artifact_dir"], project_root, "artifact_dir"
        ),
        model_adapter=str(values["model_adapter"]),
        model_path=resolve_project_path(
            values["model_path"], project_root, "model_path"
        ),
        dataset_dir=resolve_project_path(
            values["dataset_dir"], project_root, "dataset_dir"
        ),
        expected_dataset_fingerprint=str(values["expected_dataset_fingerprint"]),
        expected_sample_count=int(values["expected_sample_count"]),
        output_dir=resolve_project_path(
            values["output_dir"], project_root, "output_dir"
        ),
        prompt_protocol=str(values["prompt_protocol"]),
        projection_protocol=str(values["projection_protocol"]),
        image_max_pixels=image_max_pixels,
        device=str(values["device"]),
        checkpoint_every=int(values["checkpoint_every"]),
        seed=int(values["seed"]),
    )
    config.validate()
    return config
