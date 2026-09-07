"""Define strict SpinBench projection configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.core.models.runtime import RUNTIME_SPECS
from sspace.core.models.qwen import QWEN_MODEL_SPECS

from .adapter import SPINBENCH_TEST_SHA256
from .projection import (
    PROBE_CONTEXTS,
    PROJECTION_PROTOCOL,
    PROMPT_STYLE,
)


ARTIFACT_SELECTED_LAYER = "coco1800_margin_selected_layer_v1"
READOUT_SELECTION_PROTOCOLS = {ARTIFACT_SELECTED_LAYER}


@dataclass(frozen=True)
class SpinBenchProjectionConfig:
    """Define one complete model-by-premise EVAL-09 run."""

    schema_version: str
    evaluation_id: str
    artifact_dir: Path
    model_adapter: str
    model_path: Path
    annotation_path: Path
    image_root: Path
    premise_mode: str
    annotation_sha256: str
    expected_sample_count: int
    expected_dataset_fingerprint: str
    selected_layer: int
    readout_selection_protocol: str
    readout_selection_manifest: Path | None
    projection_protocol: str
    prompt_style: str
    probe_context: str
    image_max_pixels: int | None
    output_dir: Path
    device: str
    checkpoint_every: int
    seed: int

    def validate(self) -> None:
        """Reject fields outside the frozen rotation-matrix protocol."""
        if self.schema_version != "1.1.0" or not self.evaluation_id.strip():
            raise ValueError("SpinBench projection schema or ID is invalid")
        if self.model_adapter not in RUNTIME_SPECS:
            raise ValueError(f"Unknown model adapter {self.model_adapter!r}")
        if self.premise_mode not in {"without", "with"}:
            raise ValueError("SpinBench premise_mode must be 'without' or 'with'")
        expected_count = 146 if self.premise_mode == "without" else 144
        if self.expected_sample_count != expected_count:
            raise ValueError(
                f"SpinBench {self.premise_mode} requires {expected_count} rows"
            )
        expected_context = (
            "none" if self.premise_mode == "without" else "official_source_premise"
        )
        if (
            self.probe_context not in PROBE_CONTEXTS
            or self.probe_context != expected_context
        ):
            raise ValueError("SpinBench premise mode and probe context differ")
        if self.annotation_sha256 != SPINBENCH_TEST_SHA256:
            raise ValueError("SpinBench annotation checksum is not pinned")
        fingerprint = self.expected_dataset_fingerprint
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("SpinBench fingerprint must be lowercase SHA-256")
        if self.selected_layer < 0:
            raise ValueError("SpinBench selected layer must be nonnegative")
        if self.readout_selection_protocol not in READOUT_SELECTION_PROTOCOLS:
            raise ValueError("SpinBench readout selection protocol is not frozen")
        if self.readout_selection_manifest is not None:
            raise ValueError("Frozen readout must not declare a second manifest")
        if self.projection_protocol != PROJECTION_PROTOCOL:
            raise ValueError("SpinBench projection protocol is not frozen")
        if self.prompt_style != PROMPT_STYLE:
            raise ValueError("SpinBench projection requires the baseline prompt")
        if self.image_max_pixels is not None:
            if self.model_adapter not in QWEN_MODEL_SPECS:
                raise ValueError("image_max_pixels is supported only by Qwen")
            if isinstance(self.image_max_pixels, bool) or not isinstance(
                self.image_max_pixels, int
            ):
                raise ValueError("image_max_pixels must be a JSON integer or null")
            qwen_spec = QWEN_MODEL_SPECS[self.model_adapter]
            if (
                not qwen_spec.image_min_pixels
                <= self.image_max_pixels
                <= qwen_spec.image_max_pixels
            ):
                raise ValueError("SpinBench Qwen image_max_pixels is outside bounds")
        if self.checkpoint_every <= 0 or self.seed < 0:
            raise ValueError("SpinBench checkpoint interval or seed is invalid")
        for path in (
            self.artifact_dir,
            self.model_path,
            self.annotation_path,
            self.image_root,
        ):
            if not path.exists():
                raise FileNotFoundError(path)
        if (
            self.readout_selection_manifest is not None
            and not self.readout_selection_manifest.is_file()
        ):
            raise FileNotFoundError(self.readout_selection_manifest)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


def load_spinbench_projection_config(
    path: Path,
    project_root: Path,
) -> SpinBenchProjectionConfig:
    """Load exact EVAL-09 fields and resolve project-relative paths."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = set(SpinBenchProjectionConfig.__dataclass_fields__) | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"SpinBench projection fields {set(values)} != {expected}")
    validate_launch_metadata(values, "spinbench_projection")

    def resolve_optional(name: str) -> Path | None:
        value = values[name]
        if value is None:
            return None
        return resolve_project_path(value, project_root, name)

    image_max_pixels = values["image_max_pixels"]
    config = SpinBenchProjectionConfig(
        schema_version=str(values["schema_version"]),
        evaluation_id=str(values["evaluation_id"]),
        artifact_dir=resolve_project_path(
            values["artifact_dir"], project_root, "artifact_dir"
        ),
        model_adapter=str(values["model_adapter"]),
        model_path=resolve_project_path(
            values["model_path"], project_root, "model_path"
        ),
        annotation_path=resolve_project_path(
            values["annotation_path"], project_root, "annotation_path"
        ),
        image_root=resolve_project_path(
            values["image_root"], project_root, "image_root"
        ),
        premise_mode=str(values["premise_mode"]),
        annotation_sha256=str(values["annotation_sha256"]),
        expected_sample_count=int(values["expected_sample_count"]),
        expected_dataset_fingerprint=str(values["expected_dataset_fingerprint"]),
        selected_layer=int(values["selected_layer"]),
        readout_selection_protocol=str(values["readout_selection_protocol"]),
        readout_selection_manifest=resolve_optional("readout_selection_manifest"),
        projection_protocol=str(values["projection_protocol"]),
        prompt_style=str(values["prompt_style"]),
        probe_context=str(values["probe_context"]),
        image_max_pixels=image_max_pixels,
        output_dir=resolve_project_path(
            values["output_dir"], project_root, "output_dir"
        ),
        device=str(values["device"]),
        checkpoint_every=int(values["checkpoint_every"]),
        seed=int(values["seed"]),
    )
    config.validate()
    return config
