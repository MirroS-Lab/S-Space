"""Define strict configurations for separate COCO training and validation sets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


COCO_TRAINING_PROTOCOL = "coco_train6000_mask_depth_v2"
COCO_VALIDATION_PROTOCOL = "coco_validation1800_vertical_dx020_v2"


def _resolve_relative(root: Path, value: object, field: str) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        raise ValueError(f"{field} must be relative to its declared root")
    resolved_root = Path(root).expanduser().resolve()
    resolved = (resolved_root / path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes its declared root")
    return resolved


def _validate_sources(
    annotation_path: Path,
    image_root: Path,
    depth_model_path: Path,
    output_dir: Path,
    depth_batch_size: int,
    *,
    require_output_absent: bool,
) -> None:
    if depth_batch_size <= 0:
        raise ValueError("depth_batch_size must be positive")
    for path in (annotation_path, image_root, depth_model_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if require_output_absent and output_dir.exists():
        raise FileExistsError(output_dir)


@dataclass(frozen=True)
class CocoTrainingConfig:
    """Define the frozen 6,000-row axis-construction dataset."""

    annotation_path: Path
    image_root: Path
    depth_model_path: Path
    output_dir: Path
    protocol: str
    samples_per_endpoint: int
    depth_candidate_count: int
    depth_confidence: float
    depth_batch_size: int
    seed: int
    device: str

    @property
    def sample_count(self) -> int:
        """Return the number of training relations."""
        return 6 * self.samples_per_endpoint

    def validate(self, *, require_output_absent: bool = True) -> None:
        """Reject any departure from the frozen COCO-6000 protocol."""
        if self.protocol != COCO_TRAINING_PROTOCOL:
            raise ValueError(f"Unknown COCO training protocol {self.protocol!r}")
        if self.samples_per_endpoint != 1000:
            raise ValueError("COCO-6000 requires exactly 1,000 rows per endpoint")
        if self.depth_candidate_count != 3200:
            raise ValueError("COCO-6000 requires exactly 3,200 depth candidates")
        if self.depth_confidence != 0.12:
            raise ValueError("COCO-6000 depth confidence must be exactly 0.12")
        if self.seed != 42:
            raise ValueError("COCO-6000 seed must be exactly 42")
        _validate_sources(
            self.annotation_path,
            self.image_root,
            self.depth_model_path,
            self.output_dir,
            self.depth_batch_size,
            require_output_absent=require_output_absent,
        )


@dataclass(frozen=True)
class CocoValidationConfig:
    """Define the frozen 1,800-row held-out layer-selection dataset."""

    training_dataset: Path
    annotation_path: Path
    image_root: Path
    depth_model_path: Path
    output_dir: Path
    protocol: str
    samples_per_endpoint: int
    depth_candidate_count: int
    depth_confidence: float
    depth_batch_size: int
    seed: int
    device: str

    @property
    def sample_count(self) -> int:
        """Return the number of validation relations."""
        return 6 * self.samples_per_endpoint

    def validate(self, *, require_output_absent: bool = True) -> None:
        """Reject any departure from the frozen COCO-1800 protocol."""
        if self.protocol != COCO_VALIDATION_PROTOCOL:
            raise ValueError(f"Unknown COCO validation protocol {self.protocol!r}")
        if self.samples_per_endpoint != 300:
            raise ValueError("COCO-1800 requires exactly 300 rows per endpoint")
        if self.depth_candidate_count != 1400:
            raise ValueError("COCO-1800 requires exactly 1,400 depth candidates")
        if self.depth_confidence != 0.12:
            raise ValueError("COCO-1800 depth confidence must be exactly 0.12")
        if self.seed != 42:
            raise ValueError("COCO-1800 seed must be exactly 42")
        if not self.training_dataset.exists():
            raise FileNotFoundError(self.training_dataset)
        _validate_sources(
            self.annotation_path,
            self.image_root,
            self.depth_model_path,
            self.output_dir,
            self.depth_batch_size,
            require_output_absent=require_output_absent,
        )


def _load_values(path: Path, expected: set[str], label: str) -> dict[str, object]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if set(values) != expected:
        raise ValueError(f"{label} fields {set(values)} != {expected}")
    return values


def load_coco_training_config(
    path: Path,
    project_root: Path,
    asset_root: Path,
    *,
    require_output_absent: bool = True,
) -> CocoTrainingConfig:
    """Load and validate the exact COCO-6000 JSON schema."""
    expected = {
        "annotation_path",
        "image_root",
        "depth_model_path",
        "output_dir",
        "protocol",
        "samples_per_endpoint",
        "depth_candidate_count",
        "depth_confidence",
        "depth_batch_size",
        "seed",
        "device",
    }
    values = _load_values(path, expected, "COCO training config")
    config = CocoTrainingConfig(
        annotation_path=_resolve_relative(
            asset_root, values["annotation_path"], "annotation_path"
        ),
        image_root=_resolve_relative(asset_root, values["image_root"], "image_root"),
        depth_model_path=_resolve_relative(
            asset_root, values["depth_model_path"], "depth_model_path"
        ),
        output_dir=_resolve_relative(project_root, values["output_dir"], "output_dir"),
        protocol=str(values["protocol"]),
        samples_per_endpoint=int(values["samples_per_endpoint"]),
        depth_candidate_count=int(values["depth_candidate_count"]),
        depth_confidence=float(values["depth_confidence"]),
        depth_batch_size=int(values["depth_batch_size"]),
        seed=int(values["seed"]),
        device=str(values["device"]),
    )
    config.validate(require_output_absent=require_output_absent)
    return config


def load_coco_validation_config(
    path: Path,
    project_root: Path,
    asset_root: Path,
    *,
    require_output_absent: bool = True,
) -> CocoValidationConfig:
    """Load and validate the exact COCO-1800 JSON schema."""
    expected = {
        "training_dataset",
        "annotation_path",
        "image_root",
        "depth_model_path",
        "output_dir",
        "protocol",
        "samples_per_endpoint",
        "depth_candidate_count",
        "depth_confidence",
        "depth_batch_size",
        "seed",
        "device",
    }
    values = _load_values(path, expected, "COCO validation config")
    config = CocoValidationConfig(
        training_dataset=_resolve_relative(
            project_root, values["training_dataset"], "training_dataset"
        ),
        annotation_path=_resolve_relative(
            asset_root, values["annotation_path"], "annotation_path"
        ),
        image_root=_resolve_relative(asset_root, values["image_root"], "image_root"),
        depth_model_path=_resolve_relative(
            asset_root, values["depth_model_path"], "depth_model_path"
        ),
        output_dir=_resolve_relative(project_root, values["output_dir"], "output_dir"),
        protocol=str(values["protocol"]),
        samples_per_endpoint=int(values["samples_per_endpoint"]),
        depth_candidate_count=int(values["depth_candidate_count"]),
        depth_confidence=float(values["depth_confidence"]),
        depth_batch_size=int(values["depth_batch_size"]),
        seed=int(values["seed"]),
        device=str(values["device"]),
    )
    config.validate(require_output_absent=require_output_absent)
    return config
