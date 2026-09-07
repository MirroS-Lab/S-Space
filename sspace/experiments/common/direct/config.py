"""Define strict sharded direct-generation configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.core.models.molmo import MOLMO_MODEL_SPECS
from sspace.core.models.qwen import QWEN_MODEL_SPECS


DIRECT_DATASET_ADAPTERS = {
    "coco_construction",
    "spatialtunnel",
    "embspatial_balanced1200",
    "cvbench_pairwise",
}
DIRECT_DECODING_PROTOCOL = "greedy_32k_no_thinking_budget_v1"
MOLMO_DIRECT_DECODING_PROTOCOL = "greedy_128_no_thinking_v1"
DIRECT_MODEL_ADAPTERS = {*QWEN_MODEL_SPECS, *MOLMO_MODEL_SPECS}


@dataclass(frozen=True)
class DirectGenerationConfig:
    """Define one complete sharded matched-pair generation evaluation."""

    schema_version: str
    evaluation_id: str
    model_adapter: str
    model_path: Path
    dataset_adapter: str
    dataset_source: Path
    parent_dataset: Path | None
    split: str
    expected_sample_count: int
    expected_dataset_fingerprint: str
    prompt_style: str
    thinking: bool
    decoding_protocol: str
    max_new_tokens: int
    image_max_pixels: int | None
    output_dir: Path
    shard_count: int
    device: str
    checkpoint_every: int
    seed: int

    def validate(self) -> None:
        """Reject any configuration outside the frozen EVAL-07 protocol."""
        if self.schema_version != "1.0.0":
            raise ValueError("Direct-generation schema_version must be 1.0.0")
        if not self.evaluation_id.strip():
            raise ValueError("evaluation_id must be non-empty")
        if self.model_adapter not in DIRECT_MODEL_ADAPTERS:
            raise ValueError(f"Unsupported direct model adapter {self.model_adapter!r}")
        if self.dataset_adapter not in DIRECT_DATASET_ADAPTERS:
            raise ValueError(f"Unknown direct dataset adapter {self.dataset_adapter!r}")
        if self.prompt_style != "baseline":
            raise ValueError("EVAL-07 requires the frozen baseline prompt style")
        if self.model_adapter in QWEN_MODEL_SPECS:
            if self.decoding_protocol != DIRECT_DECODING_PROTOCOL:
                raise ValueError(
                    f"Qwen decoding_protocol must be {DIRECT_DECODING_PROTOCOL!r}"
                )
            if self.max_new_tokens != 32768:
                raise ValueError("Qwen EVAL-07 requires max_new_tokens=32768")
        else:
            if self.thinking:
                raise ValueError(
                    "Molmo matched-pair generation does not support Thinking"
                )
            if self.decoding_protocol != MOLMO_DIRECT_DECODING_PROTOCOL:
                raise ValueError(
                    "Molmo decoding_protocol must be "
                    f"{MOLMO_DIRECT_DECODING_PROTOCOL!r}"
                )
            if self.max_new_tokens != 128:
                raise ValueError("Molmo EVAL-07 requires max_new_tokens=128")
            if self.image_max_pixels is not None:
                raise ValueError(
                    "Molmo direct generation requires image_max_pixels=null"
                )
        if self.image_max_pixels is not None:
            if isinstance(self.image_max_pixels, bool) or not isinstance(
                self.image_max_pixels, int
            ):
                raise ValueError("image_max_pixels must be an integer or null")
            spec = QWEN_MODEL_SPECS[self.model_adapter]
            if not (
                spec.image_min_pixels <= self.image_max_pixels <= spec.image_max_pixels
            ):
                raise ValueError(
                    "image_max_pixels must be within the pinned Qwen bounds"
                )
        if self.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported direct-generation split {self.split!r}")
        if self.dataset_adapter == "coco_construction":
            if self.split not in {"train", "validation"}:
                raise ValueError("COCO construction split must be train or validation")
            if self.split == "validation" and self.parent_dataset is None:
                raise ValueError("COCO validation requires the parent dataset")
            if self.split == "train" and self.parent_dataset is not None:
                raise ValueError("COCO train must not declare a parent dataset")
        elif self.split != "test" or self.parent_dataset is not None:
            raise ValueError("Pairwise benchmarks require split=test and no parent")
        if self.expected_sample_count <= 0:
            raise ValueError("expected_sample_count must be positive")
        fingerprint = self.expected_dataset_fingerprint
        if len(fingerprint) != 64 or any(
            c not in "0123456789abcdef" for c in fingerprint
        ):
            raise ValueError("expected_dataset_fingerprint must be lowercase SHA-256")
        if self.shard_count <= 0 or self.checkpoint_every <= 0:
            raise ValueError("shard_count and checkpoint_every must be positive")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for path in (self.model_path, self.dataset_source):
            if not path.exists():
                raise FileNotFoundError(path)
        if self.parent_dataset is not None and not self.parent_dataset.exists():
            raise FileNotFoundError(self.parent_dataset)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


def load_direct_generation_config(
    path: Path, project_root: Path
) -> DirectGenerationConfig:
    """Load exact EVAL-07 JSON fields and resolve project-relative paths."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "evaluation_id",
        "model_adapter",
        "model_path",
        "dataset_adapter",
        "dataset_source",
        "parent_dataset",
        "split",
        "expected_sample_count",
        "expected_dataset_fingerprint",
        "prompt_style",
        "thinking",
        "decoding_protocol",
        "max_new_tokens",
        "image_max_pixels",
        "output_dir",
        "shard_count",
        "device",
        "checkpoint_every",
        "seed",
    } | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"Direct-generation config fields {set(values)} != {expected}")
    validate_launch_metadata(values, "direct_generation")

    parent_value = values["parent_dataset"]
    if parent_value is not None and not isinstance(parent_value, str):
        raise ValueError("parent_dataset must be a path string or null")
    if not isinstance(values["thinking"], bool):
        raise ValueError("thinking must be a JSON boolean")
    image_max_pixels = values["image_max_pixels"]
    if image_max_pixels is not None and (
        isinstance(image_max_pixels, bool) or not isinstance(image_max_pixels, int)
    ):
        raise ValueError("image_max_pixels must be a JSON integer or null")
    config = DirectGenerationConfig(
        schema_version=str(values["schema_version"]),
        evaluation_id=str(values["evaluation_id"]),
        model_adapter=str(values["model_adapter"]),
        model_path=resolve_project_path(
            values["model_path"], project_root, "model_path"
        ),
        dataset_adapter=str(values["dataset_adapter"]),
        dataset_source=resolve_project_path(
            values["dataset_source"], project_root, "dataset_source"
        ),
        parent_dataset=(
            None
            if parent_value is None
            else resolve_project_path(parent_value, project_root, "parent_dataset")
        ),
        split=str(values["split"]),
        expected_sample_count=int(values["expected_sample_count"]),
        expected_dataset_fingerprint=str(values["expected_dataset_fingerprint"]),
        prompt_style=str(values["prompt_style"]),
        thinking=values["thinking"],
        decoding_protocol=str(values["decoding_protocol"]),
        max_new_tokens=int(values["max_new_tokens"]),
        image_max_pixels=image_max_pixels,
        output_dir=resolve_project_path(
            values["output_dir"], project_root, "output_dir"
        ),
        shard_count=int(values["shard_count"]),
        device=str(values["device"]),
        checkpoint_every=int(values["checkpoint_every"]),
        seed=int(values["seed"]),
    )
    config.validate()
    return config
