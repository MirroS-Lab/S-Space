"""Define strict SpinBench direct-inference configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.core.models.runtime import RUNTIME_SPECS
from sspace.core.models.qwen import QWEN36_SPEC, QWEN_MODEL_SPECS

from .adapter import SPINBENCH_TEST_SHA256


SPINBENCH_DECODING_PROTOCOL = "greedy_32k_no_thinking_budget_v1"
SPINBENCH_PAIRED_DECODING_PROTOCOL = "greedy_32k_thinking_toggle_v2"
SPINBENCH_QWEN_NO_THINKING_PROTOCOL = "greedy_128_no_thinking_v1"
SPINBENCH_MOLMO_DECODING_PROTOCOL = "greedy_128_no_thinking_v1"


@dataclass(frozen=True)
class SpinBenchDirectConfig:
    """Define one complete formal SpinBench direct-generation run."""

    schema_version: str
    evaluation_id: str
    model_adapter: str
    model_path: Path
    annotation_path: Path
    image_root: Path
    premise_mode: str
    annotation_sha256: str
    expected_sample_count: int
    expected_dataset_fingerprint: str
    thinking: bool
    decoding_protocol: str
    max_new_tokens: int
    image_max_pixels: int | None
    output_dir: Path
    device: str
    checkpoint_every: int
    seed: int

    def validate(self) -> None:
        """Reject any field outside the frozen SpinBench Thinking protocol."""
        if self.schema_version != "1.0.0" or not self.evaluation_id.strip():
            raise ValueError("SpinBench direct schema or evaluation ID is invalid")
        if self.model_adapter not in RUNTIME_SPECS:
            raise ValueError("SpinBench direct evaluation requires a pinned model")
        if self.premise_mode not in {"without", "with"}:
            raise ValueError("SpinBench premise_mode must be 'without' or 'with'")
        expected_count = 146 if self.premise_mode == "without" else 144
        if self.expected_sample_count != expected_count:
            raise ValueError(
                f"SpinBench {self.premise_mode} requires {expected_count} rows"
            )
        if self.annotation_sha256 != SPINBENCH_TEST_SHA256:
            raise ValueError("SpinBench annotation checksum is not pinned")
        fingerprint = self.expected_dataset_fingerprint
        if len(fingerprint) != 64 or any(
            character not in "0123456789abcdef" for character in fingerprint
        ):
            raise ValueError("SpinBench fingerprint must be lowercase SHA-256")
        if self.model_adapter in QWEN_MODEL_SPECS:
            if self.decoding_protocol == SPINBENCH_DECODING_PROTOCOL:
                if self.model_adapter != QWEN36_SPEC.key or self.thinking is not True:
                    raise ValueError(
                        "Legacy SpinBench direct protocol is Qwen3.6 Thinking-only"
                    )
            elif self.thinking:
                if self.decoding_protocol != SPINBENCH_PAIRED_DECODING_PROTOCOL:
                    raise ValueError("SpinBench Qwen Thinking protocol is not frozen")
            elif self.decoding_protocol != SPINBENCH_QWEN_NO_THINKING_PROTOCOL:
                raise ValueError("SpinBench Qwen no-Thinking protocol is not frozen")
            expected_tokens = 32768 if self.thinking else 128
            if self.max_new_tokens != expected_tokens:
                raise ValueError(
                    "SpinBench Qwen direct generation requires "
                    f"max_new_tokens={expected_tokens} when thinking={self.thinking}"
                )
        else:
            if (
                self.thinking
                or self.decoding_protocol != SPINBENCH_MOLMO_DECODING_PROTOCOL
            ):
                raise ValueError(
                    "SpinBench Molmo direct generation must disable Thinking"
                )
            if self.max_new_tokens != 128:
                raise ValueError(
                    "SpinBench Molmo direct generation requires max_new_tokens=128"
                )
        if self.image_max_pixels is not None:
            if isinstance(self.image_max_pixels, bool) or not isinstance(
                self.image_max_pixels, int
            ):
                raise ValueError("image_max_pixels must be an integer or null")
            if self.model_adapter not in QWEN_MODEL_SPECS:
                raise ValueError("SpinBench Molmo image_max_pixels must be null")
            qwen_spec = QWEN_MODEL_SPECS[self.model_adapter]
            if (
                not qwen_spec.image_min_pixels
                <= self.image_max_pixels
                <= qwen_spec.image_max_pixels
            ):
                raise ValueError("SpinBench image_max_pixels is outside pinned bounds")
        if self.checkpoint_every <= 0 or self.seed < 0:
            raise ValueError("SpinBench checkpoint interval or seed is invalid")
        for path in (self.model_path, self.annotation_path, self.image_root):
            if not path.exists():
                raise FileNotFoundError(path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


def load_spinbench_direct_config(
    path: Path,
    project_root: Path,
) -> SpinBenchDirectConfig:
    """Load exact EVAL-06 JSON fields and resolve project-relative paths."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = set(SpinBenchDirectConfig.__dataclass_fields__) | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"SpinBench direct config fields {set(values)} != {expected}")
    validate_launch_metadata(values, "spinbench_direct")

    if not isinstance(values["thinking"], bool):
        raise ValueError("thinking must be a JSON boolean")
    image_max_pixels = values["image_max_pixels"]
    config = SpinBenchDirectConfig(
        schema_version=str(values["schema_version"]),
        evaluation_id=str(values["evaluation_id"]),
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
        thinking=values["thinking"],
        decoding_protocol=str(values["decoding_protocol"]),
        max_new_tokens=int(values["max_new_tokens"]),
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
