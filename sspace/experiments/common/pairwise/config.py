"""Define strict pairwise projection configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.core.models.runtime import RUNTIME_SPECS
from sspace.core.prompts.templates import PROMPT_STYLE_ORDER
from sspace.core.models.qwen import QWEN_MODEL_SPECS


def _optional_integer(value: Any) -> int | None:
    """Parse an exact optional JSON integer without accepting booleans."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("image_max_pixels must be a JSON integer or null")
    return value


def _optional_layer_id(value: Any) -> int | None:
    """Parse an integer layer ID or explicit JSON null for all layers."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("layer_id must be a JSON integer or null")
    return value


def _validate_image_max_pixels(
    model_adapter: str,
    image_max_pixels: int | None,
) -> None:
    """Validate the explicit image protocol for one runtime adapter."""
    if image_max_pixels is None:
        return
    if model_adapter not in QWEN_MODEL_SPECS:
        raise ValueError("image_max_pixels overrides are supported only by Qwen")
    spec = QWEN_MODEL_SPECS[model_adapter]
    if not (spec.image_min_pixels <= image_max_pixels <= spec.image_max_pixels):
        raise ValueError(
            "Qwen image_max_pixels must be within the pinned processor bounds"
        )


@dataclass(frozen=True)
class PairwiseEvaluationConfig:
    """Define one artifact/model/benchmark pairwise evaluation run."""

    artifact_dir: Path
    model_adapter: str
    model_path: Path
    benchmark: str
    benchmark_source: Path
    expected_dataset_fingerprint: str
    output_dir: Path
    layer_id: int | None
    prompt_style: str
    decision_protocol: str
    image_max_pixels: int | None
    device: str
    checkpoint_every: int
    seed: int

    def validate(self) -> None:
        """Reject unknown protocols and absent immutable inputs."""
        if self.model_adapter not in RUNTIME_SPECS:
            raise ValueError(
                f"Unknown runtime model adapter {self.model_adapter!r}; "
                f"expected {tuple(RUNTIME_SPECS)}"
            )
        if self.prompt_style not in PROMPT_STYLE_ORDER:
            raise ValueError(f"Unknown prompt style {self.prompt_style!r}")
        if self.decision_protocol != "margin_sign_v1":
            raise ValueError(f"Unknown decision protocol {self.decision_protocol!r}")
        if self.layer_id is not None and self.layer_id < 0:
            raise ValueError("layer_id must be nonnegative or null")
        if self.checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be positive")
        if len(self.expected_dataset_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.expected_dataset_fingerprint
        ):
            raise ValueError("Pairwise dataset fingerprint must be lowercase SHA-256")
        _validate_image_max_pixels(self.model_adapter, self.image_max_pixels)
        for path in (self.artifact_dir, self.model_path, self.benchmark_source):
            if not path.exists():
                raise FileNotFoundError(path)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe resolved configuration."""
        return {
            name: str(value) if isinstance(value, Path) else value
            for name, value in asdict(self).items()
        }


def load_evaluation_config(path: Path, project_root: Path) -> PairwiseEvaluationConfig:
    """Load exact JSON fields and resolve project-relative paths."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "artifact_dir",
        "model_adapter",
        "model_path",
        "benchmark",
        "benchmark_source",
        "expected_dataset_fingerprint",
        "output_dir",
        "layer_id",
        "prompt_style",
        "decision_protocol",
        "image_max_pixels",
        "device",
        "checkpoint_every",
        "seed",
    } | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"Evaluation config fields {set(values)} != {expected}")
    validate_launch_metadata(values, "pairwise_projection")

    def resolve(name: str) -> Path:
        return resolve_project_path(values[name], project_root, name)

    config = PairwiseEvaluationConfig(
        artifact_dir=resolve("artifact_dir"),
        model_adapter=str(values["model_adapter"]),
        model_path=resolve("model_path"),
        benchmark=str(values["benchmark"]),
        benchmark_source=resolve("benchmark_source"),
        expected_dataset_fingerprint=str(values["expected_dataset_fingerprint"]),
        output_dir=resolve("output_dir"),
        layer_id=_optional_layer_id(values["layer_id"]),
        prompt_style=str(values["prompt_style"]),
        decision_protocol=str(values["decision_protocol"]),
        image_max_pixels=_optional_integer(values["image_max_pixels"]),
        device=str(values["device"]),
        checkpoint_every=int(values["checkpoint_every"]),
        seed=int(values["seed"]),
    )
    config.validate()
    return config


def load_bound_artifact(config: PairwiseEvaluationConfig) -> SSpaceArtifact:
    """Load axes only when the artifact matches the runtime adapter.

    Args:
        config: Validated all-layer evaluation configuration.

    Returns:
        A checksum-validated artifact whose model identity and exported
        post-block layers match the configured runtime adapter.

    Raises:
        FileNotFoundError: A required artifact file is absent.
        ValueError: Artifact schema, checksums, model identity, tokenizer
            identity, hidden size, architecture, or layer availability differs.

    Side effects:
        Reads the artifact directory without modifying it.
    """
    artifact = SSpaceArtifact.load(config.artifact_dir)
    adapter = RUNTIME_SPECS[config.model_adapter]
    artifact.validate_model(
        ModelSpec(
            model_id=adapter.model_id,
            revision=adapter.revision,
            architecture=adapter.architecture,
            tokenizer_id=adapter.model_id,
            tokenizer_revision=adapter.revision,
            hidden_size=adapter.hidden_size,
            layer_ids=tuple(range(adapter.layer_count)),
            selected_layer=artifact.manifest.model.selected_layer,
        )
    )
    return artifact
