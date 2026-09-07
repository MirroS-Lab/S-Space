"""Define strict configuration for formal sample-parallel training."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from sspace.config_paths import resolve_project_path

from .extraction.sharding import SHARDING_PROTOCOL
from .models.runtime import RUNTIME_SPECS
from .prompts.templates import PROMPT_SET_ID


COCO6000_FINGERPRINT = (
    "d826f948bc3db36c0cecd965118a6fa96c53ac29d91e7bf84a3b2abc2099fe6f"
)
COCO1800_FINGERPRINT = (
    "125146e175ddb8f688628caca0e4544e3049bc12a80fca2924d75b192e8b7c3d"
)
CPU_THREADS = 1


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _json_integer(value: Any, field: str) -> int:
    if not _is_integer(value):
        raise ValueError(f"{field} must be a JSON integer")
    return value


@dataclass(frozen=True)
class SampleSelection:
    """Select either the complete split or explicit global pilot indices."""

    mode: str
    indices: tuple[int, ...]

    def validate(self, run_kind: str) -> None:
        if not isinstance(self.mode, str) or any(
            not _is_integer(index) for index in self.indices
        ):
            raise ValueError("Sample selection mode and indices have invalid types")
        if self.mode == "all":
            if self.indices:
                raise ValueError("All-sample selection cannot include explicit indices")
        elif self.mode == "explicit_indices":
            if run_kind == "full":
                raise ValueError("Full runs must use all split samples")
            if not self.indices or tuple(sorted(set(self.indices))) != self.indices:
                raise ValueError(
                    "Pilot indices must be non-empty, unique, and increasing"
                )
            if self.indices[0] < 0:
                raise ValueError("Pilot indices must be nonnegative")
        else:
            raise ValueError(f"Unknown sample selection mode {self.mode!r}")

    def resolve(self, sample_count: int) -> tuple[int, ...]:
        if self.mode == "all":
            return tuple(range(sample_count))
        if self.indices[-1] >= sample_count:
            raise ValueError("Explicit sample selection exceeds the split size")
        return self.indices


@dataclass(frozen=True)
class DistributedTrainingConfig:
    """Define one complete one-or-more-worker Final-logit Jacobian run."""

    schema_version: str
    run_kind: str
    model_adapter: str
    model_path: Path
    model_source: str
    dataset: Path
    dataset_fingerprint: str
    validation_dataset: Path
    validation_dataset_fingerprint: str
    world_size: int
    train_selection: SampleSelection
    validation_selection: SampleSelection
    checkpoint_every: int
    seed: int
    output_dir: Path
    artifact_dir: Path
    artifact_id: str

    def validate(self) -> None:
        """Reject any value outside the frozen sample-parallel protocol."""
        if self.schema_version != "1.2.0":
            raise ValueError("Distributed config schema_version must be 1.2.0")
        if self.run_kind not in {"smoke", "equivalence", "full"}:
            raise ValueError(f"Unknown distributed run kind {self.run_kind!r}")
        if self.model_adapter not in RUNTIME_SPECS:
            raise ValueError(f"Unknown model adapter {self.model_adapter!r}")
        if not self.model_source:
            raise ValueError("model_source must record local snapshot provenance")
        if (
            not self.model_path.is_dir()
            or not self.dataset.is_dir()
            or not self.validation_dataset.is_dir()
        ):
            raise FileNotFoundError(
                "Model, construction dataset, and validation dataset must exist"
            )
        if self.dataset_fingerprint != COCO6000_FINGERPRINT:
            raise ValueError("Dataset fingerprint differs from frozen COCO-6000")
        if self.validation_dataset_fingerprint != COCO1800_FINGERPRINT:
            raise ValueError(
                "Validation fingerprint differs from the frozen run protocol"
            )
        if self.validation_dataset == self.dataset:
            raise ValueError("Training requires a separate COCO-1800 dataset")
        if not _is_integer(self.world_size) or not _is_integer(self.checkpoint_every):
            raise ValueError("world_size and checkpoint_every must be integers")
        if not _is_integer(self.seed) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if self.world_size <= 0 or self.checkpoint_every <= 0:
            raise ValueError("world_size and checkpoint_every must be positive")
        if self.output_dir == self.artifact_dir:
            raise ValueError(
                "Run output and immutable artifact directories must differ"
            )
        if not self.artifact_id:
            raise ValueError("artifact_id must be non-empty")
        self.train_selection.validate(self.run_kind)
        self.validation_selection.validate(self.run_kind)

    def to_dict(self) -> dict[str, Any]:
        """Return the lean input plus registry-derived runtime provenance."""
        values = asdict(self)
        for name in (
            "model_path",
            "dataset",
            "validation_dataset",
            "output_dir",
            "artifact_dir",
        ):
            values[name] = str(values[name])
        values["train_selection"]["indices"] = list(self.train_selection.indices)
        values["validation_selection"]["indices"] = list(
            self.validation_selection.indices
        )
        spec = RUNTIME_SPECS[self.model_adapter]
        values["runtime"] = {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "tokenizer_id": spec.model_id,
            "tokenizer_revision": spec.revision,
            "transformers_version": spec.transformers_version,
            "layer_count": spec.layer_count,
            "hidden_size": spec.hidden_size,
            "dtype": str(spec.dtype).removeprefix("torch."),
            "attention_backend": spec.attention_backend,
            "sdp_kernel": spec.sdp_kernel,
            "orientation_batch_size": spec.orientation_batch_size,
        }
        values["cpu_threads"] = CPU_THREADS
        values["prompt_set"] = PROMPT_SET_ID
        values["sharding_protocol"] = SHARDING_PROTOCOL
        return values


def _selection(value: Any, field: str) -> SampleSelection:
    if not isinstance(value, dict) or set(value) != {"mode", "indices"}:
        raise ValueError(f"{field} must contain exactly mode and indices")
    if not isinstance(value["mode"], str):
        raise ValueError(f"{field}.mode must be a JSON string")
    if not isinstance(value["indices"], list):
        raise ValueError(f"{field}.indices must be a JSON list")
    indices = tuple(
        _json_integer(index, f"{field}.indices") for index in value["indices"]
    )
    return SampleSelection(value["mode"], indices)


def load_distributed_config(
    path: Path, project_root: Path
) -> DistributedTrainingConfig:
    """Load exact JSON fields and resolve every project-relative path."""
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version",
        "run_kind",
        "model_adapter",
        "model_path",
        "model_source",
        "dataset",
        "dataset_fingerprint",
        "validation_dataset",
        "validation_dataset_fingerprint",
        "world_size",
        "train_selection",
        "validation_selection",
        "checkpoint_every",
        "seed",
        "output_dir",
        "artifact_dir",
        "artifact_id",
    }
    if set(values) != expected:
        raise ValueError(f"Distributed config fields {set(values)} != {expected}")

    def resolve(name: str) -> Path:
        return resolve_project_path(values[name], project_root, name)

    config = DistributedTrainingConfig(
        schema_version=str(values["schema_version"]),
        run_kind=str(values["run_kind"]),
        model_adapter=str(values["model_adapter"]),
        model_path=resolve("model_path"),
        model_source=str(values["model_source"]),
        dataset=resolve("dataset"),
        dataset_fingerprint=str(values["dataset_fingerprint"]),
        validation_dataset=resolve("validation_dataset"),
        validation_dataset_fingerprint=str(values["validation_dataset_fingerprint"]),
        world_size=_json_integer(values["world_size"], "world_size"),
        train_selection=_selection(values["train_selection"], "train_selection"),
        validation_selection=_selection(
            values["validation_selection"], "validation_selection"
        ),
        checkpoint_every=_json_integer(values["checkpoint_every"], "checkpoint_every"),
        seed=_json_integer(values["seed"], "seed"),
        output_dir=resolve("output_dir"),
        artifact_dir=resolve("artifact_dir"),
        artifact_id=str(values["artifact_id"]),
    )
    config.validate()
    return config
