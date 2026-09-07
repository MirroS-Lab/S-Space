"""Orchestrate merge, validation, and artifact publication."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from sspace.core.artifacts import ModelSpec, SSpaceArtifact, SSpaceManifest
from sspace.run_records import atomic_json, file_sha256, failure_is_current

from .config import DistributedTrainingConfig
from .data import ConstructionSample
from .extraction.merge import MINIMUM_MEAN_GRADIENT_L2, merge_completed_shards
from .extraction.sharding import SHARDING_PROTOCOL
from .models.runtime import LoadedModelRuntime
from .prompts.templates import AXIS_ORDER, PROMPT_SET_ID, PROMPT_STYLE_ORDER
from sspace.experiments.validation.coco1800.selection.layers import (
    build_validation_evidence,
    validation_metrics,
)
from sspace.experiments.validation.coco1800.selection.readout import (
    extract_validation_margins,
)


def wait_for_worker_shards(
    output_dir: Path,
    world_size: int,
    poll_seconds: float = 5.0,
) -> None:
    """Wait until every worker is sealed, failing on any recorded worker error."""
    if poll_seconds <= 0:
        raise ValueError("Worker poll interval must be positive")
    while True:
        complete = 0
        for rank in range(world_size):
            directory = Path(output_dir) / "shards" / f"rank_{rank:03d}"
            rank_failure = Path(output_dir) / "rank_failures" / f"rank_{rank:03d}.json"
            if rank_failure.exists():
                value = json.loads(rank_failure.read_text(encoding="utf-8"))
                if failure_is_current(value):
                    raise RuntimeError(
                        f"Worker rank {rank} failed before sealing its shard: "
                        f"{value.get('type')}: {value.get('message')}"
                    )
            manifest = directory / "manifest.json"
            if manifest.exists():
                value = json.loads(manifest.read_text(encoding="utf-8"))
                if value.get("status") != "complete":
                    raise ValueError(f"Worker rank {rank} has a non-complete manifest")
                complete += 1
                continue
            failure = directory / "failure.json"
            if failure.exists():
                value = json.loads(failure.read_text(encoding="utf-8"))
                if failure_is_current(value):
                    raise RuntimeError(
                        f"Worker rank {rank} failed at job {value.get('next_job')}: "
                        f"{value.get('type')}: {value.get('message')}"
                    )
        if complete == world_size:
            return
        time.sleep(poll_seconds)


def _load_matching_artifact(
    directory: Path, expected: SSpaceArtifact
) -> SSpaceArtifact:
    """Load an existing immutable artifact only when every value matches."""
    loaded = SSpaceArtifact.load(directory)
    if loaded.manifest.to_dict() != expected.manifest.to_dict():
        raise ValueError("Existing artifact manifest differs from the completed run")
    if not np.array_equal(loaded.axes_tensor, expected.axes_tensor):
        raise ValueError("Existing artifact tensors differ from the completed run")
    return loaded


def _validation_metadata(
    samples: Sequence[ConstructionSample], selected_indices: Sequence[int]
) -> tuple[list[str], list[str], list[str]]:
    groups: list[str] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    for index in selected_indices:
        sample = samples[int(index)]
        for _style in PROMPT_STYLE_ORDER:
            groups.append(sample.group)
            labels.append(sample.label)
            sample_ids.append(sample.sample_id)
    return groups, labels, sample_ids


def finalize_distributed_training(
    config: DistributedTrainingConfig,
    runtime: LoadedModelRuntime,
    train_manifest: dict[str, Any],
    validation_manifest: dict[str, Any],
    train_samples: Sequence[ConstructionSample],
    validation_samples: Sequence[ConstructionSample],
    selected_train_indices: Sequence[int],
    selected_validation_indices: Sequence[int],
    run_config: dict[str, Any],
) -> SSpaceArtifact:
    """Strictly merge shards, select on validation, and publish ART-01–03."""
    wait_for_worker_shards(config.output_dir, config.world_size)
    merge_manifest = merge_completed_shards(
        config.output_dir,
        train_samples,
        selected_train_indices,
        run_config,
        config.world_size,
        runtime.spec.layer_count,
        runtime.spec.hidden_size,
    )
    merged_dir = config.output_dir / "merged"
    with np.load(merged_dir / "axes.npz", allow_pickle=False) as archive:
        axes = archive["axes"].astype(np.float32, copy=True)
        layer_ids = archive["layer_ids"].astype(np.int64, copy=True)
    validation_config = {
        "stage": "distributed_validation_readout",
        "run_fingerprint": merge_manifest["run_fingerprint"],
        "axes_sha256": file_sha256(merged_dir / "axes.npz"),
        "validation_sample_indices": list(map(int, selected_validation_indices)),
        "prompt_set": PROMPT_SET_ID,
        "orientations": ["original", "swapped"],
    }
    extract_validation_margins(
        runtime,
        validation_samples,
        selected_validation_indices,
        axes,
        layer_ids,
        config.output_dir / "validation",
        validation_config,
        config.checkpoint_every,
    )
    validation_margins = np.load(
        config.output_dir / "validation" / "margins.npy", mmap_mode="r"
    )
    groups, labels, sample_ids = _validation_metadata(
        validation_samples, selected_validation_indices
    )
    metrics = validation_metrics(
        validation_margins[:, 0, :],
        groups,
        labels,
        sample_ids,
        layer_ids,
    )
    validation_record = build_validation_evidence(
        dataset_id=validation_manifest["dataset_id"],
        dataset_fingerprint=validation_manifest["dataset_fingerprint"],
        metrics=metrics,
        train_image_ids=[
            train_samples[int(index)].image_id for index in selected_train_indices
        ],
        validation_image_ids=[
            validation_samples[int(index)].image_id
            for index in selected_validation_indices
        ],
    )
    selected_layer = int(validation_record["selected_layer"])
    validation_metrics_path = config.output_dir / "validation_metrics.json"
    atomic_json(validation_metrics_path, validation_record)

    manifest = SSpaceManifest(
        artifact_id=config.artifact_id,
        method="final_logit_jacobian",
        model=ModelSpec(
            model_id=runtime.spec.model_id,
            revision=runtime.spec.revision,
            architecture=type(runtime.model).__name__,
            tokenizer_id=runtime.spec.model_id,
            tokenizer_revision=runtime.spec.revision,
            hidden_size=runtime.spec.hidden_size,
            layer_ids=tuple(map(int, layer_ids)),
            selected_layer=selected_layer,
        ),
        construction={
            "dataset_id": train_manifest["dataset_id"],
            "dataset_manifest_sha256": file_sha256(config.dataset / "manifest.json"),
            "dataset_fingerprint": config.dataset_fingerprint,
            "run_kind": config.run_kind,
            "train_samples": len(selected_train_indices),
            "validation_samples": len(selected_validation_indices),
            "prompt_set": PROMPT_SET_ID,
            "prompt_styles": list(PROMPT_STYLE_ORDER),
            "orientations": ["original", "swapped"],
            "orientation_batch_size": runtime.spec.orientation_batch_size,
            "attention_backend": runtime.spec.attention_backend,
            "sdp_kernel": runtime.spec.sdp_kernel,
            "cpu_threads": 1,
            "cublas_workspace_config": ":4096:8",
            "float32_matmul_precision": "highest",
            "object_role_order": ["query", "reference"],
            "object_token_protocol": "last_overlapping_subtoken",
            "role_gradient": "(grad_query(z_g)-grad_reference(z_g))/2",
            "axis_order": list(AXIS_ORDER),
            "sharding_protocol": SHARDING_PROTOCOL,
            "world_size": config.world_size,
            "canonical_sum_order": merge_manifest["canonical_sum_order"],
            "axis_statistics": {
                "protocol": "mean_gradient_l2_gate_v1",
                "minimum_mean_gradient_l2": MINIMUM_MEAN_GRADIENT_L2,
                "raw_mean_gradient_l2": merge_manifest["raw_mean_gradient_l2"],
            },
            "raw_worker_cache": {
                "object_states": "[J_rank,2,L_model,2,D] float32",
                "role_gradients": "[J_rank,2,L_model,D] float32",
                "logit_contrasts": "[J_rank,2,3] float32",
            },
            "merged_evidence": {
                "all_axis_margins": "[J,2,L_valid,3] float32",
                "task_margins": "[J,2,L_valid] float32",
            },
            "model_source": config.model_source,
            "validation": validation_record,
        },
        software={
            name: str(run_config["numerical_runtime"][name])
            for name in ("python", "torch", "numpy", "transformers", "safetensors")
        },
    )
    artifact = SSpaceArtifact(manifest, axes)
    artifact.validate()
    if config.artifact_dir.exists():
        return _load_matching_artifact(config.artifact_dir, artifact)
    artifact.save(config.artifact_dir)
    return SSpaceArtifact.load(config.artifact_dir)
