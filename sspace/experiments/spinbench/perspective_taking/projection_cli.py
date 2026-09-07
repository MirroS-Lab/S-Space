"""Run a SpinBench rotation-projection condition."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.identity import bind_launch_identity
from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.run_records import RunRecorder, file_sha256
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    configure_image_max_pixels,
    load_model_runtime,
)

from .adapter import (
    VIEW_ROTATIONS,
    load_spinbench,
    spinbench_fingerprint,
)
from .projection_config import load_spinbench_projection_config
from .projection_runner import run_spinbench_projection


def main() -> None:
    """Validate pinned inputs, load one model, and run EVAL-09."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_spinbench_projection_config(args.config, project_root)
    if file_sha256(config.annotation_path) != config.annotation_sha256:
        raise ValueError("SpinBench annotation bytes changed")
    samples = load_spinbench(
        config.annotation_path,
        config.image_root,
        config.premise_mode,
    )
    dataset_fingerprint = spinbench_fingerprint(samples)
    if dataset_fingerprint != config.expected_dataset_fingerprint:
        raise ValueError("SpinBench dataset fingerprint changed")
    if args.limit is not None:
        samples = samples[: args.limit]
    artifact = SSpaceArtifact.load(config.artifact_dir)
    spec = RUNTIME_SPECS[config.model_adapter]
    artifact.validate_model(
        ModelSpec(
            model_id=spec.model_id,
            revision=spec.revision,
            architecture=spec.architecture,
            tokenizer_id=spec.model_id,
            tokenizer_revision=spec.revision,
            hidden_size=spec.hidden_size,
            layer_ids=tuple(range(spec.layer_count)),
            selected_layer=config.selected_layer,
        )
    )
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "artifact_id": artifact.manifest.artifact_id,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "transformers_version": spec.transformers_version,
        "dtype": str(spec.dtype),
        "attention_backend": spec.attention_backend,
        "orientation_batch_size": spec.orientation_batch_size,
        "axis_order": list(artifact.manifest.axis_order),
        "readout_selection_manifest_sha256": (
            None
            if config.readout_selection_manifest is None
            else file_sha256(config.readout_selection_manifest)
        ),
        "rotation_matrices_hvd": {
            view: matrix.tolist() for view, matrix in VIEW_ROTATIONS.items()
        },
        "sample_limit": args.limit,
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(config.output_dir, resolved, project_root)
    stage = "spinbench_projection"
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        runtime = load_model_runtime(
            config.model_adapter,
            config.model_path,
            config.device,
        )
        configure_image_max_pixels(runtime, config.image_max_pixels)
        recorder.event(
            stage,
            "started",
            sample_count=len(samples),
            premise_mode=config.premise_mode,
            selected_layer=config.selected_layer,
        )
        summary = run_spinbench_projection(
            runtime,
            artifact,
            samples,
            dataset_fingerprint,
            config,
            config.output_dir,
            resolved,
            sample_limit=args.limit,
        )
        recorder.event(
            stage,
            "complete",
            **summary["overall"]["margin"],
        )
        recorder.complete(
            sample_count=len(samples),
            dataset_fingerprint=dataset_fingerprint,
            artifact_id=artifact.manifest.artifact_id,
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()
