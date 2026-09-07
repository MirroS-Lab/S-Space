"""Run the frozen H* HOS-600 all-layer image-order evaluation."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.identity import bind_launch_identity
from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.experiments.multi_view.hstar.hstar_multiview import (
    hstar_multiview_fingerprint,
    load_hstar_multiview,
)
from sspace.run_records import RunRecorder, file_sha256
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    configure_image_max_pixels,
    load_model_runtime,
)

from .hstar_config import load_hstar_multiview_config
from sspace.experiments.multi_view.common.runner import (
    run_layerwise_multiview_order_evaluation,
)
from sspace.experiments.multi_view.common.schema import balanced_axis_subset


def main() -> None:
    """Validate PRE-06 inputs, load one model, and run EVAL-13."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_hstar_multiview_config(args.config, project_root)
    samples = load_hstar_multiview(config.dataset_dir)
    dataset_fingerprint = hstar_multiview_fingerprint(samples)
    if dataset_fingerprint != config.expected_dataset_fingerprint:
        raise ValueError("H* multi-view dataset fingerprint changed")
    samples = balanced_axis_subset(samples, args.limit)
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
            selected_layer=artifact.manifest.model.selected_layer,
        )
    )
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_manifest_sha256": file_sha256(config.dataset_dir / "manifest.json"),
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "transformers_version": spec.transformers_version,
        "dtype": str(spec.dtype),
        "attention_backend": spec.attention_backend,
        "layer_ids": list(artifact.manifest.model.layer_ids),
        "sample_limit": args.limit,
        "selected_sample_ids": [sample.sample_id for sample in samples],
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(config.output_dir, resolved, project_root)
    stage = "hstar_hos600_multiview_layerwise_order_swap"
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        runtime = load_model_runtime(
            config.model_adapter, config.model_path, config.device
        )
        configure_image_max_pixels(runtime, config.image_max_pixels)
        recorder.event(stage, "started", sample_count=len(samples))
        summary = run_layerwise_multiview_order_evaluation(
            runtime,
            samples,
            dataset_fingerprint,
            artifact.axes_tensor,
            artifact.manifest.model.layer_ids,
            config,
            config.output_dir,
            resolved,
            sample_limit=args.limit,
        )
        recorder.event(
            stage,
            "complete",
            diagnostic_best_layer=summary["diagnostic_best_layer"],
            diagnostic_best_accuracy=summary["diagnostic_best_metrics"]["overall"][
                "accuracy"
            ],
        )
        recorder.complete(
            sample_count=len(samples), dataset_fingerprint=dataset_fingerprint
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()
