"""Run one formal-artifact contextual S-Space case manifest."""

from __future__ import annotations

import argparse
import json
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

from .config import load_case_config
from .runner import run_case_evaluation
from .schema import load_case_manifest


def main() -> None:
    """Validate exact inputs, run token projections, and seal all evidence."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_case_config(args.config, project_root)
    adapter = RUNTIME_SPECS[config.model_adapter]
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "manifest_sha256": file_sha256(config.manifest_path),
        "model_id": adapter.model_id,
        "model_revision": adapter.revision,
        "transformers_version": adapter.transformers_version,
        "dtype": str(adapter.dtype),
        "attention_backend": adapter.attention_backend,
        "sample_limit": args.limit,
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(config.output_dir, resolved, project_root)
    stage = "contextual_case_projection"
    try:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        artifact = SSpaceArtifact.load(config.artifact_dir)
        samples = load_case_manifest(config.manifest_path, config.image_root)
        if args.limit is not None:
            samples = samples[: args.limit]
        runtime = load_model_runtime(
            config.model_adapter, config.model_path, config.device
        )
        configure_image_max_pixels(runtime, config.image_max_pixels)
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
        recorder.event(stage, "started", sample_count=len(samples))
        summary = run_case_evaluation(runtime, artifact, samples, config, resolved)
        summary_path = config.output_dir / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        recorder.event(stage, "complete", summary=summary)
        recorder.complete(summary=summary)
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()
