"""Run a formal SpinBench direct-generation condition."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.identity import bind_launch_identity
from sspace.run_records import RunRecorder, file_sha256
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    configure_image_max_pixels,
    load_model_runtime,
)

from .adapter import load_spinbench, spinbench_fingerprint
from .direct import run_spinbench_direct
from .direct_config import load_spinbench_direct_config


def main() -> None:
    """Validate pinned inputs, load one model, and run one complete condition."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_spinbench_direct_config(args.config, project_root)
    if file_sha256(config.annotation_path) != config.annotation_sha256:
        raise ValueError("SpinBench annotation bytes changed")
    samples = load_spinbench(
        config.annotation_path,
        config.image_root,
        config.premise_mode,
    )
    fingerprint = spinbench_fingerprint(samples)
    if fingerprint != config.expected_dataset_fingerprint:
        raise ValueError("SpinBench dataset fingerprint changed")
    if args.limit is not None:
        samples = samples[: args.limit]
    spec = RUNTIME_SPECS[config.model_adapter]
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "dataset_fingerprint": fingerprint,
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "transformers_version": spec.transformers_version,
        "dtype": str(spec.dtype),
        "attention_backend": spec.attention_backend,
        "do_sample": False,
        "num_beams": 1,
        "thinking_budget": "unset",
        "sample_limit": args.limit,
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(config.output_dir, resolved, project_root)
    stage = "spinbench_direct"
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
            thinking=config.thinking,
        )
        summary = run_spinbench_direct(
            runtime,
            samples,
            fingerprint,
            config,
            recorder,
            sample_limit=args.limit,
        )
        recorder.event(stage, "complete", **summary["overall"])
        recorder.complete(
            sample_count=len(samples),
            dataset_fingerprint=fingerprint,
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()
