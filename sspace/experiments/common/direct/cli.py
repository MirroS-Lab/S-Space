"""Run matched-pair direct-generation shards."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.identity import (
    bind_launch_identity,
    experiment_launch_fingerprint,
)
from sspace.run_records import RunRecorder, file_sha256
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    configure_image_max_pixels,
    load_model_runtime,
)

from .config import DirectGenerationConfig, load_direct_generation_config
from .dataset import load_direct_generation_samples
from .generation import run_direct_generation_shard


def _effective_shard_count(config: DirectGenerationConfig, limit: int | None) -> int:
    """Return the non-empty shard count for one independently sized dataset."""
    sample_count = config.expected_sample_count
    if limit is not None:
        sample_count = min(sample_count, limit)
    return min(config.shard_count, sample_count)


def _completed_and_valid(directory: Path, launch_fingerprint: str) -> bool:
    """Return true only for a completed shard whose declared files still match."""
    manifest_path = directory / "run_manifest.json"
    checksums_path = directory / "checksums.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = directory / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed direct shard checksum changed: {path}")
    resolved_path = directory / "resolved_config.json"
    if not resolved_path.is_file():
        return False
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return resolved.get("launch_fingerprint") == launch_fingerprint


def _resolved(
    config: DirectGenerationConfig,
    config_path: Path,
    dataset_fingerprint: str,
    rank: int,
    sample_limit: int | None = None,
    effective_shard_count: int | None = None,
) -> dict:
    spec = RUNTIME_SPECS[config.model_adapter]
    resolved = {
        **config.to_dict(),
        "config_path": str(config_path.resolve()),
        "dataset_fingerprint": dataset_fingerprint,
        "rank": rank,
        "effective_shard_count": effective_shard_count or config.shard_count,
        "sample_limit": sample_limit,
        "sharding_protocol": "global_sample_index_mod_world_size_v1",
        "model_id": spec.model_id,
        "model_revision": spec.revision,
        "transformers_version": spec.transformers_version,
        "dtype": str(spec.dtype),
        "attention_backend": spec.attention_backend,
        "do_sample": False,
        "num_beams": 1,
        "thinking_budget": "unset",
    }
    bind_launch_identity(
        resolved,
        config_path,
        Path(__file__).resolve().parents[4],
        sample_limit,
    )
    return resolved


def main() -> None:
    """Validate all datasets, load one model, and run one or every shard."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    configs = [
        load_direct_generation_config(path, project_root) for path in args.config
    ]
    first = configs[0]
    if args.rank is not None and not 0 <= args.rank < first.shard_count:
        raise ValueError("rank is outside the configured shard count")
    for config in configs[1:]:
        if (
            config.model_adapter,
            config.model_path,
            config.device,
            config.shard_count,
            config.max_new_tokens,
            config.seed,
        ) != (
            first.model_adapter,
            first.model_path,
            first.device,
            first.shard_count,
            first.max_new_tokens,
            first.seed,
        ):
            raise ValueError(
                "One direct-generation suite must share model/run settings"
            )
    random.seed(first.seed)
    np.random.seed(first.seed)
    torch.manual_seed(first.seed)
    torch.cuda.manual_seed_all(first.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    stage = "direct_generation"
    runtime = None
    effective_shard_counts = tuple(
        _effective_shard_count(config, args.limit) for config in configs
    )
    max_shard_count = max(effective_shard_counts)
    if args.rank is not None and args.rank >= max_shard_count:
        raise ValueError("rank is outside the effective shard count")
    ranks = range(max_shard_count) if args.rank is None else (args.rank,)
    for rank in ranks:
        pending = []
        for path, config, shard_count in zip(
            args.config, configs, effective_shard_counts, strict=True
        ):
            if rank >= shard_count:
                continue
            directory = config.output_dir / "shards" / f"rank_{rank:03d}"
            launch_fingerprint = experiment_launch_fingerprint(
                path, project_root, args.limit
            )
            if _completed_and_valid(directory, launch_fingerprint):
                print(f"direct generation already complete: {directory}", flush=True)
                continue
            pending.append((path, config, directory, shard_count))
        for path, config, directory, shard_count in pending:
            recorder = RunRecorder(
                directory,
                _resolved(
                    config,
                    path,
                    config.expected_dataset_fingerprint,
                    rank,
                    args.limit,
                    shard_count,
                ),
                project_root,
            )
            try:
                samples, fingerprint = load_direct_generation_samples(config)
                if args.limit is not None:
                    samples = samples[: args.limit]
                if runtime is None:
                    runtime = load_model_runtime(
                        first.model_adapter, first.model_path, first.device
                    )
                configure_image_max_pixels(runtime, config.image_max_pixels)
                recorder.event(
                    stage,
                    "started",
                    sample_count=len(samples),
                    rank=rank,
                    shard_count=shard_count,
                    thinking=config.thinking,
                )
                summary = run_direct_generation_shard(
                    runtime,
                    samples,
                    fingerprint,
                    config,
                    rank,
                    recorder,
                    sample_limit=args.limit,
                    effective_shard_count=shard_count,
                )
                recorder.event(stage, "complete", **summary["overall"])
                recorder.complete(
                    sample_count=summary["shard_sample_count"],
                    dataset_fingerprint=fingerprint,
                )
            except BaseException as error:
                recorder.fail(error, stage)
                raise


if __name__ == "__main__":
    main()
