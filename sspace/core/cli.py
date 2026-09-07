"""Run formal S-Space training with one or more workers."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sspace.run_records import (
    atomic_json,
    canonical_fingerprint,
    code_provenance,
    file_sha256,
    failure_is_current,
)
from sspace.determinism import configure_deterministic_cuda

from .config import CPU_THREADS, load_distributed_config
from .data import load_construction_split
from .extraction.shard import extract_worker_shard
from .models.runtime import load_model_runtime
from .pipeline import finalize_distributed_training
from .prompts.templates import AXIS_ORDER


def _source_provenance(project_root: Path) -> dict[str, Any]:
    """Hash every formal Python source so dirty code has an exact identity."""
    return code_provenance(project_root)


def _distributed_identity(expected_world_size: int) -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size:
        raise ValueError(
            f"Runtime WORLD_SIZE={world_size} differs from config {expected_world_size}"
        )
    if not 0 <= rank < world_size or not 0 <= local_rank < torch.cuda.device_count():
        raise ValueError("Distributed rank or CUDA local rank is invalid")
    return rank, local_rank, world_size


def _numerical_runtime(local_rank: int) -> dict[str, Any]:
    """Identify the actual software and accelerator used by this worker."""
    device = torch.cuda.get_device_properties(local_rank)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "safetensors": importlib.metadata.version("safetensors"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": device.name,
        "gpu_compute_capability": [device.major, device.minor],
    }


def _initialize_run(
    output_dir: Path,
    resolved: dict[str, Any],
    rank: int,
) -> None:
    ready_path = output_dir / "run_ready.json"
    attempt_id = os.environ.get("SSPACE_ATTEMPT_ID")
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        fingerprint = canonical_fingerprint(resolved)
        manifest_path = output_dir / "run_manifest.json"
        existing = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        if existing is not None and existing.get("run_fingerprint") != fingerprint:
            raise ValueError("Run directory contains a different distributed config")
        if existing is not None and existing.get("status") == "complete":
            raise FileExistsError("Completed distributed run is immutable")
        atomic_json(output_dir / "resolved_config.json", resolved)
        manifest = {
            "attempt_id": attempt_id,
            "schema_version": "1.0.0",
            "status": "running",
            "run_fingerprint": fingerprint,
            "created_at": (
                existing["created_at"]
                if existing is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            "resumed_at": (
                datetime.now(timezone.utc).isoformat() if existing is not None else None
            ),
            "code": resolved["code_provenance"],
            "config": resolved,
        }
        atomic_json(manifest_path, manifest)
        atomic_json(ready_path, {"run_fingerprint": fingerprint, "attempt_id": attempt_id})
        return
    while True:
        failure = output_dir / "failure.json"
        if failure.exists() and failure_is_current(json.loads(failure.read_text(encoding="utf-8"))):
            raise RuntimeError("Rank zero failed before distributed run initialization")
        if ready_path.exists():
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            if ready.get("attempt_id") == attempt_id:
                break
        time.sleep(1.0)
    if ready.get("run_fingerprint") != canonical_fingerprint(resolved):
        raise ValueError("Rank-zero ready marker has a different run fingerprint")


def _validate_group_coverage(
    samples: list[Any], indices: tuple[int, ...], stage: str
) -> None:
    groups = {samples[index].group for index in indices}
    if groups != set(AXIS_ORDER):
        raise ValueError(f"{stage} selection must cover H/V/D exactly, found {groups}")


def _event(output_dir: Path, stage: str, status: str, **fields: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "status": status,
        **fields,
    }
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_failure(
    output_dir: Path,
    rank: int,
    world_size: int,
    resolved: dict[str, Any],
    error: BaseException,
) -> None:
    failure = {
        "attempt_id": os.environ.get("SSPACE_ATTEMPT_ID"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rank": rank,
        "world_size": world_size,
        "run_fingerprint": canonical_fingerprint(resolved),
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }
    directory = output_dir / "rank_failures"
    directory.mkdir(parents=True, exist_ok=True)
    atomic_json(directory / f"rank_{rank:03d}.json", failure)
    if rank == 0:
        atomic_json(output_dir / "failure.json", failure)
        manifest_path = output_dir / "run_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest.update(status="failed", failure=failure)
            atomic_json(manifest_path, manifest)


def main() -> None:
    """Validate one strict config and execute the assigned worker."""
    # torchrun gives every rank the same fresh rendezvous ID.
    os.environ["SSPACE_ATTEMPT_ID"] = os.environ.get("TORCHELASTIC_RUN_ID") or uuid.uuid4().hex
    configure_deterministic_cuda()
    os.environ["OMP_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    config = load_distributed_config(args.config, project_root)
    rank, local_rank, world_size = _distributed_identity(config.world_size)
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "code_provenance": _source_provenance(project_root),
        "numerical_runtime": _numerical_runtime(local_rank),
    }
    try:
        _initialize_run(config.output_dir, resolved, rank)
        if rank == 0:
            _event(
                config.output_dir, "distributed_run", "started", world_size=world_size
            )
        train_manifest, train_samples = load_construction_split(config.dataset, "train")
        if train_manifest["dataset_fingerprint"] != config.dataset_fingerprint:
            raise ValueError("Validated construction dataset fingerprint differs")
        train_indices = config.train_selection.resolve(len(train_samples))
        _validate_group_coverage(train_samples, train_indices, "Train")
        validation_manifest = None
        validation_samples = None
        validation_indices = None
        if rank == 0:
            validation_manifest, validation_samples = load_construction_split(
                config.validation_dataset, "validation"
            )
            if (
                validation_manifest["dataset_fingerprint"]
                != config.validation_dataset_fingerprint
            ):
                raise ValueError("Validation dataset fingerprint differs")
            validation_indices = config.validation_selection.resolve(
                len(validation_samples)
            )
            _validate_group_coverage(
                validation_samples, validation_indices, "Validation"
            )

        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.use_deterministic_algorithms(True)
        torch.set_num_threads(CPU_THREADS)
        torch.set_num_interop_threads(CPU_THREADS)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "PyTorch CPU thread counts differ from the frozen value 1"
            )
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        sdp_state = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
        )
        if sdp_state != (False, False, False, True):
            raise RuntimeError(f"CUDA SDP kernels differ from math-only: {sdp_state}")
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
        runtime = load_model_runtime(config.model_adapter, config.model_path, device)
        extract_worker_shard(
            runtime,
            train_samples,
            train_indices,
            config.output_dir,
            resolved,
            rank,
            world_size,
            config.checkpoint_every,
        )
        if rank != 0:
            return
        _event(config.output_dir, "rank_zero_shard", "complete")
        assert validation_samples is not None and validation_indices is not None
        _event(config.output_dir, "merge_validation_artifact", "started")
        artifact = finalize_distributed_training(
            config,
            runtime,
            train_manifest,
            validation_manifest,
            train_samples,
            validation_samples,
            train_indices,
            validation_indices,
            resolved,
        )
        top_checksums = {
            "merged/checksums.json": file_sha256(
                config.output_dir / "merged" / "checksums.json"
            ),
            "validation/checksums.json": file_sha256(
                config.output_dir / "validation" / "checksums.json"
            ),
            "validation_metrics.json": file_sha256(
                config.output_dir / "validation_metrics.json"
            ),
            "artifact/checksums.json": file_sha256(
                config.artifact_dir / "checksums.json"
            ),
        }
        atomic_json(config.output_dir / "checksums.json", top_checksums)
        manifest_path = config.output_dir / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            artifact_id=artifact.manifest.artifact_id,
            artifact_dir=str(config.artifact_dir),
            selected_layer=artifact.manifest.model.selected_layer,
        )
        atomic_json(manifest_path, manifest)
        _event(
            config.output_dir,
            "distributed_run",
            "complete",
            artifact_id=artifact.manifest.artifact_id,
            selected_layer=artifact.manifest.model.selected_layer,
        )
    except BaseException as error:
        _record_failure(config.output_dir, rank, world_size, resolved, error)
        raise


if __name__ == "__main__":
    main()
