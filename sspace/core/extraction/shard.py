"""Execute a deterministic Final-logit Jacobian worker shard."""

from __future__ import annotations

import traceback
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image

from sspace.run_records import atomic_json, canonical_fingerprint

from ..data import ConstructionSample
from ..models.runtime import LoadedModelRuntime
from ..prompts.rendering import render_prompt_pair
from ..prompts.templates import AXIS_ORDER, PROMPT_STYLE_ORDER
from .execution import LayerwiseFixedContrastGradient, move_model_inputs
from .sharding import expand_global_job_indices, shard_sample_indices
from .storage import ShardLayout, ShardStorage, validate_complete_shard


def extract_worker_shard(
    runtime: LoadedModelRuntime,
    samples: Sequence[ConstructionSample],
    selected_sample_indices: Sequence[int],
    output_dir: Path,
    run_config: dict[str, Any],
    rank: int,
    world_size: int,
    checkpoint_every: int,
) -> dict[str, Any]:
    """Extract the sample-modulo shard assigned to one worker (TRN-02D).

    Endpoint labels are neither accepted as a control input nor used in the
    Jacobian calculation. The task group only selects one of the three fixed
    H/V/D final-logit contrasts. Each job keeps all five prompt styles and its
    original/swapped orientations inside the same sample-assigned worker.

    Returns:
        The validated complete worker manifest.

    Raises:
        ValueError: Assignment, tensor, token, or checkpoint invariants fail.
        FloatingPointError: A model result contains non-finite values.

    Side effects:
        Runs model forward/backward passes and writes one resume-safe shard.
    """
    if checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be positive")
    shard_dir = Path(output_dir) / "shards" / f"rank_{rank:03d}"
    run_fingerprint = canonical_fingerprint(run_config)
    if (shard_dir / "manifest.json").exists():
        return validate_complete_shard(shard_dir, run_fingerprint)

    local_samples = shard_sample_indices(selected_sample_indices, rank, world_size)
    if local_samples.size == 0:
        raise ValueError(f"Worker rank {rank} has no samples")
    if int(local_samples[-1]) >= len(samples):
        raise ValueError("Worker sample index exceeds the construction split")
    global_jobs = expand_global_job_indices(local_samples, len(PROMPT_STYLE_ORDER))
    storage = ShardStorage(
        shard_dir,
        run_config,
        rank,
        world_size,
        local_samples,
        global_jobs,
        ShardLayout(
            len(global_jobs), runtime.spec.layer_count, runtime.spec.hidden_size
        ),
    )
    jobs = [
        (int(sample_index), style_index, PROMPT_STYLE_ORDER[style_index])
        for sample_index in local_samples
        for style_index in range(len(PROMPT_STYLE_ORDER))
    ]
    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    try:
        with LayerwiseFixedContrastGradient(runtime.model, runtime.blocks) as extractor:
            while storage.next_job < len(jobs):
                local_job = storage.next_job
                sample_index, style_index, style = jobs[local_job]
                sample = samples[sample_index]
                if sample.group not in axis_index:
                    raise ValueError(f"Unknown construction group {sample.group!r}")
                prompts = render_prompt_pair(
                    style,
                    sample.group,
                    sample.query,
                    sample.reference,
                )
                with Image.open(sample.image_path) as source:
                    image = source.convert("RGB")
                prepared_values = []
                result_values = []
                batch_size = runtime.spec.orientation_batch_size
                for start in range(0, len(prompts), batch_size):
                    prepared = runtime.prepare(
                        runtime.processor, image, prompts[start : start + batch_size]
                    )
                    prepared_values.append(prepared)
                    result_values.append(
                        extractor.extract(
                            move_model_inputs(prepared.inputs, runtime.model),
                            prepared.positions,
                            runtime.endpoint_token_ids,
                            sample.group,
                        )
                    )
                contrasts, gradients, states = (
                    torch.cat([value[index] for value in result_values], dim=0)
                    for index in range(3)
                )
                positions = tuple(
                    position
                    for prepared in prepared_values
                    for position in prepared.positions
                )
                token_pieces = tuple(
                    piece
                    for prepared in prepared_values
                    for piece in prepared.token_pieces
                )
                global_job = sample_index * len(PROMPT_STYLE_ORDER) + style_index
                gradient_l2 = np.linalg.norm(
                    gradients.numpy().astype(np.float64), axis=-1
                )
                state_l2 = np.linalg.norm(states.numpy().astype(np.float64), axis=-1)
                records = [
                    {
                        "local_job_index": local_job,
                        "global_job_index": global_job,
                        "global_sample_index": sample_index,
                        "sample_id": sample.sample_id,
                        "split": sample.split,
                        "style": style,
                        "group": sample.group,
                        "orientation": orientation,
                        "orientation_index": orientation_index,
                        "prompt": prompts[orientation_index].text,
                        "positions": positions[orientation_index],
                        "token_pieces": token_pieces[orientation_index],
                        "logit_contrasts": contrasts[orientation_index].tolist(),
                        "role_gradient_l2": gradient_l2[orientation_index].tolist(),
                        "object_state_l2": state_l2[orientation_index].tolist(),
                    }
                    for orientation_index, orientation in enumerate(
                        ("original", "swapped")
                    )
                ]
                storage.append(
                    states.numpy(),
                    gradients.numpy(),
                    contrasts.numpy(),
                    axis_index[sample.group],
                    records,
                )
                if storage.next_job % checkpoint_every == 0:
                    storage.checkpoint()
                    print(
                        f"rank {rank} extraction jobs {storage.next_job}/{len(jobs)} "
                        f"last_global_job={global_job}",
                        flush=True,
                    )
        return storage.complete()
    except BaseException as error:
        failure = {
            "attempt_id": os.environ.get("SSPACE_ATTEMPT_ID"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "rank": rank,
            "world_size": world_size,
            "run_fingerprint": run_fingerprint,
            "next_job": storage.next_job,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_json(shard_dir / "failure.json", failure)
        raise
