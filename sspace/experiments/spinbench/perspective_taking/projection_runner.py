"""Run checkpointed SpinBench S-Space projection."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sspace.core.artifacts import SSpaceArtifact
from sspace.run_records import atomic_json
from sspace.core.models.runtime import LoadedModelRuntime

from .adapter import SpinBenchSample, open_spinbench_image
from sspace.experiments.common.pairwise.runner import (
    _extract_prompt_pair_margins,
)
from .projection import (
    render_spinbench_projection_prompts,
    score_spinbench_projection,
    summarize_spinbench_projection,
)
from .projection_config import SpinBenchProjectionConfig


def _validate_readout_selection(
    config: SpinBenchProjectionConfig,
    artifact: SSpaceArtifact,
) -> None:
    """Validate the declared layer source without consulting SpinBench labels."""
    artifact_layer = artifact.manifest.model.selected_layer
    if config.selected_layer != artifact_layer:
        raise ValueError("SpinBench layer differs from artifact selected_layer")


def _write_immutable_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    encoded = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"Existing {path.name} differs from deterministic scoring")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, path)


def run_spinbench_projection(
    runtime: LoadedModelRuntime,
    artifact: SSpaceArtifact,
    samples: Sequence[SpinBenchSample],
    dataset_fingerprint: str,
    config: SpinBenchProjectionConfig,
    output_dir: Path,
    resolved_config: dict[str, Any],
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Extract, rotate, score, and persist one complete EVAL-09 condition.

    The runner reads only the source relation axis at the declared layer. The
    layer must be either the artifact default or a checksum-traceable external
    validation selection. It delegates all target-view conversion to explicit
    H/V/D rotation matrices and never contains transform-specific answer rules.

    Returns:
        Margin-sign rotation accuracy over the complete official premise subset.

    Raises:
        ValueError: Runtime, artifact, data, prompt, margin, or scoring checks fail.

    Side effects:
        Runs model forwards and writes resumable raw margins, per-row projection
        evidence, and a deterministic summary below ``output_dir``.
    """
    if runtime.spec.key != config.model_adapter:
        raise ValueError("SpinBench runtime differs from the configured adapter")
    _validate_readout_selection(config, artifact)
    expected_count = sample_limit or config.expected_sample_count
    if len(samples) != expected_count or any(
        sample.premise_mode != config.premise_mode for sample in samples
    ):
        raise ValueError("SpinBench samples differ from the configured subset")
    prompt_pairs = tuple(
        render_spinbench_projection_prompts(sample, config.probe_context)
        for sample in samples
    )
    margins, _ = _extract_prompt_pair_margins(
        runtime=runtime,
        artifact=artifact,
        samples=samples,
        prompt_pairs=prompt_pairs,
        layer=config.selected_layer,
        output_dir=output_dir / "readout",
        run_config={
            **resolved_config,
            "stage": "spinbench_projection_readout",
            "dataset_fingerprint": dataset_fingerprint,
        },
        checkpoint_every=config.checkpoint_every,
        image_loader=open_spinbench_image,
    )
    records = score_spinbench_projection(
        samples=samples,
        margins=margins,
        prompt_pairs=prompt_pairs,
        selected_layer=config.selected_layer,
        evaluation_id=config.evaluation_id,
    )
    payload = [record.to_dict() for record in records]
    _write_immutable_jsonl(output_dir / "projection_records.jsonl", payload)
    summary = summarize_spinbench_projection(payload)
    summary.update(
        dataset_fingerprint=dataset_fingerprint,
        artifact_id=artifact.manifest.artifact_id,
        artifact_selected_layer=artifact.manifest.model.selected_layer,
        readout_selection_protocol=config.readout_selection_protocol,
        prompt_style=config.prompt_style,
        probe_context=config.probe_context,
        rotation_mapping="explicit_hvd_proper_rotation_matrix_v1",
    )
    atomic_json(output_dir / "summary.json", summary)
    return summary
