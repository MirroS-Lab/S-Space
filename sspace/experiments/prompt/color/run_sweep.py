"""Run the exploratory all-layer non-spatial color-prompt control."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.common.pairwise.config import (
    PairwiseEvaluationConfig,
    load_bound_artifact,
    load_evaluation_config,
)
from sspace.experiments.identity import (
    bind_launch_identity,
    experiment_launch_fingerprint,
)
from sspace.experiments.common.pairwise.decision import classify_pairwise_margin
from sspace.experiments.common.pairwise.layerwise import (
    diagnostic_best_layer,
    score_layerwise_margins,
)
from sspace.experiments.prompt.color.scoring import (
    COLOR_CONTROL_PROMPT_PROTOCOL,
    query_mentioned_first,
    render_color_control_prompt,
)
from sspace.experiments.common.pairwise.registry import BenchmarkRegistry
from sspace.experiments.common.pairwise.schema import (
    PairwiseEvaluationSample,
    open_pairwise_image,
    pairwise_dataset_fingerprint,
)
from sspace.experiments.prompt.config import (
    completed_prompt_output,
    load_prompt_sweep_config,
)
from sspace.experiments.prompt.sampling import balanced_prompt_subset
from sspace.run_records import (
    RunRecorder,
    atomic_text,
    atomic_json,
    canonical_fingerprint,
    file_sha256,
)
from sspace.core.artifacts import SSpaceArtifact
from sspace.core.extraction.execution import (
    LayerwisePostBlockStates,
    move_model_inputs,
)
from sspace.core.models.runtime import (
    LoadedModelRuntime,
    configure_image_max_pixels,
    load_model_runtime,
)
from sspace.core.prompts.templates import AXIS_ORDER


def _run_config(
    config: PairwiseEvaluationConfig,
    layer_ids: np.ndarray,
    source_config: Path,
    artifact: SSpaceArtifact,
    output_dir: Path,
) -> dict[str, Any]:
    """Bind one analysis run to its immutable model, axes, and dataset inputs."""
    return {
        "analysis": "nonspatial_color_layer_sweep_v1",
        "prompt_protocol": COLOR_CONTROL_PROMPT_PROTOCOL,
        "source_config": str(source_config.resolve()),
        "source_config_sha256": file_sha256(source_config),
        "artifact_dir": str(config.artifact_dir),
        "artifact_manifest_sha256": file_sha256(config.artifact_dir / "manifest.json"),
        "artifact_id": artifact.manifest.artifact_id,
        "selected_layer": artifact.manifest.model.selected_layer,
        "model_adapter": config.model_adapter,
        "model_path": str(config.model_path),
        "benchmark": config.benchmark,
        "benchmark_source": str(config.benchmark_source),
        "output_dir": str(output_dir),
        "layer_ids": layer_ids.tolist(),
        "image_max_pixels": config.image_max_pixels,
        "device": config.device,
        "checkpoint_every": config.checkpoint_every,
        "seed": config.seed,
        "decision_protocol": "margin_sign_v1",
    }


def _extract_all_axis_margins(
    runtime: LoadedModelRuntime,
    axes: np.ndarray,
    layer_ids: np.ndarray,
    samples: Sequence[PairwiseEvaluationSample],
    output_dir: Path,
    run_config: dict[str, Any],
    checkpoint_every: int,
    seed: int,
) -> np.ndarray:
    """Extract color-prompt margins ``[N,L,3]`` with one forward per row.

    For sample ``i``, layer ``l``, and axis ``g``, this computes
    ``m[i,l,g] = v[l,g]^T(h_query[i,l] - h_reference[i,l])``. The prompt asks
    only for object colors. Mention order is counterbalanced by sample ID and
    never changes the identity-aligned query-minus-reference subtraction.
    """
    axis_values = np.asarray(axes, dtype=np.float32)
    layers = tuple(int(value) for value in layer_ids)
    if axis_values.shape != (len(layers), 3, runtime.spec.hidden_size):
        raise ValueError("Color-control axes must have shape [L,3,D]")
    if not np.isfinite(axis_values).all():
        raise FloatingPointError("Color-control axes must be finite")
    for sample in samples:
        sample.validate()

    output_dir.mkdir(parents=True, exist_ok=True)
    margin_path = output_dir / "all_axis_margins.npy"
    records_path = output_dir / "margin_records.jsonl"
    checkpoint_path = output_dir / "margin_checkpoint.json"
    shape = (len(samples), len(layers), len(AXIS_ORDER))
    fingerprint = canonical_fingerprint(run_config)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Color-control checkpoint fingerprint mismatch")
        margins = np.load(margin_path, mmap_mode="r+")
        if margins.shape != shape or margins.dtype != np.float32:
            raise ValueError("Color-control checkpoint array is incompatible")
        next_sample = int(checkpoint["next_sample"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
    else:
        if margin_path.exists() or records_path.exists():
            raise FileExistsError("Color-control outputs exist without a checkpoint")
        margins = open_memmap(margin_path, mode="w+", dtype=np.float32, shape=shape)
        margins[:] = np.nan
        margins.flush()
        records_path.touch(exist_ok=False)
        next_sample = 0

    axis_tensor = torch.from_numpy(axis_values)

    def checkpoint(stream: Any) -> None:
        margins.flush()
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample": next_sample,
                "records_offset": stream.tell(),
                "margin_shape": list(shape),
            },
        )

    with records_path.open("a", encoding="utf-8") as records:
        checkpoint(records)
        with LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                query_first = query_mentioned_first(sample.sample_id, seed)
                prompt = render_color_control_prompt(
                    sample.query, sample.reference, query_first=query_first
                )
                prepared = runtime.prepare_objects(
                    runtime.processor, open_pairwise_image(sample), [prompt]
                )
                differences = extractor.extract(
                    move_model_inputs(prepared.inputs, runtime.model),
                    prepared.positions,
                )[0, list(layers), :]
                values = torch.einsum("ld,lgd->lg", differences, axis_tensor)
                if values.shape != (len(layers), 3) or not torch.isfinite(values).all():
                    raise FloatingPointError("Color-control projection is invalid")
                margins[next_sample] = values.numpy()
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "sample_id": sample.sample_id,
                            "group": sample.group,
                            "query": sample.query,
                            "reference": sample.reference,
                            "gold_endpoint": sample.gold_endpoint,
                            "query_mentioned_first": query_first,
                            "prompt": prompt.text,
                            "positions": prepared.positions[0],
                            "token_pieces": prepared.token_pieces[0],
                            "all_axis_margins": values.tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                if next_sample % checkpoint_every == 0:
                    checkpoint(records)
                    print(
                        f"{run_config['benchmark']} {next_sample}/{len(samples)}",
                        flush=True,
                    )
        checkpoint(records)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Completed color-control margins are non-finite")
    return np.asarray(margins)


def _metric_rows(
    samples: Sequence[PairwiseEvaluationSample],
    all_axis_margins: np.ndarray,
    layer_ids: np.ndarray,
) -> list[dict[str, Any]]:
    """Return overall, axis, endpoint, and class-balance metrics per layer."""
    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    group_margins = np.asarray(
        [
            all_axis_margins[row, :, axis_index[sample.group]]
            for row, sample in enumerate(samples)
        ]
    )
    base = score_layerwise_margins(samples, group_margins, layer_ids)
    rows = []
    for layer_position, metric in enumerate(base):
        predictions = []
        for row, sample in enumerate(samples):
            predictions.append(
                classify_pairwise_margin(
                    sample.group, group_margins[row, layer_position]
                )
            )
        endpoint_accuracy = {}
        for endpoint in ("left", "right", "above", "below", "far", "close"):
            selected = [
                index
                for index, sample in enumerate(samples)
                if sample.gold_endpoint == endpoint
            ]
            if not selected:
                raise ValueError(f"Benchmark has no {endpoint} samples")
            endpoint_accuracy[endpoint] = float(
                np.mean([predictions[index] == endpoint for index in selected])
            )
        row = metric.to_dict()
        row["endpoint_accuracy"] = endpoint_accuracy
        rows.append(row)
    return rows


def _write_metrics(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write one readable CSV plus the complete structured metrics JSON."""
    with atomic_text(output_dir / "layerwise_metrics.csv", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layer",
                "overall",
                "horizontal",
                "vertical",
                "distance",
                "worst_axis",
                "left",
                "right",
                "above",
                "below",
                "far",
                "close",
            ]
        )
        for row in rows:
            endpoints = row["endpoint_accuracy"]
            writer.writerow(
                [
                    row["layer_id"],
                    row["overall_accuracy"],
                    *row["axis_accuracy"],
                    row["worst_axis_accuracy"],
                    *(
                        endpoints[name]
                        for name in (
                            "left",
                            "right",
                            "above",
                            "below",
                            "far",
                            "close",
                        )
                    ),
                ]
            )
    atomic_json(output_dir / "layerwise_metrics.json", {"per_layer": rows})


def _load_source_config(path: Path, project_root: Path) -> PairwiseEvaluationConfig:
    """Load one formal benchmark config for the color-prompt control."""
    return load_evaluation_config(path, project_root)


def _load_configs(
    paths: Sequence[Path], project_root: Path
) -> list[PairwiseEvaluationConfig]:
    configs = [_load_source_config(path, project_root) for path in paths]
    identity = {
        (config.model_adapter, config.model_path, config.artifact_dir, config.device)
        for config in configs
    }
    if len(identity) != 1:
        raise ValueError("One process can only reuse one model, artifact, and device")
    return configs


def main() -> None:
    """Run one model across one or more benchmark color controls."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    launch = load_prompt_sweep_config(args.config, project_root, "prompt_color")
    configs = _load_configs(launch.source_configs, project_root)
    if configs[0].model_adapter != launch.model_adapter:
        raise ValueError("Prompt config model differs from source configs")
    artifact = load_bound_artifact(configs[0])
    axes = artifact.axes_tensor
    layer_ids = np.asarray(artifact.manifest.model.layer_ids, dtype=np.int64)

    random.seed(configs[0].seed)
    np.random.seed(configs[0].seed)
    torch.manual_seed(configs[0].seed)
    torch.cuda.manual_seed_all(configs[0].seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = load_model_runtime(
        configs[0].model_adapter, configs[0].model_path, configs[0].device
    )

    for source_path, config in zip(launch.source_configs, configs, strict=True):
        configure_image_max_pixels(runtime, config.image_max_pixels)
        samples = BenchmarkRegistry.load(config.benchmark, config.benchmark_source)
        if pairwise_dataset_fingerprint(samples) != config.expected_dataset_fingerprint:
            raise ValueError("Prompt source dataset fingerprint changed")
        samples = balanced_prompt_subset(samples, args.limit)
        output_dir = launch.output_dir.resolve() / config.benchmark
        launch_fingerprint = experiment_launch_fingerprint(
            args.config, project_root, args.limit
        )
        if completed_prompt_output(output_dir, launch_fingerprint):
            print(f"skip completed prompt output: {output_dir}", flush=True)
            continue
        resolved = _run_config(config, layer_ids, source_path, artifact, output_dir)
        resolved["sample_limit"] = args.limit
        bind_launch_identity(resolved, args.config, project_root, args.limit)
        recorder = RunRecorder(output_dir, resolved, project_root)
        try:
            margins = _extract_all_axis_margins(
                runtime,
                axes,
                layer_ids,
                samples,
                output_dir,
                resolved,
                config.checkpoint_every,
                config.seed,
            )
            rows = _metric_rows(samples, margins, layer_ids)
            _write_metrics(output_dir, rows)
            best = diagnostic_best_layer(
                score_layerwise_margins(
                    samples,
                    np.asarray(
                        [
                            margins[row, :, AXIS_ORDER.index(sample.group)]
                            for row, sample in enumerate(samples)
                        ]
                    ),
                    layer_ids,
                )
            )
            recorder.complete(
                sample_count=len(samples),
                diagnostic_best_layer=best,
                metrics_sha256=file_sha256(output_dir / "layerwise_metrics.json"),
            )
            print(f"complete {config.benchmark}: best L{best}", flush=True)
        except BaseException as error:
            recorder.fail(error, "nonspatial_color_layer_sweep")
            raise


if __name__ == "__main__":
    main()
