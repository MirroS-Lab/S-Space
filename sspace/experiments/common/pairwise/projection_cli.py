"""Run a fixed-layer or all-layer pairwise projection benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch

from sspace.determinism import configure_deterministic_cuda
from sspace.experiments.identity import bind_launch_identity
from sspace.run_records import RunRecorder, atomic_text, file_sha256
from sspace.core.models.runtime import (
    configure_image_max_pixels,
    load_model_runtime,
)

from .config import load_bound_artifact, load_evaluation_config
from .layerwise import diagnostic_best_layer, score_layerwise_margins
from .registry import BenchmarkRegistry
from .runner import extract_benchmark_margins, extract_layerwise_benchmark_margins
from .schema import pairwise_dataset_fingerprint
from .scoring import evaluate_pairwise_scores, summarize_pairwise_records


def _write_layerwise_metrics(path: Path, metrics: tuple) -> None:
    """Write one accuracy and prediction-balance row per layer."""
    with atomic_text(path, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layer",
                "overall_accuracy",
                "horizontal_accuracy",
                "vertical_accuracy",
                "distance_accuracy",
                "worst_axis_accuracy",
                "horizontal_positive_rate",
                "vertical_positive_rate",
                "distance_positive_rate",
                "correct",
                "total",
            ]
        )
        for metric in metrics:
            writer.writerow(
                [
                    metric.layer_id,
                    metric.overall_accuracy,
                    *metric.axis_accuracy,
                    metric.worst_axis_accuracy,
                    *metric.positive_prediction_rate,
                    metric.correct,
                    metric.total,
                ]
            )


def _run_fixed_layer(config, artifact, runtime, samples, resolved, recorder) -> None:
    """Run and serialize one explicitly requested layer."""
    layer_id = config.layer_id
    if layer_id is None:
        raise ValueError("Fixed-layer execution requires an integer layer_id")
    if layer_id not in artifact.manifest.model.layer_ids:
        raise ValueError(f"Layer {layer_id} is not present in the artifact")
    margins, prompts = extract_benchmark_margins(
        runtime,
        artifact,
        samples,
        layer_id,
        config.prompt_style,
        config.output_dir,
        resolved,
        config.checkpoint_every,
    )
    records = evaluate_pairwise_scores(
        artifact,
        samples,
        margins[:, 0],
        margins[:, 1],
        prompts,
        layer_id,
    )
    predictions = config.output_dir / "predictions.jsonl"
    with atomic_text(predictions) as stream:
        for record in records:
            stream.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
    summary = summarize_pairwise_records(records)
    summary_path = config.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recorder.complete(
        mode="fixed_layer",
        layer_id=layer_id,
        summary=summary,
        predictions_sha256=file_sha256(predictions),
        summary_sha256=file_sha256(summary_path),
    )


def _run_all_layers(config, artifact, runtime, samples, resolved, recorder) -> None:
    """Run and serialize every layer declared by the artifact."""
    layer_ids = np.asarray(artifact.manifest.model.layer_ids, dtype=np.int64)
    margins = extract_layerwise_benchmark_margins(
        runtime,
        artifact.axes_tensor,
        layer_ids,
        samples,
        config.prompt_style,
        config.output_dir / "readout",
        resolved,
        config.checkpoint_every,
    )
    metrics = score_layerwise_margins(samples, margins[:, 0, :], layer_ids)
    best_layer = diagnostic_best_layer(metrics)
    metrics_csv = config.output_dir / "layerwise_metrics.csv"
    metrics_json = config.output_dir / "layerwise_metrics.json"
    _write_layerwise_metrics(metrics_csv, metrics)
    metrics_json.write_text(
        json.dumps(
            {
                "decision_protocol": config.decision_protocol,
                "benchmark": config.benchmark,
                "sample_count": len(samples),
                "diagnostic_best_layer": best_layer,
                "per_layer": [metric.to_dict() for metric in metrics],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    recorder.complete(
        mode="all_layers",
        layer_ids=layer_ids.tolist(),
        diagnostic_best_layer=best_layer,
        metrics_sha256=file_sha256(metrics_json),
    )


def main() -> None:
    """Run the explicit integer layer or JSON-null all-layer mode."""
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_evaluation_config(args.config, project_root)
    artifact = load_bound_artifact(config)
    samples = BenchmarkRegistry.load(config.benchmark, config.benchmark_source)
    dataset_fingerprint = pairwise_dataset_fingerprint(samples)
    if dataset_fingerprint != config.expected_dataset_fingerprint:
        raise ValueError("Pairwise benchmark fingerprint changed")
    if args.limit is not None:
        samples = samples[: args.limit]
    layer_ids = list(artifact.manifest.model.layer_ids)
    resolved = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "artifact_manifest_sha256": file_sha256(config.artifact_dir / "manifest.json"),
        "evaluation_mode": "all_layers" if config.layer_id is None else "fixed_layer",
        "evaluated_layer_ids": layer_ids
        if config.layer_id is None
        else [config.layer_id],
        "sample_limit": args.limit,
        "dataset_fingerprint": dataset_fingerprint,
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(config.output_dir, resolved, project_root)
    stage = "pairwise_projection"
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
        recorder.event(
            stage,
            "started",
            mode=resolved["evaluation_mode"],
            sample_count=len(samples),
        )
        if config.layer_id is None:
            _run_all_layers(config, artifact, runtime, samples, resolved, recorder)
        else:
            _run_fixed_layer(config, artifact, runtime, samples, resolved, recorder)
    except BaseException as error:
        recorder.fail(error, stage)
        raise


if __name__ == "__main__":
    main()
