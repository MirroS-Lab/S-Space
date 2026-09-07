"""Run layer-matched, zero-boundary SpinBench rotation sweeps."""

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

from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.experiments.identity import bind_launch_identity, runtime_identity
from sspace.experiments.spinbench.perspective_taking.adapter import (
    SpinBenchSample,
    load_spinbench,
    open_spinbench_image,
    spinbench_fingerprint,
)
from sspace.experiments.common.pairwise.runner import (
    _extract_prompt_pair_differences,
)
from sspace.experiments.spinbench.perspective_taking.projection import (
    render_spinbench_projection_prompts,
    score_spinbench_projection,
    summarize_spinbench_projection,
)
from sspace.experiments.spinbench.perspective_taking.projection_config import (
    SpinBenchProjectionConfig,
    load_spinbench_projection_config,
)
from sspace.experiments.spinbench.perspective_taking.projection_runner import (
    _write_immutable_jsonl,
)
from sspace.run_records import (
    RunRecorder,
    atomic_json,
    canonical_fingerprint,
    file_sha256,
)
from sspace.core.extraction.execution import LayerwisePostBlockStates
from sspace.core.models.runtime import (
    RUNTIME_SPECS,
    configure_image_max_pixels,
    load_model_runtime,
)
from sspace.core.prompts.templates import AXIS_ORDER


PROTOCOL = "layer_matched_margin_rotation_matrix_v1"


def _extract_layerwise_margins(
    runtime: Any,
    artifact: SSpaceArtifact,
    samples: Sequence[SpinBenchSample],
    config: SpinBenchProjectionConfig,
    output_dir: Path,
    resolved: dict[str, Any],
) -> tuple[np.ndarray, tuple[tuple[Any, Any], ...]]:
    """Save `v[l,g]^T(h_A[l]-h_B[l])` as `[N,2,L]`.

    The two orientations are the original A-minus-B prompt and the exact
    object-swapped prompt. Layer IDs are zero-based post-block IDs, and each
    hidden-state layer is paired only with its matching unit artifact axis.
    """
    layer_ids = artifact.manifest.model.layer_ids
    axes = np.asarray(artifact.axes_tensor, dtype=np.float32)
    if axes.shape != (len(layer_ids), 3, runtime.spec.hidden_size):
        raise ValueError("SpinBench artifact axes have an incompatible shape")
    if layer_ids[-1] >= len(runtime.blocks):
        raise ValueError("SpinBench artifact exports an unsupported model layer")
    prompts = tuple(
        render_spinbench_projection_prompts(sample, config.probe_context)
        for sample in samples
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    margin_path = output_dir / "margins.npy"
    record_path = output_dir / "margin_records.jsonl"
    checkpoint_path = output_dir / "margin_checkpoint.json"
    shape = (len(samples), 2, len(layer_ids))
    fingerprint = canonical_fingerprint(resolved)
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("SpinBench layerwise checkpoint fingerprint mismatch")
        margins = np.load(margin_path, mmap_mode="r+")
        if margins.shape != shape or margins.dtype != np.float32:
            raise ValueError("SpinBench layerwise checkpoint shape changed")
        next_sample = int(checkpoint["next_sample"])
        with record_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
    else:
        if margin_path.exists() or record_path.exists():
            raise FileExistsError("SpinBench readout exists without a checkpoint")
        margins = open_memmap(margin_path, mode="w+", dtype=np.float32, shape=shape)
        margins[:] = np.nan
        margins.flush()
        record_path.touch(exist_ok=False)
        next_sample = 0

    axis_index = {group: index for index, group in enumerate(AXIS_ORDER)}
    axis_tensor = torch.from_numpy(axes)

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

    with record_path.open("a", encoding="utf-8") as records:
        checkpoint(records)
        with LayerwisePostBlockStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                differences, positions, token_pieces = _extract_prompt_pair_differences(
                    runtime,
                    extractor,
                    open_spinbench_image(sample),
                    prompts[next_sample],
                )
                selected = differences[:, list(layer_ids), :]
                group_axes = axis_tensor[:, axis_index[sample.source_group], :]
                values = torch.einsum("bld,ld->bl", selected, group_axes)
                if values.shape != shape[1:] or not torch.isfinite(values).all():
                    raise FloatingPointError("SpinBench layerwise margins are invalid")
                margins[next_sample] = values.numpy()
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "sample_id": sample.sample_id,
                            "premise_mode": sample.premise_mode,
                            "source_group": sample.source_group,
                            "target_view": sample.target_view,
                            "target_property": sample.target_property,
                            "object_a": sample.object_a,
                            "object_b": sample.object_b,
                            "gold_answer": sample.answer,
                            "projection_prompts": [
                                prompt.text for prompt in prompts[next_sample]
                            ],
                            "positions": positions,
                            "token_pieces": token_pieces,
                            "margins": values.tolist(),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                if next_sample % config.checkpoint_every == 0:
                    checkpoint(records)
                    print(
                        f"{config.model_adapter} {config.premise_mode} "
                        f"{next_sample}/{len(samples)}",
                        flush=True,
                    )
        checkpoint(records)
    if not np.isfinite(margins).all():
        raise FloatingPointError("Completed SpinBench margins are non-finite")
    return np.asarray(margins), prompts


def _score_layers(
    samples: Sequence[SpinBenchSample],
    margins: np.ndarray,
    prompts: Sequence[Sequence[Any]],
    layer_ids: Sequence[int],
    evaluation_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Rotate and score every layer without selecting on SpinBench labels."""
    records: list[dict[str, Any]] = []
    metrics = []
    for position, layer_id in enumerate(layer_ids):
        layer_records = score_spinbench_projection(
            samples,
            margins[:, :, position],
            prompts,
            int(layer_id),
            evaluation_id,
        )
        payload = [record.to_dict() for record in layer_records]
        records.extend(payload)
        summary = summarize_spinbench_projection(payload)
        metrics.append(
            {
                "layer_id": int(layer_id),
                "overall": summary["overall"]["margin"],
                "by_source_group": {
                    key: value["margin"]
                    for key, value in summary["by_source_group"].items()
                },
                "by_target_view": {
                    key: value["margin"]
                    for key, value in summary["by_target_view"].items()
                },
                "by_target_property": {
                    key: value["margin"]
                    for key, value in summary["by_target_property"].items()
                },
                "by_transform": {
                    key: value["margin"]
                    for key, value in summary["by_transform"].items()
                },
            }
        )
    _write_immutable_jsonl(output_dir / "projection_records.jsonl", records)
    best_accuracy = max(row["overall"]["accuracy"] for row in metrics)
    best_layers = [
        row["layer_id"]
        for row in metrics
        if row["overall"]["accuracy"] == best_accuracy
    ]
    result = {
        "protocol": PROTOCOL,
        "sample_count": len(samples),
        "layer_ids": [int(layer) for layer in layer_ids],
        "diagnostic_oracle_layers": best_layers,
        "diagnostic_oracle_accuracy": best_accuracy,
        "per_layer": metrics,
    }
    atomic_json(output_dir / "layerwise_metrics.json", result)
    csv_path = output_dir / "layerwise_metrics.csv"
    with csv_path.with_suffix(".csv.tmp").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "layer_id",
                "overall_accuracy",
                "back_accuracy",
                "left_view_accuracy",
                "right_view_accuracy",
                "closer_accuracy",
                "left_property_accuracy",
            ]
        )
        for row in metrics:
            target_views = row["by_target_view"]
            target_properties = row["by_target_property"]
            writer.writerow(
                [
                    row["layer_id"],
                    row["overall"]["accuracy"],
                    target_views.get("back", {}).get("accuracy"),
                    target_views.get("left", {}).get("accuracy"),
                    target_views.get("right", {}).get("accuracy"),
                    target_properties.get("closer", {}).get("accuracy"),
                    target_properties.get("left", {}).get("accuracy"),
                ]
            )
    os.replace(csv_path.with_suffix(".csv.tmp"), csv_path)
    return result


def _run_condition(
    runtime: Any,
    artifact: SSpaceArtifact,
    config: SpinBenchProjectionConfig,
    config_path: Path,
    output_dir: Path,
    project_root: Path,
    limit: int | None = None,
) -> None:
    if file_sha256(config.annotation_path) != config.annotation_sha256:
        raise ValueError("SpinBench annotation checksum changed")
    samples = load_spinbench(
        config.annotation_path, config.image_root, config.premise_mode
    )
    if len(samples) != config.expected_sample_count:
        raise ValueError("SpinBench sample count differs from the frozen config")
    fingerprint = spinbench_fingerprint(samples)
    if fingerprint != config.expected_dataset_fingerprint:
        raise ValueError("SpinBench dataset fingerprint changed")
    if limit is not None:
        samples = samples[:limit]
    resolved = {
        **config.to_dict(),
        "runtime_identity": runtime_identity(config.model_adapter, project_root),
        "config_path": str(config_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "dataset_fingerprint": fingerprint,
        "artifact_id": artifact.manifest.artifact_id,
        "artifact_manifest_sha256": file_sha256(config.artifact_dir / "manifest.json"),
        "protocol": PROTOCOL,
        "layer_ids": list(artifact.manifest.model.layer_ids),
        "limit": limit,
    }
    bind_launch_identity(resolved, config_path, project_root, limit)
    manifest_path = output_dir / "run_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_fingerprint") != canonical_fingerprint(resolved):
            raise ValueError("SpinBench layerwise output belongs to a different run")
        if manifest.get("status") == "complete":
            if manifest.get("sample_count") != len(samples):
                raise ValueError("SpinBench layerwise completed sample count changed")
            print(f"skip complete {config.model_adapter} {config.premise_mode}")
            return
    recorder = RunRecorder(output_dir, resolved, project_root)
    stage = "spinbench_layerwise_projection"
    try:
        recorder.event(
            stage,
            "started",
            sample_count=len(samples),
            premise_mode=config.premise_mode,
        )
        margins, prompts = _extract_layerwise_margins(
            runtime,
            artifact,
            samples,
            config,
            output_dir / "readout",
            resolved,
        )
        result = _score_layers(
            samples,
            margins,
            prompts,
            artifact.manifest.model.layer_ids,
            f"{config.evaluation_id}_layerwise",
            output_dir,
        )
        recorder.event(
            stage,
            "complete",
            diagnostic_oracle_layers=result["diagnostic_oracle_layers"],
            diagnostic_oracle_accuracy=result["diagnostic_oracle_accuracy"],
        )
        recorder.complete(
            sample_count=len(samples),
            dataset_fingerprint=fingerprint,
            layer_count=len(artifact.manifest.model.layer_ids),
            margins_sha256=file_sha256(output_dir / "readout" / "margins.npy"),
            records_sha256=file_sha256(output_dir / "projection_records.jsonl"),
        )
    except BaseException as error:
        recorder.fail(error, stage)
        raise


def main() -> None:
    """Load one model once and run its With/Without Premise sweeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, nargs=2, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[5]
    configs = tuple(
        load_spinbench_projection_config(path, project_root) for path in args.configs
    )
    if (
        len({config.model_adapter for config in configs}) != 1
        or len({config.model_path for config in configs}) != 1
        or len({config.artifact_dir for config in configs}) != 1
        or len({config.device for config in configs}) != 1
        or len({config.seed for config in configs}) != 1
        or len({config.image_max_pixels for config in configs}) != 1
        or {config.premise_mode for config in configs} != {"without", "with"}
    ):
        raise ValueError("Layerwise suite requires one model and both premise modes")
    random.seed(configs[0].seed)
    np.random.seed(configs[0].seed)
    torch.manual_seed(configs[0].seed)
    torch.cuda.manual_seed_all(configs[0].seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    artifact = SSpaceArtifact.load(configs[0].artifact_dir)
    spec = RUNTIME_SPECS[configs[0].model_adapter]
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
    runtime = load_model_runtime(
        configs[0].model_adapter,
        configs[0].model_path,
        configs[0].device,
    )
    configure_image_max_pixels(runtime, configs[0].image_max_pixels)
    for config_path, config in zip(args.configs, configs, strict=True):
        output_dir = args.output_root / config.model_adapter / config.premise_mode
        _run_condition(
            runtime,
            artifact,
            config,
            config_path,
            output_dir,
            project_root,
            args.limit,
        )


if __name__ == "__main__":
    main()
