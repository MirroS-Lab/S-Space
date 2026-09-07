"""Extract coordinates for the InstructPart action-supervision experiment."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from PIL import Image

from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256
from sspace.experiments.identity import runtime_identity
from sspace.determinism import configure_deterministic_cuda
from sspace.core.artifacts.artifacts import ModelSpec, SSpaceArtifact
from sspace.core.extraction.execution import (
    LayerwiseNamedObjectStates,
    move_model_inputs,
)
from sspace.core.models.runtime import RUNTIME_SPECS, load_model_runtime
from sspace.core.prompts.rendering import RenderedObjectPrompt

from .config import DEFAULT_CONFIG, load_config, project_path
from .prompts import (
    ACTION_TEMPLATES,
    PROMPT_PROTOCOL,
    PROMPT_TEMPLATES,
    TARGET_ORDER,
    TARGET_ROLE,
    render_prompt_ensemble,
)


def _part_centroid(mask_path: Path) -> tuple[float, float]:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 127
    rows, columns = np.nonzero(mask)
    if not len(columns):
        raise ValueError(f"Part mask is empty: {mask_path}")
    height, width = mask.shape
    return float(columns.mean() / width), float(rows.mean() / height)


def _validate_artifact(adapter: str, artifact: SSpaceArtifact) -> None:
    spec = RUNTIME_SPECS[adapter]
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


def _load_samples(selection_path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Prompt-ensemble selection must be a non-empty list")
    ids = [int(row["showcase_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Prompt-ensemble showcase IDs must be unique")
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    for row in rows:
        for field in ("object", "part", "action", "image", "mask"):
            if not str(row[field]).strip():
                raise ValueError(f"Empty {field} for showcase {row['showcase_id']}")
        render_prompt_ensemble(row["action"], row["object"], row["part"])
    return rows


def _validate_inputs(samples: list[dict[str, Any]], assets_dir: Path) -> None:
    for sample in samples:
        image_path = assets_dir / sample["image"]
        mask_path = assets_dir / sample["mask"]
        if not image_path.is_file() or not mask_path.is_file():
            raise FileNotFoundError(
                f"Missing image or mask for showcase {sample['showcase_id']}"
            )
        _part_centroid(mask_path)


def _prompt_evidence(
    prompts: Sequence[RenderedObjectPrompt],
    positions: Sequence[dict[str, int]],
    pieces: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "prompt": prompt.text,
            "position": int(position[TARGET_ROLE]),
            "token_piece": piece[TARGET_ROLE],
        }
        for prompt, position, piece in zip(prompts, positions, pieces, strict=True)
    ]


def _extract_prompt_pair(
    runtime: Any,
    extractor: LayerwiseNamedObjectStates,
    image: Image.Image,
    prompts: tuple[RenderedObjectPrompt, RenderedObjectPrompt],
    layer_ids: tuple[int, ...],
    axes: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    coordinates: list[np.ndarray] = []
    evidence: list[dict[str, Any]] = []
    batch_size = int(runtime.spec.orientation_batch_size)
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        prepared = runtime.prepare_objects(runtime.processor, image, batch)
        states = extractor.extract(
            move_model_inputs(prepared.inputs, runtime.model),
            prepared.positions,
            (TARGET_ROLE,),
        )
        expected = (len(batch), len(runtime.blocks), 1, runtime.spec.hidden_size)
        if tuple(states.shape) != expected:
            raise ValueError(f"Target states {tuple(states.shape)} != {expected}")
        selected_states = states[:, list(layer_ids), 0].numpy()
        projected = np.einsum("bld,lad->bla", selected_states, axes)
        if projected.shape != (len(batch), len(layer_ids), 3):
            raise ValueError(f"Projected coordinates have shape {projected.shape}")
        coordinates.extend(projected[index] for index in range(len(batch)))
        evidence.extend(
            _prompt_evidence(batch, prepared.positions, prepared.token_pieces)
        )
    values = np.stack(coordinates).astype(np.float32, copy=False)
    if values.shape != (2, len(layer_ids), 3) or not np.isfinite(values).all():
        raise FloatingPointError("Prompt-pair coordinates are invalid")
    return values, evidence


def _resolved_config(
    config_path: Path,
    adapter: str,
    model_path: Path,
    artifact_dir: Path,
    artifact: SSpaceArtifact,
    selection_path: Path,
    sample_count: int,
    limit: int | None,
    device: str,
) -> dict[str, Any]:
    return {
        "protocol": PROMPT_PROTOCOL,
        "runtime_identity": runtime_identity(adapter, Path(__file__).resolve().parents[4]),
        "source_config": str(config_path),
        "source_config_sha256": file_sha256(config_path),
        "adapter": adapter,
        "model_path": str(model_path),
        "artifact_dir": str(artifact_dir),
        "artifact_manifest_sha256": file_sha256(artifact_dir / "manifest.json"),
        "artifact_tensors_sha256": file_sha256(artifact_dir / "tensors.safetensors"),
        "artifact_id": artifact.manifest.artifact_id,
        "selection_sha256": file_sha256(selection_path),
        "sample_limit": limit,
        "sample_count": sample_count,
        "templates": [
            {
                "template_id": template.template_id,
                "group": template.group,
                "text": template.text,
            }
            for template in PROMPT_TEMPLATES
        ],
        "target_order": list(TARGET_ORDER),
        "layer_ids": list(artifact.manifest.model.layer_ids),
        "axis_order": ["horizontal", "vertical", "distance"],
        "positive_endpoints": ["right", "below", "close"],
        "action_templates": ACTION_TEMPLATES,
        "instruction_source": "dataset action rendered with dataset object name",
        "forward_protocol": "independent object and part prompts on the same image",
        "margin": "axis @ (part_target_state - object_target_state)",
        "prompt_batch_size": RUNTIME_SPECS[adapter].orientation_batch_size,
        "device": device,
    }


def main() -> None:
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--adapter", choices=tuple(RUNTIME_SPECS), required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.adapter not in config["evaluation"]["models"]:
        raise ValueError(f"Adapter {args.adapter} is not in the experiment config")
    dataset_root = project_path(config["dataset"]["dataset_root"])
    selection_path = dataset_root / "pseudo_ground_truth/retained_selection.json"
    assets_dir = dataset_root / "source/assets"
    samples = _load_samples(selection_path, args.limit)
    _validate_inputs(samples, assets_dir)
    model_config = config["evaluation"]["models"][args.adapter]
    model_path = project_path(model_config["model_path"])
    artifact_dir = project_path(model_config["artifact_dir"])
    artifact = SSpaceArtifact.load(artifact_dir)
    _validate_artifact(args.adapter, artifact)
    if args.validate_only:
        print(
            f"validated {len(samples)} samples and {len(PROMPT_TEMPLATES)} "
            f"templates for {args.adapter}",
            flush=True,
        )
        return
    device = args.device or config["evaluation"]["device"]
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else project_path(config["evaluation"]["output_root"])
    )
    output_dir = output_root / args.adapter
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved = _resolved_config(
        config_path,
        args.adapter,
        model_path,
        artifact_dir,
        artifact,
        selection_path,
        len(samples),
        args.limit,
        device,
    )
    resolved_path = output_dir / "resolved_config.json"
    encoded = json.dumps(resolved, indent=2, sort_keys=True) + "\n"
    if resolved_path.exists() and resolved_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("Output directory contains a different resolved config")
    if not resolved_path.exists():
        atomic_json(resolved_path, resolved)
    fingerprint = canonical_fingerprint(resolved)
    layer_ids = artifact.manifest.model.layer_ids
    coordinate_shape = (
        len(samples),
        len(PROMPT_TEMPLATES),
        len(TARGET_ORDER),
        len(layer_ids),
        3,
    )
    coordinates_path = output_dir / "coordinates.npy"
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Prompt-ensemble checkpoint fingerprint differs")
        coordinates = np.load(coordinates_path, mmap_mode="r+")
        if coordinates.shape != coordinate_shape or coordinates.dtype != np.float32:
            raise ValueError("Checkpoint coordinate array is incompatible")
        next_sample = int(checkpoint["next_sample"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint["records_offset"]))
    else:
        if coordinates_path.exists() or records_path.exists():
            raise FileExistsError("Outputs exist without a checkpoint")
        coordinates = open_memmap(
            coordinates_path, mode="w+", dtype=np.float32, shape=coordinate_shape
        )
        coordinates[:] = np.nan
        coordinates.flush()
        records_path.touch(exist_ok=False)
        next_sample = 0
        atomic_json(checkpoint_path, {
            "fingerprint": fingerprint,
            "next_sample": 0,
            "records_offset": 0,
            "coordinate_shape": list(coordinate_shape),
        })
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    runtime = load_model_runtime(args.adapter, model_path, device)
    axes = artifact.axes_tensor.astype(np.float32, copy=False)
    started = time.monotonic()
    with records_path.open("a", encoding="utf-8") as records:
        with LayerwiseNamedObjectStates(runtime.model, runtime.blocks) as extractor:
            while next_sample < len(samples):
                sample = samples[next_sample]
                image_path = assets_dir / sample["image"]
                mask_path = assets_dir / sample["mask"]
                image = Image.open(image_path).convert("RGB")
                evidence = []
                prompt_pairs = render_prompt_ensemble(
                    sample["action"], sample["object"], sample["part"]
                )
                for template_index, (template, pair) in enumerate(
                    zip(PROMPT_TEMPLATES, prompt_pairs, strict=True)
                ):
                    values, pair_evidence = _extract_prompt_pair(
                        runtime, extractor, image, pair, layer_ids, axes
                    )
                    coordinates[next_sample, template_index] = values
                    evidence.append(
                        {
                            "template_id": template.template_id,
                            "group": template.group,
                            "targets": dict(
                                zip(TARGET_ORDER, pair_evidence, strict=True)
                            ),
                        }
                    )
                records.write(
                    json.dumps(
                        {
                            "sample_index": next_sample,
                            "showcase_id": sample["showcase_id"],
                            "mirror_question_id": sample["mirror_question_id"],
                            "object": sample["object"],
                            "part": sample["part"],
                            "action": sample["action"],
                            "image_sha256": file_sha256(image_path),
                            "part_mask_sha256": file_sha256(mask_path),
                            "part_centroid_xy_normalized": _part_centroid(mask_path),
                            "template_evidence": evidence,
                            "coordinate_index": next_sample,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                next_sample += 1
                coordinates.flush()
                records.flush()
                atomic_json(
                    checkpoint_path,
                    {
                        "fingerprint": fingerprint,
                        "next_sample": next_sample,
                        "records_offset": records.tell(),
                        "coordinate_shape": list(coordinate_shape),
                    },
                )
                print(
                    f"{args.adapter} sample {next_sample}/{len(samples)} "
                    f"elapsed_seconds={time.monotonic() - started:.1f}",
                    flush=True,
                )
    values = np.asarray(coordinates)
    if values.shape != coordinate_shape or not np.isfinite(values).all():
        raise FloatingPointError("Completed prompt-ensemble coordinates are invalid")
    margin_path = output_dir / "template_part_minus_object.npy"
    np.save(margin_path, values[:, :, 1] - values[:, :, 0])
    atomic_json(
        output_dir / "summary.json",
        {
            "status": "complete",
            "sample_count": len(samples),
            "template_count": len(PROMPT_TEMPLATES),
            "coordinate_shape": list(values.shape),
            "margin_shape": list(np.load(margin_path, mmap_mode="r").shape),
            "coordinates_sha256": file_sha256(coordinates_path),
            "margins_sha256": file_sha256(margin_path),
        },
    )
    print(f"complete {args.adapter}: {len(samples)} samples", flush=True)


if __name__ == "__main__":
    main()
