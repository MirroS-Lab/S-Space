#!/usr/bin/env python3
"""Frozen model registry for the Object Coordinate Lens.

The minimal CLI keeps model loading in a separate worker. Each entry describes
the representation space and the worker must run under the matching isolated
Python environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from sspace.core.artifacts import SSpaceArtifact


RELEASE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = RELEASE_ROOT.parent


FULL_AXIS_ROOT = RELEASE_ROOT / "core" / "artifacts" / "pretrained"
MODEL_ROOT = PROJECT_ROOT / ".cache" / "assets" / "models"
AXIS_ORDER = ("horizontal", "vertical", "distance")
DEFAULT_MODEL_ID = "molmo2_er"


def _snapshot(repo: str, revision: str, hub_cache: Path) -> Path:
    encoded_repo = "models--" + repo.replace("/", "--")
    return Path(hub_cache) / encoded_repo / "snapshots" / revision


@dataclass(frozen=True)
class ModelSpec:
    """Declare one exact model-native coordinate representation space."""

    id: str
    label: str
    repo: str
    revision: str
    checkpoint_path: Path
    axes_path: Path
    metadata_path: Path
    default_layer: int
    hidden: int
    layer_count: int
    runtime_overlay: str | None
    transformers_version: str
    architecture: str
    block_path: str
    attention_implementation: str
    family: str


def _spec(
    *,
    model_id: str,
    label: str,
    repo: str,
    revision: str,
    axis_artifact: str,
    default_layer: int,
    hidden: int,
    layer_count: int,
    runtime_overlay: str | None,
    transformers_version: str,
    architecture: str,
    block_path: str,
    attention_implementation: str,
    family: str,
) -> ModelSpec:
    bundle = FULL_AXIS_ROOT / axis_artifact
    return ModelSpec(
        id=model_id,
        label=label,
        repo=repo,
        revision=revision,
        checkpoint_path=MODEL_ROOT / model_id,
        axes_path=bundle / "tensors.safetensors",
        metadata_path=bundle / "manifest.json",
        default_layer=default_layer,
        hidden=hidden,
        layer_count=layer_count,
        runtime_overlay=runtime_overlay,
        transformers_version=transformers_version,
        architecture=architecture,
        block_path=block_path,
        attention_implementation=attention_implementation,
        family=family,
    )


_MODEL_SPECS = {
    "molmo2_er": _spec(
        model_id="molmo2_er",
        label="Molmo2-ER",
        repo="allenai/Molmo2-ER",
        revision="dab22564403d2607855bb1fffb0721285b445081",
        axis_artifact="molmo2_er_coco6000_final_logit_axes_v2",
        default_layer=21,
        hidden=2560,
        layer_count=36,
        runtime_overlay=None,
        transformers_version="4.57.6",
        architecture="Molmo2ForConditionalGeneration",
        block_path="model.transformer.blocks",
        attention_implementation="sdpa",
        family="molmo",
    ),
    "molmoact2": _spec(
        model_id="molmoact2",
        label="MolmoAct2",
        repo="allenai/MolmoAct2",
        revision="e432d85f6e039edca44afb93c262f3084ab72a9c",
        axis_artifact="molmoact2_coco6000_final_logit_axes_v2",
        default_layer=19,
        hidden=2560,
        layer_count=36,
        runtime_overlay=None,
        transformers_version="5.3.0",
        architecture="MolmoAct2ForConditionalGeneration",
        block_path="model.transformer.blocks",
        attention_implementation="sdpa",
        family="molmo",
    ),
    "molmoact2_pretrain": _spec(
        model_id="molmoact2_pretrain",
        label="MolmoAct2-Pretrain",
        repo="allenai/MolmoAct2-Pretrain",
        revision="a05effca9ba36c1177359b42a9d5d7a4568dbe3c",
        axis_artifact="molmoact2_pretrain_coco6000_final_logit_axes_v2",
        default_layer=19,
        hidden=2560,
        layer_count=36,
        runtime_overlay=None,
        transformers_version="5.3.0",
        architecture="MolmoAct2ForConditionalGeneration",
        block_path="model.transformer.blocks",
        attention_implementation="sdpa",
        family="molmo",
    ),
    "qwen35_4b": _spec(
        model_id="qwen35_4b",
        label="Qwen3.5-4B",
        repo="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        axis_artifact="qwen35_4b_coco6000_final_logit_axes_v2",
        default_layer=15,
        hidden=2560,
        layer_count=32,
        runtime_overlay=None,
        transformers_version="5.9.0",
        architecture="Qwen3_5ForConditionalGeneration",
        block_path="model.language_model.layers",
        attention_implementation="sdpa",
        family="qwen",
    ),
    "qwen36_27b": _spec(
        model_id="qwen36_27b",
        label="Qwen3.6-27B",
        repo="Qwen/Qwen3.6-27B",
        revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        axis_artifact="qwen36_27b_coco6000_final_logit_axes_v2",
        default_layer=43,
        hidden=5120,
        layer_count=64,
        runtime_overlay=None,
        transformers_version="5.9.0",
        architecture="Qwen3_5ForConditionalGeneration",
        block_path="model.language_model.layers",
        attention_implementation="sdpa",
        family="qwen",
    ),
}

MODEL_SPECS: Mapping[str, ModelSpec] = MappingProxyType(_MODEL_SPECS)


def get_model_spec(model_id: str) -> ModelSpec:
    """Return a registered model or raise with the complete allowed set."""

    try:
        return MODEL_SPECS[model_id]
    except KeyError as exc:
        allowed = ", ".join(MODEL_SPECS)
        raise KeyError(
            f"Unknown model {model_id!r}; expected one of: {allowed}"
        ) from exc


def checkpoint_path_for(spec: ModelSpec, hub_cache: Path | None = None) -> Path:
    """Resolve an explicit project binding or the pinned Hub snapshot."""

    override = os.environ.get(f"OBJECT_LENS_MODEL_PATH_{spec.id.upper()}")
    if override:
        return Path(override).expanduser().resolve()
    if hub_cache is None:
        return spec.checkpoint_path
    return _snapshot(spec.repo, spec.revision, Path(hub_cache))


def load_model_axes_by_layer(
    spec: ModelSpec,
) -> tuple[dict[int, Any], dict[int, dict[str, Any]]]:
    """Load and validate every released layer row for one model.

    Layer availability comes exclusively from the artifact manifest.  This is
    intentionally separate from the model's runtime block count because the
    released artifacts omit the final block.
    """

    import numpy as np

    if spec.axes_path.suffix.lower() != ".safetensors":
        raise ValueError(f"Expected a full safetensors axis artifact: {spec.axes_path}")
    if spec.axes_path.name != "tensors.safetensors":
        raise ValueError(f"Unexpected axis tensor filename: {spec.axes_path.name}")

    artifact_dir = spec.axes_path.parent.resolve()
    artifact = SSpaceArtifact.load(artifact_dir)
    raw_metadata = artifact.manifest.to_dict()
    raw_metadata["axis_order"] = list(artifact.manifest.axis_order)
    raw_metadata["positive_endpoints"] = list(artifact.manifest.positive_endpoints)
    raw_metadata["negative_endpoints"] = list(artifact.manifest.negative_endpoints)
    raw_metadata["model"]["layer_ids"] = list(artifact.manifest.model.layer_ids)
    model_metadata = raw_metadata["model"]
    if artifact.manifest.artifact_id != artifact_dir.name:
        raise ValueError(
            f"{spec.id} artifact ID {artifact.manifest.artifact_id!r} "
            f"does not match directory {artifact_dir.name!r}"
        )
    expected_model = {
        "model_id": spec.repo,
        "revision": spec.revision,
        "architecture": spec.architecture,
        "hidden_size": spec.hidden,
        "layer_numbering": "zero_based_post_block",
    }
    for key, expected in expected_model.items():
        if model_metadata.get(key) != expected:
            raise ValueError(
                f"{spec.id} full artifact model {key}={model_metadata.get(key)!r}, "
                f"expected {expected!r}"
            )
    if raw_metadata.get("axis_order") != list(AXIS_ORDER):
        raise ValueError(
            f"{spec.id} axis order {raw_metadata.get('axis_order')!r} != "
            f"{list(AXIS_ORDER)!r}"
        )

    raw_layer_ids = model_metadata.get("layer_ids")
    if not isinstance(raw_layer_ids, list) or not raw_layer_ids:
        raise ValueError(f"{spec.id} full artifact has no layer IDs")
    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in raw_layer_ids
    ):
        raise ValueError(f"{spec.id} full artifact layer IDs must be integers")
    layer_ids = tuple(int(value) for value in raw_layer_ids)
    if len(set(layer_ids)) != len(layer_ids) or tuple(sorted(layer_ids)) != layer_ids:
        raise ValueError(f"{spec.id} full artifact layer IDs must be unique and sorted")
    if any(layer < 0 or layer >= spec.layer_count for layer in layer_ids):
        raise ValueError(f"{spec.id} full artifact has an out-of-range layer ID")
    if spec.default_layer not in layer_ids:
        raise ValueError(
            f"{spec.id} default L{spec.default_layer} has no released axis row"
        )

    all_axes = np.asarray(artifact.axes_tensor)
    expected_shape = (len(layer_ids), 3, spec.hidden)
    if all_axes.shape != expected_shape or all_axes.dtype != np.float32:
        raise ValueError(
            f"{spec.id} full axes {all_axes.shape}/{all_axes.dtype} != "
            f"{expected_shape}/float32"
        )
    if not np.isfinite(all_axes).all():
        raise FloatingPointError(f"{spec.id} full axes contain non-finite values")
    norms = np.linalg.norm(all_axes.astype(np.float64), axis=2)
    if not np.allclose(norms, np.ones_like(norms), rtol=1e-6, atol=1e-6):
        raise ValueError(f"{spec.id} full axes contain non-unit vectors")

    axes_by_layer: dict[int, Any] = {}
    metadata_by_layer: dict[int, dict[str, Any]] = {}
    for row_index, layer in enumerate(layer_ids):
        axes_by_layer[layer] = all_axes[row_index]
        metadata_by_layer[layer] = {
            "schema_version": raw_metadata.get("schema_version"),
            "model_adapter": spec.id,
            "representation_space_id": f"{spec.id}_L{layer}",
            "artifact_id": raw_metadata.get("artifact_id"),
            "selected_layer": layer,
            "artifact_selected_layer": model_metadata.get("selected_layer"),
            "available_layers": list(layer_ids),
            "axis_order": list(AXIS_ORDER),
            "axes_shape": [3, spec.hidden],
            "axes_dtype": "float32",
            "layer_numbering": model_metadata.get("layer_numbering"),
            "model_id": model_metadata.get("model_id"),
            "revision": model_metadata.get("revision"),
            "positive_endpoints": raw_metadata.get("positive_endpoints"),
            "negative_endpoints": raw_metadata.get("negative_endpoints"),
            "selection_scope": "explicit_full_artifact_layer",
        }
    return axes_by_layer, metadata_by_layer


def validate_model_artifacts(spec: ModelSpec) -> dict[str, Any]:
    """Strictly validate one selected representation space against its spec."""

    _axes_by_layer, metadata_by_layer = load_model_axes_by_layer(spec)
    return metadata_by_layer[spec.default_layer]
