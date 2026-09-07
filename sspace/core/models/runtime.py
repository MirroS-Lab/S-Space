"""Select an explicit model runtime for formal training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch
from PIL import Image

from .molmo import (
    MOLMO_MODEL_SPECS,
    PreparedPrompt,
    answer_token_ids as molmo_answer_token_ids,
    load_molmo_model,
    prepare_direct_generation as prepare_molmo_generation,
    prepare_multi_image_object_prompt as prepare_molmo_multi_image,
    prepare_object_prompt_batch as prepare_molmo_object_batch,
    prepare_prompt_batch as prepare_molmo_batch,
)
from ..prompts.rendering import (
    RenderedMultiImagePrompt,
    RenderedObjectPrompt,
    RenderedPrompt,
)
from .qwen import (
    QWEN35_4B_SPEC,
    QWEN36_SPEC,
    QWEN_MODEL_SPECS,
    answer_token_ids as qwen_answer_token_ids,
    _load_qwen_model,
    prepare_direct_generation as prepare_qwen_generation,
    prepare_multi_image_object_prompt as prepare_qwen_multi_image,
    prepare_object_prompt_batch as prepare_qwen_object_batch,
    prepare_prompt_batch as prepare_qwen_batch,
)


@dataclass(frozen=True)
class RuntimeSpec:
    """Describe one model-native text space accepted by the formal pipeline."""

    key: str
    model_id: str
    revision: str
    transformers_version: str
    architecture: str
    layer_count: int
    hidden_size: int
    dtype: torch.dtype
    attention_backend: str
    sdp_kernel: str
    orientation_batch_size: int


RUNTIME_SPECS = {
    "molmo2_er": RuntimeSpec(
        "molmo2_er",
        MOLMO_MODEL_SPECS["molmo2_er"].model_id,
        MOLMO_MODEL_SPECS["molmo2_er"].revision,
        MOLMO_MODEL_SPECS["molmo2_er"].transformers_version,
        "Molmo2ForConditionalGeneration",
        36,
        2560,
        torch.float32,
        "sdpa",
        "math",
        MOLMO_MODEL_SPECS["molmo2_er"].orientation_batch_size,
    ),
    "molmoact2": RuntimeSpec(
        "molmoact2",
        MOLMO_MODEL_SPECS["molmoact2"].model_id,
        MOLMO_MODEL_SPECS["molmoact2"].revision,
        MOLMO_MODEL_SPECS["molmoact2"].transformers_version,
        "MolmoAct2ForConditionalGeneration",
        36,
        2560,
        torch.float32,
        "sdpa",
        "math",
        MOLMO_MODEL_SPECS["molmoact2"].orientation_batch_size,
    ),
    "molmoact2_pretrain": RuntimeSpec(
        "molmoact2_pretrain",
        MOLMO_MODEL_SPECS["molmoact2_pretrain"].model_id,
        MOLMO_MODEL_SPECS["molmoact2_pretrain"].revision,
        MOLMO_MODEL_SPECS["molmoact2_pretrain"].transformers_version,
        "MolmoAct2ForConditionalGeneration",
        36,
        2560,
        torch.float32,
        "sdpa",
        "math",
        MOLMO_MODEL_SPECS["molmoact2_pretrain"].orientation_batch_size,
    ),
    "qwen36_27b": RuntimeSpec(
        QWEN36_SPEC.key,
        QWEN36_SPEC.model_id,
        QWEN36_SPEC.revision,
        QWEN36_SPEC.transformers_version,
        QWEN36_SPEC.architecture,
        QWEN36_SPEC.layer_count,
        QWEN36_SPEC.hidden_size,
        torch.bfloat16,
        "sdpa",
        "math",
        1,
    ),
    "qwen35_4b": RuntimeSpec(
        QWEN35_4B_SPEC.key,
        QWEN35_4B_SPEC.model_id,
        QWEN35_4B_SPEC.revision,
        QWEN35_4B_SPEC.transformers_version,
        QWEN35_4B_SPEC.architecture,
        QWEN35_4B_SPEC.layer_count,
        QWEN35_4B_SPEC.hidden_size,
        torch.bfloat16,
        "sdpa",
        "math",
        1,
    ),
}


@dataclass
class LoadedModelRuntime:
    """Bundle one loaded model with its exact formal adapter operations."""

    spec: RuntimeSpec
    processor: Any
    model: torch.nn.Module
    blocks: tuple[torch.nn.Module, ...]
    endpoint_token_ids: dict[str, tuple[int, int]]
    prepare: Callable[[Any, Image.Image, Sequence[RenderedPrompt]], PreparedPrompt]
    prepare_objects: Callable[
        [Any, Image.Image, Sequence[RenderedObjectPrompt]], PreparedPrompt
    ]
    prepare_multi_image_objects: Callable[
        [Any, Sequence[Image.Image], RenderedMultiImagePrompt], PreparedPrompt
    ]
    prepare_generation: Callable[[Any, Image.Image, str], Any]


def _prepare_qwen_no_thinking_generation(
    processor: Any,
    image: Image.Image,
    prompt: str,
) -> Any:
    """Bind EVAL-08 Qwen generation to the no-thinking chat template."""
    return prepare_qwen_generation(processor, image, prompt, thinking=False)


def configure_image_max_pixels(
    runtime: LoadedModelRuntime,
    image_max_pixels: int | None,
) -> int | None:
    """Fix the processor image limit before forward execution (DIA-02).

    ``None`` restores the pinned checkpoint limit. A numeric override is
    supported only by registered Qwen3.5-family models and deterministically
    bounds processor resize;
    it never changes source image bytes or enables OOM recovery.

    Args:
        runtime: Loaded formal model runtime.
        image_max_pixels: Inclusive pixel ceiling, or ``None`` for the pinned
            checkpoint default.

    Returns:
        The active Qwen maximum pixel count, or ``None`` for Molmo runtimes.

    Raises:
        ValueError: The adapter, value, or processor layout violates the pinned
            image protocol.
    """
    if runtime.spec.key not in QWEN_MODEL_SPECS:
        if image_max_pixels is not None:
            raise ValueError("image_max_pixels overrides are supported only by Qwen")
        return None
    if image_max_pixels is not None and (
        isinstance(image_max_pixels, bool) or not isinstance(image_max_pixels, int)
    ):
        raise ValueError("image_max_pixels must be an integer or null")
    qwen_spec = QWEN_MODEL_SPECS[runtime.spec.key]
    target = (
        qwen_spec.image_max_pixels if image_max_pixels is None else image_max_pixels
    )
    if not qwen_spec.image_min_pixels <= target <= qwen_spec.image_max_pixels:
        raise ValueError(
            "Qwen image_max_pixels must be within the pinned processor bounds"
        )
    image_processor = getattr(runtime.processor, "image_processor", None)
    size = getattr(image_processor, "size", None)
    if size is None or "shortest_edge" not in size or "longest_edge" not in size:
        raise ValueError("Qwen image processor does not expose pinned pixel bounds")
    if int(size["shortest_edge"]) != qwen_spec.image_min_pixels:
        raise ValueError("Qwen image processor minimum pixel bound changed")
    size["longest_edge"] = target
    if int(size["longest_edge"]) != target:
        raise ValueError("Qwen image processor maximum pixel bound was not applied")
    return target


def load_model_runtime(
    adapter_key: str,
    model_path: Path,
    device: str,
) -> LoadedModelRuntime:
    """Load one declared model adapter without fallback or backend substitution."""
    if adapter_key not in RUNTIME_SPECS:
        raise KeyError(f"Unknown distributed model adapter {adapter_key!r}")
    spec = RUNTIME_SPECS[adapter_key]
    if adapter_key in MOLMO_MODEL_SPECS:
        processor, model = load_molmo_model(
            model_path,
            device,
            MOLMO_MODEL_SPECS[adapter_key],
            spec.attention_backend,
        )
        blocks = tuple(model.model.transformer.blocks)
        token_ids = molmo_answer_token_ids(processor.tokenizer)
        prepare = prepare_molmo_batch
        prepare_objects = prepare_molmo_object_batch
        prepare_multi_image_objects = prepare_molmo_multi_image
        prepare_generation = prepare_molmo_generation
    elif adapter_key in QWEN_MODEL_SPECS:
        processor, model = _load_qwen_model(
            model_path,
            device,
            QWEN_MODEL_SPECS[adapter_key],
            spec.attention_backend,
        )
        blocks = tuple(model.model.language_model.layers)
        token_ids = qwen_answer_token_ids(processor.tokenizer)
        prepare = prepare_qwen_batch
        prepare_objects = prepare_qwen_object_batch
        prepare_multi_image_objects = prepare_qwen_multi_image
        prepare_generation = _prepare_qwen_no_thinking_generation
    else:
        raise AssertionError("Adapter registry and loader branches differ")
    if len(blocks) != spec.layer_count:
        raise ValueError(
            f"Runtime exposes {len(blocks)} layers, expected {spec.layer_count}"
        )
    if type(model).__name__ != spec.architecture:
        raise ValueError(
            f"Runtime architecture {type(model).__name__!r} != {spec.architecture!r}"
        )
    allowed_dtypes = {spec.dtype}
    if adapter_key in QWEN_MODEL_SPECS:
        expected_float32_numel = QWEN_MODEL_SPECS[adapter_key].float32_parameter_numel
        if expected_float32_numel:
            allowed_dtypes.add(torch.float32)
    invalid_dtypes = sorted(
        {
            str(parameter.dtype)
            for parameter in model.parameters()
            if parameter.is_floating_point() and parameter.dtype not in allowed_dtypes
        }
    )
    if invalid_dtypes:
        raise ValueError(
            f"Runtime floating parameters differ from {spec.dtype}: {invalid_dtypes}"
        )
    float32_numel = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype == torch.float32
    )
    if adapter_key in QWEN_MODEL_SPECS and float32_numel != expected_float32_numel:
        raise ValueError(
            f"Runtime FP32 parameter count {float32_numel} differs from pinned "
            f"{expected_float32_numel}"
        )
    return LoadedModelRuntime(
        spec,
        processor,
        model,
        blocks,
        token_ids,
        prepare,
        prepare_objects,
        prepare_multi_image_objects,
        prepare_generation,
    )
