"""Implement pinned Qwen-family adapters for formal S-Space execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image

from ..prompts.rendering import (
    RenderedMultiImagePrompt,
    RenderedObjectPrompt,
    RenderedPrompt,
)
from .molmo import PreparedPrompt, _validate_snapshot_path
from .token_alignment import named_object_token_positions, object_token_positions


@dataclass(frozen=True)
class Qwen35ModelSpec:
    """Declare one frozen Qwen3.5-family representation space."""

    key: str
    model_id: str
    revision: str
    transformers_version: str
    architecture: str
    layer_count: int
    hidden_size: int
    image_min_pixels: int
    image_max_pixels: int
    float32_parameter_numel: int


QWEN36_SPEC = Qwen35ModelSpec(
    key="qwen36_27b",
    model_id="Qwen/Qwen3.6-27B",
    revision="6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
    transformers_version="5.9.0",
    architecture="Qwen3_5ForConditionalGeneration",
    layer_count=64,
    hidden_size=5120,
    image_min_pixels=65536,
    image_max_pixels=16777216,
    float32_parameter_numel=0,
)
QWEN35_4B_SPEC = Qwen35ModelSpec(
    key="qwen35_4b",
    model_id="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    transformers_version="5.9.0",
    architecture="Qwen3_5ForConditionalGeneration",
    layer_count=32,
    hidden_size=2560,
    image_min_pixels=65536,
    image_max_pixels=16777216,
    float32_parameter_numel=0,
)
QWEN_MODEL_SPECS = {spec.key: spec for spec in (QWEN36_SPEC, QWEN35_4B_SPEC)}
# Preserve the original public type name for downstream imports.
Qwen36ModelSpec = Qwen35ModelSpec
ENDPOINTS = {
    "horizontal": ("left", "right"),
    "vertical": ("above", "below"),
    "distance": ("far", "close"),
}


def _load_qwen_model(
    model_path: Path,
    device: str,
    spec: Qwen35ModelSpec,
    attention_backend: str,
) -> tuple[Any, torch.nn.Module]:
    """Load one complete pinned Qwen runtime on one GPU.

    Args:
        model_path: Local Hugging Face snapshot whose directory name is the
            declared revision SHA.
        device: Exact target device for the complete checkpoint.
        spec: Registered Qwen representation-space declaration.
        attention_backend: Frozen attention implementation; only ``sdpa`` is
            accepted.

    Returns:
        The audited processor and evaluation-mode model.

    Raises:
        RuntimeError: Transformers differs from the audited 5.9.0 runtime.
        ValueError: Tokenizer, architecture, layer, hidden-size, or runtime dtype
            checks fail.

    Side effects:
        Loads the complete declared model on ``device``.
    """
    if spec.key not in QWEN_MODEL_SPECS or QWEN_MODEL_SPECS[spec.key] != spec:
        raise ValueError(f"Unknown Qwen model specification {spec.key!r}")
    model_path = _validate_snapshot_path(model_path, spec.model_id, spec.revision)
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != spec.transformers_version:
        raise RuntimeError(
            f"{spec.model_id} requires transformers "
            f"{spec.transformers_version}, found {transformers.__version__}"
        )
    if attention_backend != "sdpa":
        raise ValueError("Formal Qwen extraction requires sdpa attention")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    if not processor.tokenizer.is_fast:
        raise ValueError(f"{spec.model_id} requires a fast tokenizer for exact offsets")
    image_size = processor.image_processor.size
    if (
        int(image_size["shortest_edge"]) != spec.image_min_pixels
        or int(image_size["longest_edge"]) != spec.image_max_pixels
    ):
        raise ValueError(
            "Qwen image processor pixel bounds differ from the frozen spec"
        )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map={"": device},
        attn_implementation=attention_backend,
        local_files_only=True,
    ).eval()
    text_config = model.config.text_config
    if type(model).__name__ != spec.architecture:
        raise ValueError(f"Unexpected Qwen architecture {type(model).__name__!r}")
    if (
        int(text_config.num_hidden_layers) != spec.layer_count
        or int(text_config.hidden_size) != spec.hidden_size
    ):
        raise ValueError(
            "Qwen text layer count or hidden size differs from the frozen spec"
        )
    allowed_dtypes = {torch.bfloat16}
    if spec.float32_parameter_numel:
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
            f"Qwen floating parameters use undeclared dtypes: {invalid_dtypes}"
        )
    float32_numel = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.is_floating_point() and parameter.dtype == torch.float32
    )
    if float32_numel != spec.float32_parameter_numel:
        raise ValueError(
            f"Qwen FP32 parameter count {float32_numel} differs from pinned "
            f"{spec.float32_parameter_numel}"
        )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def answer_token_ids(tokenizer: Any) -> dict[str, tuple[int, int]]:
    """Return exact one-token negative/positive Qwen IDs for H/V/D."""
    output: dict[str, tuple[int, int]] = {}
    for group, endpoints in ENDPOINTS.items():
        values = []
        for word in endpoints:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) != 1 or tokenizer.decode(ids) != word:
                raise ValueError(
                    f"Qwen endpoint {word!r} is not one exact token: {ids}"
                )
            values.append(int(ids[0]))
        output[group] = (values[0], values[1])
    return output


def prepare_prompt_batch(
    processor: Any,
    image: Image.Image,
    prompts: Sequence[RenderedPrompt],
) -> PreparedPrompt:
    """Apply Qwen's chat template with thinking explicitly disabled.

    The returned positions select the last text subtoken overlapping each
    declared object span. Hooks later attach only to Qwen text decoder blocks.
    """
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.text},
                ],
            }
        ]
        for prompt in prompts
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={"padding": True},
    )
    positions, pieces = [], []
    for batch, prompt in enumerate(prompts):
        position, piece = object_token_positions(
            processor.tokenizer,
            inputs["input_ids"][batch].tolist(),
            prompt,
        )
        if max(position.values()) >= inputs["input_ids"].shape[1] - 1:
            raise ValueError("Qwen object tokens must precede the answer position")
        positions.append(position)
        pieces.append(piece)
    return PreparedPrompt(inputs, tuple(positions), tuple(pieces))


def prepare_object_prompt_batch(
    processor: Any,
    image: Image.Image,
    prompts: Sequence[RenderedObjectPrompt],
) -> PreparedPrompt:
    """Apply Qwen's no-thinking template and align all object roles (EVAL-08)."""
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt.text},
                ],
            }
        ]
        for prompt in prompts
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={"padding": True},
    )
    positions, pieces = [], []
    for batch_index, prompt in enumerate(prompts):
        position, piece = named_object_token_positions(
            processor.tokenizer,
            inputs["input_ids"][batch_index].tolist(),
            prompt.text,
            prompt.object_spans,
        )
        if max(position.values()) >= inputs["input_ids"].shape[1] - 1:
            raise ValueError("Qwen object tokens must precede the answer position")
        positions.append(position)
        pieces.append(piece)
    return PreparedPrompt(inputs, tuple(positions), tuple(pieces))


def prepare_multi_image_object_prompt(
    processor: Any,
    images: Sequence[Image.Image],
    prompt: RenderedMultiImagePrompt,
) -> PreparedPrompt:
    """Format one Qwen multi-image prompt with Thinking disabled."""
    if len(images) != len(prompt.text_segments):
        raise ValueError("Multi-image and text-segment counts differ")
    content = []
    for image, text in zip(images, prompt.text_segments, strict=True):
        content.append({"type": "image", "image": image})
        if text:
            content.append({"type": "text", "text": text})
    inputs = processor.apply_chat_template(
        [[{"role": "user", "content": content}]],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_tensors="pt",
        return_dict=True,
        processor_kwargs={"padding": True},
    )
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Qwen multi-image input must have batch size one")
    position, piece = named_object_token_positions(
        processor.tokenizer,
        inputs["input_ids"][0].tolist(),
        prompt.text_segments[prompt.object_text_index],
        prompt.object_spans,
    )
    if max(position.values()) >= inputs["input_ids"].shape[1] - 1:
        raise ValueError(
            "Qwen multi-image object token must precede the answer position"
        )
    return PreparedPrompt(inputs, (position,), (piece,))


def prepare_direct_generation(
    processor: Any,
    image: Image.Image,
    prompt: str,
    *,
    thinking: bool,
) -> Any:
    """Apply the pinned Qwen chat template for one EVAL-07 generation.

    This boundary deliberately exposes no thinking-budget argument. The caller
    controls only whether Qwen's native Thinking template is enabled; decoding
    remains the responsibility of the direct-generation evaluator.

    Raises:
        ValueError: The prompt is empty or ``thinking`` is not boolean.

    Side effects:
        Runs the processor on one image and prompt; it does not run the model.
    """
    if not prompt.strip():
        raise ValueError("Direct-generation prompt must be non-empty")
    if not isinstance(thinking, bool):
        raise ValueError("thinking must be boolean")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
        return_tensors="pt",
        return_dict=True,
    )
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Qwen direct-generation input must have batch size one")
    return inputs
