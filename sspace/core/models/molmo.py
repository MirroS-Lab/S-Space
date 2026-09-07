"""Implement the Molmo-family boundary for formal S-Space training."""

from __future__ import annotations

import json
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
from .token_alignment import named_object_token_positions, object_token_positions
from sspace.data.hub import HUB_ASSET_MARKER


@dataclass(frozen=True)
class MolmoModelSpec:
    """Declare one pinned Molmo-family checkpoint and audited runtime.

    ``orientation_batch_size`` is the exact number of original/swapped prompts
    accepted in one forward. It is a model capability, not a performance
    heuristic, and must be either one or two.
    """

    key: str
    model_id: str
    revision: str
    transformers_version: str
    orientation_batch_size: int


MOLMO_MODEL_SPECS = {
    "molmo2_er": MolmoModelSpec(
        "molmo2_er",
        "allenai/Molmo2-ER",
        "dab22564403d2607855bb1fffb0721285b445081",
        "4.57.6",
        2,
    ),
    "molmoact2": MolmoModelSpec(
        "molmoact2",
        "allenai/MolmoAct2",
        "e432d85f6e039edca44afb93c262f3084ab72a9c",
        "5.3.0",
        2,
    ),
    "molmoact2_pretrain": MolmoModelSpec(
        "molmoact2_pretrain",
        "allenai/MolmoAct2-Pretrain",
        "a05effca9ba36c1177359b42a9d5d7a4568dbe3c",
        "5.3.0",
        1,
    ),
}
ENDPOINTS = {
    "horizontal": ("left", "right"),
    "vertical": ("above", "below"),
    "distance": ("far", "close"),
}


@dataclass(frozen=True)
class PreparedPrompt:
    """Store one model input and its audited object-token alignment."""

    inputs: dict[str, torch.Tensor]
    positions: tuple[dict[str, int], ...]
    token_pieces: tuple[dict[str, str], ...]


def _validate_snapshot_path(model_path: Path, model_id: str, revision: str) -> Path:
    """Resolve and verify one pinned Hugging Face model directory.

    Native Hub cache paths encode the revision in the resolved directory name.
    Stable Release asset paths instead carry the identity marker written only
    after ``snapshot_download`` completes the pinned repository.
    """
    resolved = Path(model_path).resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    if resolved.name == revision:
        return resolved
    marker_path = resolved / HUB_ASSET_MARKER
    if not marker_path.is_file():
        raise ValueError(
            f"{model_id} snapshot directory {resolved.name!r} has no pinned "
            f"revision marker for {revision!r}"
        )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker != {
        "schema_version": "1.0.0",
        "repo_id": model_id,
        "repo_type": "model",
        "revision": revision,
    }:
        raise ValueError(f"{model_id} snapshot revision marker differs")
    return resolved


def load_molmo_model(
    model_path: Path,
    device: str,
    spec: MolmoModelSpec,
    attention_backend: str,
) -> tuple[Any, torch.nn.Module]:
    """Load one pinned Molmo-family model for gradient extraction.

    Args:
        model_path: Local pinned Hugging Face snapshot.
        device: Exact torch device for the complete model.
        spec: Pinned model identity and required Transformers version.
        attention_backend: Explicit ``eager`` or ``sdpa`` attention path.

    Returns:
        Fast-tokenizer processor and evaluation-mode model.

    Raises:
        RuntimeError: Transformers is not the audited version.
        ValueError: The processor lacks a fast tokenizer or model identity does
            not match the pinned sspace.

    Side effects:
        Loads model weights onto ``device``.
    """
    model_path = _validate_snapshot_path(model_path, spec.model_id, spec.revision)
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != spec.transformers_version:
        raise RuntimeError(
            f"{spec.model_id} requires transformers {spec.transformers_version}, "
            f"found {transformers.__version__}"
        )
    if attention_backend not in {"eager", "sdpa"}:
        raise ValueError(f"Unsupported Molmo attention backend {attention_backend!r}")
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not processor.tokenizer.is_fast:
        raise ValueError(f"{spec.model_id} requires a fast tokenizer for exact offsets")
    dtype_argument = (
        {"dtype": "auto"}
        if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5
        else {"torch_dtype": "auto"}
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map={"": device},
        attn_implementation=attention_backend,
        local_files_only=True,
        **dtype_argument,
    ).eval()
    if str(model.config.name_or_path) not in {str(model_path), spec.model_id}:
        raise ValueError(f"Unexpected model identity {model.config.name_or_path!r}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return processor, model


def answer_token_ids(tokenizer: Any) -> dict[str, tuple[int, int]]:
    """Return one-token negative/positive IDs for fixed H/V/D contrasts.

    The returned pair order is ``(negative, positive)``. Final-logit targets
    always compute ``logit(positive)-logit(negative)``.
    """
    output = {}
    for group, endpoints in ENDPOINTS.items():
        token_ids = []
        for word in endpoints:
            ids = tokenizer(word, add_special_tokens=False)["input_ids"]
            if len(ids) != 1 or tokenizer.decode(ids) != word:
                raise ValueError(
                    f"Answer endpoint {word!r} is not one exact token: {ids}"
                )
            token_ids.append(int(ids[0]))
        output[group] = (token_ids[0], token_ids[1])
    return output


def prepare_prompt_batch(
    processor: Any,
    image: Image.Image,
    prompts: Sequence[RenderedPrompt],
) -> PreparedPrompt:
    """Format one image with multiple prompts and locate declared object spans.

    Each prompt receives the same unmodified image in an independent batch
    item. The function follows Molmo2's pinned chat template and returns exact
    token positions for the last subtoken overlapping each declared span.
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
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    positions, pieces = [], []
    for batch_index, prompt in enumerate(prompts):
        position, piece = object_token_positions(
            processor.tokenizer,
            inputs["input_ids"][batch_index].tolist(),
            prompt,
        )
        if max(position.values()) >= inputs["input_ids"].shape[1] - 1:
            raise ValueError("Object tokens must precede the answer position")
        positions.append(position)
        pieces.append(piece)
    return PreparedPrompt(inputs, tuple(positions), tuple(pieces))


def prepare_object_prompt_batch(
    processor: Any,
    image: Image.Image,
    prompts: Sequence[RenderedObjectPrompt],
) -> PreparedPrompt:
    """Format Molmo prompts and align every declared object role (EVAL-08)."""
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
        return_tensors="pt",
        return_dict=True,
        padding=True,
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
            raise ValueError("Object tokens must precede the answer position")
        positions.append(position)
        pieces.append(piece)
    return PreparedPrompt(inputs, tuple(positions), tuple(pieces))


def prepare_multi_image_object_prompt(
    processor: Any,
    images: Sequence[Image.Image],
    prompt: RenderedMultiImagePrompt,
) -> PreparedPrompt:
    """Format one Molmo multi-image prompt with exact named-object alignment.

    Args:
        processor: Pinned Molmo processor.
        images: Original images in declared view order.
        prompt: One text segment per image and spans in one declared segment.

    Returns:
        A batch-one ``PreparedPrompt`` with exact full-sequence token positions.

    Raises:
        ValueError: Image/text counts, token alignment, or batch shape differ.

    Side effects:
        Runs only the processor; it does not execute the model.
    """
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
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Molmo multi-image input must have batch size one")
    position, piece = named_object_token_positions(
        processor.tokenizer,
        inputs["input_ids"][0].tolist(),
        prompt.text_segments[prompt.object_text_index],
        prompt.object_spans,
    )
    if max(position.values()) >= inputs["input_ids"].shape[1] - 1:
        raise ValueError("Multi-image object tokens must precede the answer position")
    return PreparedPrompt(inputs, (position,), (piece,))


def prepare_direct_generation(
    processor: Any,
    image: Image.Image,
    prompt: str,
) -> Any:
    """Apply the pinned Molmo chat template for one greedy generation."""
    if not prompt.strip():
        raise ValueError("Direct-generation prompt must be non-empty")
    conversations = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    ]
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
        raise ValueError("Molmo direct-generation input must have batch size one")
    return inputs
