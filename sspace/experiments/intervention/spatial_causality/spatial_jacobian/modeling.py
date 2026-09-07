"""Molmo2-specific prompt rendering, token alignment, and activation hooks."""

from __future__ import annotations

import io
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from PIL import Image

from sspace.core.models.molmo import _validate_snapshot_path

from .constants import (
    DEFAULT_HF_HOME,
    DEFAULT_MODEL_PATH,
    MODEL_ID,
    MODEL_REVISION,
    TEMPLATES,
)
from .io_utils import clear_proxy_environment


@dataclass(frozen=True)
class PromptJob:
    sample_idx: int
    style: str
    orientation: str
    group: str
    category: str


def resolve_model_path(
    hf_home: Path = DEFAULT_HF_HOME, allow_download: bool = True
) -> Path:
    if DEFAULT_MODEL_PATH.is_dir():
        return _validate_snapshot_path(DEFAULT_MODEL_PATH, MODEL_ID, MODEL_REVISION)
    snapshot = (
        hf_home / "hub" / "models--allenai--Molmo2-ER" / "snapshots" / MODEL_REVISION
    )
    if snapshot.is_dir():
        return snapshot
    if not allow_download:
        raise FileNotFoundError(f"Pinned Molmo2-ER snapshot is not cached: {snapshot}")
    clear_proxy_environment()
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            cache_dir=hf_home / "hub",
        )
    )


def load_model(
    model_path: Path,
    device: str,
    require_gradients: bool,
) -> tuple[Any, torch.nn.Module]:
    model_path = _validate_snapshot_path(model_path, MODEL_ID, MODEL_REVISION)
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != "4.57.6":
        raise RuntimeError(
            "The formal Molmo2-ER run requires transformers==4.57.6; "
            f"found {transformers.__version__}. Use the README uv overlay command."
        )
    # Do not pass ``use_fast=False`` here: on Transformers 4.57 it also forces
    # the Qwen tokenizer onto its unavailable slow-vocab path. The checkpoint
    # itself selects the reference slow image processor and fast tokenizer.
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    if not processor.tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact object-token offsets")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype="auto",
        attn_implementation="sdpa",
        device_map="auto" if device == "auto" else {"": device},
        local_files_only=True,
    ).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(require_gradients)
    return processor, model


def render_with_spans(
    template: str, values: Mapping[str, str]
) -> tuple[str, dict[str, tuple[int, int]]]:
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for literal, field_name, format_spec, conversion in string.Formatter().parse(
        template
    ):
        parts.append(literal)
        cursor += len(literal)
        if field_name is None:
            continue
        if field_name not in {"obj1", "obj2"} or field_name in spans:
            raise ValueError(f"Invalid template field {field_name!r}")
        value: Any = values[field_name]
        if conversion == "r":
            value = repr(value)
        elif conversion == "s":
            value = str(value)
        elif conversion == "a":
            value = ascii(value)
        elif conversion is not None:
            raise ValueError(f"Unsupported conversion !{conversion}")
        rendered = format(value, format_spec)
        start = cursor
        parts.append(rendered)
        cursor += len(rendered)
        spans[field_name] = (start, cursor)
    if set(spans) != {"obj1", "obj2"}:
        raise ValueError("Each template must contain obj1 and obj2 exactly once")
    return "".join(parts), spans


def render_job(job: PromptJob, sample: Any) -> tuple[str, dict[str, tuple[int, int]]]:
    values = (
        {"obj1": sample.obj1, "obj2": sample.obj2}
        if job.orientation == "original"
        else {"obj1": sample.obj2, "obj2": sample.obj1}
    )
    return render_with_spans(TEMPLATES[job.style][job.group], values)


def _find_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    matches = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if list(sequence[index : index + len(subsequence)]) == list(subsequence)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one question-token match, found {len(matches)}")
    return matches[0]


def locate_object_tokens(
    tokenizer: Any,
    full_input_ids: Sequence[int],
    question: str,
    spans: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, int], dict[str, str]]:
    encoded = tokenizer(question, add_special_tokens=False, return_offsets_mapping=True)
    question_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    question_start = _find_subsequence(list(full_input_ids), question_ids)
    positions: dict[str, int] = {}
    pieces: dict[str, str] = {}
    for field, (char_start, char_end) in spans.items():
        overlapping = [
            index
            for index, (start, end) in enumerate(offsets)
            if start < char_end and end > char_start
        ]
        if not overlapping:
            raise ValueError(f"No token overlaps {field} span {(char_start, char_end)}")
        local_index = overlapping[-1]
        positions[field] = question_start + local_index
        pieces[field] = tokenizer.decode([question_ids[local_index]])
    return positions, pieces


def prepare_batch(
    processor: Any,
    samples: Sequence[Any],
    jobs: Sequence[PromptJob],
) -> tuple[
    dict[str, torch.Tensor], list[dict[str, int]], list[dict[str, str]], list[str]
]:
    conversations = []
    rendered: list[tuple[str, dict[str, tuple[int, int]]]] = []
    image_cache: dict[int, Image.Image] = {}
    for job in jobs:
        sample = samples[job.sample_idx]
        if job.sample_idx not in image_cache:
            image_cache[job.sample_idx] = Image.open(
                io.BytesIO(sample.image_bytes)
            ).convert("RGB")
        question, spans = render_job(job, sample)
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_cache[job.sample_idx]},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        )
        rendered.append((question, spans))
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    positions, pieces = [], []
    for batch_index, (question, spans) in enumerate(rendered):
        pos, token_pieces = locate_object_tokens(
            processor.tokenizer,
            inputs["input_ids"][batch_index].tolist(),
            question,
            spans,
        )
        if max(pos.values()) >= inputs["input_ids"].shape[1] - 1:
            raise ValueError("Object tokens must precede the final prompt token")
        positions.append(pos)
        pieces.append(token_pieces)
    return inputs, positions, pieces, [item[0] for item in rendered]


class ActivationRecorder:
    """Capture post-block activations from a zero-based Molmo2 block index."""

    def __init__(self, model: torch.nn.Module, layer: int, detach: bool = True):
        blocks = model.model.transformer.blocks
        if not 0 <= layer < len(blocks):
            raise ValueError(f"Layer {layer} outside [0,{len(blocks) - 1}]")
        self.model = model
        self.block = blocks[layer]
        self.detach = detach
        self.activation: torch.Tensor | None = None
        self.handle: Any = None

    def _hook(self, module, inputs, output):
        del module, inputs
        tensor = output[0] if isinstance(output, tuple) else output
        self.activation = tensor.detach().float().cpu() if self.detach else tensor

    def __enter__(self):
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()
        self.handle = None

    def extract(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.activation = None
        with torch.inference_mode():
            self.model.model(**inputs, use_cache=False, return_dict=True)
        if self.activation is None:
            raise RuntimeError("Activation hook did not fire")
        return self.activation


class MultiLayerFinalTokenRecorder:
    """Capture post-block final-prompt-token activations from several blocks."""

    def __init__(self, model: torch.nn.Module, layers: Sequence[int]):
        blocks = model.model.transformer.blocks
        self.model = model
        self.layers = tuple(int(layer) for layer in layers)
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("Layers must be non-empty and unique")
        if min(self.layers) < 0 or max(self.layers) >= len(blocks):
            raise ValueError(f"Layers {self.layers} outside [0,{len(blocks) - 1}]")
        self.blocks = {layer: blocks[layer] for layer in self.layers}
        self.activations: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(module, inputs, output):
            del module, inputs
            tensor = output[0] if isinstance(output, tuple) else output
            self.activations[layer] = tensor[:, -1, :].detach().float().cpu()

        return capture

    def __enter__(self):
        self.handles = [
            self.blocks[layer].register_forward_hook(self._hook(layer))
            for layer in self.layers
        ]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def extract(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.activations = {}
        with torch.inference_mode():
            self.model.model(**inputs, use_cache=False, return_dict=True)
        missing = set(self.layers) - set(self.activations)
        if missing:
            raise RuntimeError(
                f"Activation hooks did not fire for layers {sorted(missing)}"
            )
        values = torch.stack([self.activations[layer][0] for layer in self.layers])
        if values.ndim != 2 or not torch.isfinite(values).all():
            raise ValueError(f"Invalid final-token activations: {tuple(values.shape)}")
        return values


class DirectionalVJPRunner:
    """Compute source-state gradients of a target final-token direction."""

    def __init__(self, model: torch.nn.Module, source_layer: int, target_layer: int):
        blocks = model.model.transformer.blocks
        if not 0 <= source_layer < target_layer < len(blocks):
            raise ValueError(f"Need 0 <= source < target < {len(blocks)}")
        self.model = model
        self.source_block = blocks[source_layer]
        self.target_block = blocks[target_layer]
        self.source: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.handles: list[Any] = []

    def _source_hook(self, module, inputs, output):
        del module, inputs
        tensor = output[0] if isinstance(output, tuple) else output
        tensor.requires_grad_(True)
        self.source = tensor

    def _target_hook(self, module, inputs, output):
        del module, inputs
        self.target = output[0] if isinstance(output, tuple) else output

    def __enter__(self):
        self.handles = [
            self.source_block.register_forward_hook(self._source_hook),
            self.target_block.register_forward_hook(self._target_hook),
        ]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def gradients(
        self, inputs: Mapping[str, torch.Tensor], directions: torch.Tensor
    ) -> torch.Tensor:
        self.source = self.target = None
        with torch.enable_grad():
            self.model.model(**inputs, use_cache=False, return_dict=True)
            if self.source is None or self.target is None:
                raise RuntimeError("Directional VJP hooks did not fire")
            direction = directions.to(self.target.device, self.target.dtype)
            scalar = (self.target[:, -1, :] * direction).sum()
            gradient = torch.autograd.grad(scalar, self.source)[0]
        return gradient.detach().float().cpu()


def move_inputs(
    inputs: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    return {name: value.to(model.device) for name, value in inputs.items()}


def generated_answer(
    processor: Any,
    model: torch.nn.Module,
    sample: Any,
    max_new_tokens: int,
) -> str:
    image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample.original_question},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = move_inputs(inputs, model)
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    input_length = inputs["input_ids"].shape[1]
    return processor.tokenizer.decode(
        generated[0, input_length:], skip_special_tokens=True
    ).strip()
