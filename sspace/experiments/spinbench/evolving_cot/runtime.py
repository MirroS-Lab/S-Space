"""Shared token handling and pinned runtime for fixed-template CoT extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from sspace.core.models.molmo import _validate_snapshot_path
from sspace.core.models.qwen import QWEN36_SPEC

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = PROJECT_ROOT / ".cache/assets/models/qwen36_27b"
OBJECT_NAME_ALIASES = {"rubiks cube": ("rubik's cube", "rubik’s cube")}


@dataclass(frozen=True)
class _ReplayRuntime:
    processor: Any
    model: torch.nn.Module
    blocks: tuple[torch.nn.Module, ...]


def _load_runtime(attention_backend: str) -> _ReplayRuntime:
    import transformers
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if transformers.__version__ != QWEN36_SPEC.transformers_version:
        raise RuntimeError(
            f"Transformers {QWEN36_SPEC.transformers_version} required, "
            f"found {transformers.__version__}"
        )
    model_path = _validate_snapshot_path(
        MODEL_PATH,
        QWEN36_SPEC.model_id,
        QWEN36_SPEC.revision,
    )
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype="auto",
        device_map={"": "cuda:0"},
        attn_implementation=attention_backend,
        local_files_only=True,
    ).eval()
    blocks = tuple(model.model.language_model.layers)
    if (
        type(model).__name__ != QWEN36_SPEC.architecture
        or len(blocks) != QWEN36_SPEC.layer_count
        or int(model.config.text_config.hidden_size) != QWEN36_SPEC.hidden_size
    ):
        raise ValueError("Qwen3.6-27B replay runtime differs from the pinned spec")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return _ReplayRuntime(processor, model, blocks)


def _object_aliases(object_name: str) -> tuple[str, ...]:
    aliases = (object_name, *OBJECT_NAME_ALIASES.get(object_name.casefold(), ()))
    return tuple(dict.fromkeys(aliases))


def _token_variants(tokenizer: Any, object_name: str) -> tuple[tuple[int, ...], ...]:
    forms = set()
    for alias in _object_aliases(object_name):
        forms.update(
            {
                alias,
                alias.lower(),
                alias.upper(),
                alias.title(),
                alias[:1].upper() + alias[1:].lower(),
            }
        )
    variants = set()
    for form in forms:
        for prefix in ("", " ", "\n", "\t"):
            ids = tuple(
                int(value)
                for value in tokenizer(prefix + form, add_special_tokens=False)[
                    "input_ids"
                ]
            )
            if ids:
                variants.add(ids)
    return tuple(sorted(variants, key=lambda values: (len(values), values)))


def _reasoning_end(generated_ids: Sequence[int], tokenizer: Any) -> int:
    end_ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    if len(end_ids) != 1:
        raise ValueError("Qwen </think> must be one token")
    positions = [
        index for index, value in enumerate(generated_ids) if value == end_ids[0]
    ]
    return positions[-1] if positions else len(generated_ids)


def _mentions(
    generated_ids: Sequence[int], tokenizer: Any, objects: dict[str, str]
) -> list[dict[str, Any]]:
    stop = _reasoning_end(generated_ids, tokenizer)
    reasoning_ids = tuple(int(value) for value in generated_ids[:stop])
    found = set()
    output = []
    for role, name in objects.items():
        normalized_names = {
            " ".join(alias.casefold().split()) for alias in _object_aliases(name)
        }
        for variant in _token_variants(tokenizer, name):
            width = len(variant)
            for start in range(len(reasoning_ids) - width + 1):
                if reasoning_ids[start : start + width] != variant:
                    continue
                decoded = tokenizer.decode(variant, skip_special_tokens=True)
                if " ".join(decoded.casefold().split()) not in normalized_names:
                    continue
                key = (role, start, start + width)
                if key in found:
                    continue
                found.add(key)
                output.append(
                    {
                        "role": role,
                        "object_name": name,
                        "generated_token_start": start,
                        "generated_token_end": start + width,
                        "token_ids": list(variant),
                        "decoded_text": decoded,
                    }
                )
    return sorted(
        output,
        key=lambda value: (
            value["generated_token_start"],
            value["generated_token_end"],
            value["role"],
        ),
    )


def _append_generated(
    inputs: dict[str, torch.Tensor], generated: Sequence[int]
) -> None:
    ids = torch.tensor(generated, dtype=inputs["input_ids"].dtype).unsqueeze(0)
    mask = torch.ones_like(ids, dtype=inputs["attention_mask"].dtype)
    inputs["input_ids"] = torch.cat((inputs["input_ids"], ids), dim=1)
    inputs["attention_mask"] = torch.cat((inputs["attention_mask"], mask), dim=1)
    if "mm_token_type_ids" not in inputs:
        raise ValueError("Qwen multimodal replay requires mm_token_type_ids")
    text_types = torch.zeros_like(ids, dtype=inputs["mm_token_type_ids"].dtype)
    inputs["mm_token_type_ids"] = torch.cat(
        (inputs["mm_token_type_ids"], text_types), dim=1
    )
