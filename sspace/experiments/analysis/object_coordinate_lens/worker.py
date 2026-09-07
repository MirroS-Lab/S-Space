#!/usr/bin/env python3
"""Persistent JSON-lines model worker for interactive object coordinates.

Exactly one pinned checkpoint is loaded by each process.  Stdout is reserved
for the JSON-lines protocol; model and diagnostic logging always goes to
stderr so a parent process can safely multiplex workers from different
Transformers runtimes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import sys
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from sspace.core.models.molmo import _validate_snapshot_path

try:
    from .models import (
        ModelSpec,
        checkpoint_path_for,
        get_model_spec,
        validate_model_artifacts,
    )
except ImportError:  # Direct worker.py execution by the inference manager.
    from models import (  # type: ignore[no-redef]
        ModelSpec,
        checkpoint_path_for,
        get_model_spec,
        validate_model_artifacts,
    )


LOGGER = logging.getLogger("object_coordinate_lens.worker")
MENTION_INSTRUCTION = """Identify the distinct concrete physical object phrases that are both explicitly mentioned in USER_PROMPT and visible in the image. Return only a JSON array of strings. Every item must be an exact case-sensitive substring of USER_PROMPT. Exclude instructions, abstract concepts, properties, and objects not visible in the image. Return between 2 and 8 unique objects.

USER_PROMPT:
{prompt}"""


class WorkerProtocolError(ValueError):
    """A request does not conform to the worker's frozen JSON protocol."""


def _json_line(payload: Mapping[str, Any]) -> None:
    """Write one protocol message and flush it immediately."""

    sys.stdout.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")
    sys.stdout.flush()


def _reject_nonstandard_json_constant(value: str) -> None:
    raise WorkerProtocolError(f"Non-standard JSON constant {value!r} is not allowed")


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkerProtocolError(f"{field} must be a non-empty string")
    return value


def _open_rgb_image(value: Any) -> Image.Image:
    image_path = Path(_require_nonempty_text(value, "image_path")).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB").copy()


def _find_unique_subsequence(
    sequence: Sequence[int], subsequence: Sequence[int]
) -> int:
    if not subsequence:
        raise ValueError("Prompt token sequence is empty")
    matches = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if list(sequence[index : index + len(subsequence)]) == list(subsequence)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one prompt-token match, found {len(matches)}")
    return matches[0]


def _validated_spans(value: Any, prompt: str) -> dict[str, tuple[int, int]]:
    if not isinstance(value, Mapping) or not value:
        raise WorkerProtocolError("spans must be a non-empty object mapping")
    spans: dict[str, tuple[int, int]] = {}
    for object_id, raw_span in value.items():
        if not isinstance(object_id, str) or not object_id.strip():
            raise WorkerProtocolError("span keys must be non-empty strings")
        if (
            not isinstance(raw_span, Sequence)
            or isinstance(raw_span, (str, bytes))
            or len(raw_span) != 2
        ):
            raise WorkerProtocolError(f"span for {object_id!r} must be [start, end]")
        start, end = raw_span
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or not 0 <= start < end <= len(prompt)
        ):
            raise WorkerProtocolError(
                f"span for {object_id!r} is outside the prompt: {raw_span!r}"
            )
        spans[object_id] = (start, end)
    if len(set(spans.values())) != len(spans):
        raise WorkerProtocolError("object spans must be distinct")
    return spans


def locate_prompt_mentions(
    tokenizer: Any,
    full_input_ids: Sequence[int],
    prompt: str,
    spans: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, Any]]:
    """Select the last subtoken overlapping every declared prompt span."""

    encoded = tokenizer(
        prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    prompt_ids = [int(value) for value in encoded["input_ids"]]
    offsets = [tuple(map(int, value)) for value in encoded["offset_mapping"]]
    if len(prompt_ids) != len(offsets):
        raise ValueError("Prompt token IDs and offsets have different lengths")
    prompt_start = _find_unique_subsequence(
        [int(value) for value in full_input_ids], prompt_ids
    )
    result: dict[str, dict[str, Any]] = {}
    for object_id, (char_start, char_end) in spans.items():
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < char_end and token_end > char_start
        ]
        if not overlapping:
            raise ValueError(
                f"No prompt token overlaps {object_id!r} at {(char_start, char_end)}"
            )
        position = prompt_start + overlapping[-1]
        model_tokens = [
            str(tokenizer.decode([prompt_ids[index]])) for index in overlapping
        ]
        result[object_id] = {
            "position": int(position),
            "piece": model_tokens[-1],
            "model_tokens": model_tokens,
            "char_span": [int(char_start), int(char_end)],
        }
    return result


def _resolve_attribute(root: Any, dotted_path: str) -> Any:
    current = root
    for component in dotted_path.split("."):
        if not component or not hasattr(current, component):
            raise ValueError(f"Model does not expose {dotted_path!r}")
        current = getattr(current, component)
    return current


class _SelectedPostBlockRecorder:
    """Capture named states at one zero-based post-block forward hook."""

    def __init__(
        self,
        block: Any,
        mentions: Mapping[str, Mapping[str, Any]],
        hidden_size: int,
    ) -> None:
        self.block = block
        self.mentions = mentions
        self.hidden_size = int(hidden_size)
        self.states: dict[str, Any] | None = None
        self.handle: Any | None = None

    def _capture(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        import torch

        value = output[0] if isinstance(output, tuple) else output
        if not isinstance(value, torch.Tensor) or value.ndim != 3:
            raise ValueError("Post-block hook did not receive a [B,S,D] tensor")
        if value.shape[0] != 1 or value.shape[2] != self.hidden_size:
            raise ValueError(
                f"Post-block output {tuple(value.shape)} is incompatible with "
                f"batch=1, hidden={self.hidden_size}"
            )
        captured: dict[str, Any] = {}
        for object_id, metadata in self.mentions.items():
            position = int(metadata["position"])
            if not 0 <= position < value.shape[1]:
                raise ValueError(
                    f"Token position {position} for {object_id!r} exceeds "
                    f"sequence length {value.shape[1]}"
                )
            captured[object_id] = value[0, position].detach().float().cpu()
        self.states = captured

    def __enter__(self) -> "_SelectedPostBlockRecorder":
        self.states = None
        self.handle = self.block.register_forward_hook(self._capture)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None


class ModelWorker:
    """Own one fully loaded model and serve its two inference operations."""

    def __init__(
        self,
        spec: ModelSpec,
        *,
        checkpoint_path: Path,
        device: str,
        mention_max_new_tokens: int = 128,
    ) -> None:
        self.spec = spec
        self.checkpoint_path = Path(checkpoint_path)
        self.device_name = self._resolve_cuda_device(device)
        self.mention_max_new_tokens = int(mention_max_new_tokens)
        if self.mention_max_new_tokens <= 0:
            raise ValueError("mention_max_new_tokens must be positive")
        self.processor: Any
        self.model: Any
        self.blocks: tuple[Any, ...]
        self.device: Any
        self.available_layers: tuple[int, ...]
        self._load_and_validate()

    @staticmethod
    def _resolve_cuda_device(device: str) -> str:
        import torch

        if device == "auto":
            device = "cuda"
        if not isinstance(device, str) or not (
            device == "cuda" or device.startswith("cuda:")
        ):
            raise ValueError("Model workers require one CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is required but torch.cuda.is_available() is false"
            )
        resolved = torch.device(device)
        if resolved.index is not None and resolved.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {device!r} is not available")
        return device

    def _validate_checkpoint(self) -> Path:
        return _validate_snapshot_path(
            self.checkpoint_path,
            self.spec.repo,
            self.spec.revision,
        )

    def _load_and_validate(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoProcessor

        axis_metadata = validate_model_artifacts(self.spec)
        self.available_layers = tuple(
            int(value) for value in axis_metadata.get("available_layers", ())
        )
        if (
            not self.available_layers
            or self.spec.default_layer not in self.available_layers
        ):
            raise ValueError(
                "Axis artifact does not expose the registered default layer"
            )
        checkpoint = self._validate_checkpoint()
        if transformers.__version__ != self.spec.transformers_version:
            raise RuntimeError(
                f"{self.spec.id} requires transformers "
                f"{self.spec.transformers_version}, found {transformers.__version__}"
            )
        processor_kwargs = {"local_files_only": True}
        if self.spec.family == "molmo":
            processor_kwargs["trust_remote_code"] = True
        self.processor = AutoProcessor.from_pretrained(checkpoint, **processor_kwargs)
        if self.spec.family == "qwen":
            image_size = getattr(self.processor.image_processor, "size", {})
            if (
                int(image_size.get("shortest_edge", -1)) != 65_536
                or int(image_size.get("longest_edge", -1)) != 16_777_216
            ):
                raise ValueError(
                    "Qwen image processor pixel bounds differ from the frozen runtime"
                )
        tokenizer = getattr(self.processor, "tokenizer", None)
        if tokenizer is None or not tokenizer.is_fast:
            raise ValueError(f"{self.spec.id} requires its pinned fast tokenizer")
        probe = tokenizer(
            "object coordinate",
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if not probe.get("input_ids") or len(probe["input_ids"]) != len(
            probe.get("offset_mapping", ())
        ):
            raise ValueError("Fast tokenizer did not return exact offset mappings")

        model_kwargs: dict[str, Any] = {
            "device_map": {"": self.device_name},
            "attn_implementation": self.spec.attention_implementation,
            "local_files_only": True,
        }
        if self.spec.family == "molmo":
            model_kwargs["trust_remote_code"] = True
        if int(transformers.__version__.split(".", maxsplit=1)[0]) >= 5:
            model_kwargs["dtype"] = "auto"
        else:
            model_kwargs["torch_dtype"] = "auto"
        self.model = AutoModelForImageTextToText.from_pretrained(
            checkpoint, **model_kwargs
        ).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        if type(self.model).__name__ != self.spec.architecture:
            raise ValueError(
                f"Runtime architecture {type(self.model).__name__!r} != "
                f"{self.spec.architecture!r}"
            )
        declared_architectures = tuple(
            getattr(self.model.config, "architectures", ()) or ()
        )
        if declared_architectures != (self.spec.architecture,):
            raise ValueError(
                f"Checkpoint architectures {declared_architectures!r} != "
                f"{(self.spec.architecture,)!r}"
            )
        text_config = getattr(self.model.config, "text_config", None)
        if text_config is None:
            raise ValueError("Model config has no text_config")
        if (
            int(getattr(text_config, "num_hidden_layers", -1)) != self.spec.layer_count
            or int(getattr(text_config, "hidden_size", -1)) != self.spec.hidden
        ):
            raise ValueError(
                "Model text layer count or hidden size differs from the registry"
            )
        blocks = _resolve_attribute(self.model, self.spec.block_path)
        self.blocks = tuple(blocks)
        if len(self.blocks) != self.spec.layer_count:
            raise ValueError(
                f"Runtime exposes {len(self.blocks)} text blocks, expected "
                f"{self.spec.layer_count}"
            )
        if any(
            layer < 0 or layer >= len(self.blocks) for layer in self.available_layers
        ):
            raise ValueError("Axis artifact exposes a layer outside the text blocks")

        parameters = tuple(self.model.parameters())
        if not parameters:
            raise ValueError("Loaded model has no parameters")
        parameter_devices = {parameter.device for parameter in parameters}
        if len(parameter_devices) != 1 or next(iter(parameter_devices)).type != "cuda":
            raise ValueError(
                "Complete model must be on one CUDA device, found "
                f"{sorted(map(str, parameter_devices))}"
            )
        self.device = next(iter(parameter_devices))
        requested_device = torch.device(self.device_name)
        if (
            requested_device.index is not None
            and self.device.index != requested_device.index
        ):
            raise ValueError(
                f"Model loaded on {self.device}, requested {requested_device}"
            )
        expected_dtype = (
            torch.float32 if self.spec.family == "molmo" else torch.bfloat16
        )
        floating_dtypes = {
            parameter.dtype for parameter in parameters if parameter.is_floating_point()
        }
        if floating_dtypes != {expected_dtype}:
            raise ValueError(
                f"Floating parameter dtypes {sorted(map(str, floating_dtypes))} != "
                f"{str(expected_dtype)!r}"
            )
        LOGGER.info(
            "Loaded %s revision %s on %s (default=L%d, layers=%d, hidden=%d)",
            self.spec.id,
            self.spec.revision,
            self.device,
            self.spec.default_layer,
            len(self.available_layers),
            self.spec.hidden,
        )

    def _validated_layer(self, value: Any) -> int:
        if value is None:
            return self.spec.default_layer
        if isinstance(value, bool) or not isinstance(value, int):
            raise WorkerProtocolError("layer must be an integer")
        layer = int(value)
        if layer not in self.available_layers:
            allowed = ", ".join(f"L{item}" for item in self.available_layers)
            raise WorkerProtocolError(
                f"Layer L{layer} has no released basis for {self.spec.id}; "
                f"expected one of: {allowed}"
            )
        return layer

    @staticmethod
    def _conversation(image: Image.Image, text: str) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text},
                ],
            }
        ]

    def _prepare_coordinate_inputs(
        self, image: Image.Image, prompt: str
    ) -> Mapping[str, Any]:
        conversations = [self._conversation(image, prompt)]
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        if self.spec.family == "qwen":
            kwargs["enable_thinking"] = False
            kwargs["processor_kwargs"] = {"padding": True}
        else:
            kwargs["padding"] = True
        inputs = self.processor.apply_chat_template(conversations, **kwargs)
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Coordinate input must have batch size one")
        return inputs

    def _prepare_generation_inputs(
        self, image: Image.Image, prompt: str
    ) -> Mapping[str, Any]:
        kwargs: dict[str, Any] = {
            "tokenize": True,
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "return_dict": True,
        }
        conversation = self._conversation(image, prompt)
        if self.spec.family == "qwen":
            kwargs["enable_thinking"] = False
            template_input: Any = conversation
        else:
            kwargs["padding"] = True
            template_input = [conversation]
        inputs = self.processor.apply_chat_template(template_input, **kwargs)
        if inputs["input_ids"].ndim != 2 or inputs["input_ids"].shape[0] != 1:
            raise ValueError("Generation input must have batch size one")
        return inputs

    def _move_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def generate_mentions(self, image_path: Any, prompt: Any) -> dict[str, str]:
        import torch

        prompt = _require_nonempty_text(prompt, "prompt")
        image = _open_rgb_image(image_path)
        instruction = MENTION_INSTRUCTION.format(prompt=prompt)
        inputs = self._prepare_generation_inputs(image, instruction)
        moved = self._move_inputs(inputs)
        input_length = int(moved["input_ids"].shape[1])
        with torch.inference_mode():
            generated = self.model.generate(
                **moved,
                max_new_tokens=self.mention_max_new_tokens,
                do_sample=False,
            )
        if generated.ndim != 2 or generated.shape[0] != 1:
            raise ValueError("Mention generation returned an unexpected token shape")
        answer = self.processor.tokenizer.decode(
            generated[0, input_length:], skip_special_tokens=True
        ).strip()
        return {"answer": answer}

    def extract_coordinates(
        self,
        image_path: Any,
        prompt: Any,
        spans: Any,
        layer: Any = None,
    ) -> dict[str, Any]:
        import torch

        selected_layer = self._validated_layer(layer)
        prompt = _require_nonempty_text(prompt, "prompt")
        validated_spans = _validated_spans(spans, prompt)
        image = _open_rgb_image(image_path)
        inputs = self._prepare_coordinate_inputs(image, prompt)
        mentions = locate_prompt_mentions(
            self.processor.tokenizer,
            inputs["input_ids"][0].tolist(),
            prompt,
            validated_spans,
        )
        sequence_length = int(inputs["input_ids"].shape[1])
        if (
            max(int(value["position"]) for value in mentions.values())
            >= sequence_length - 1
        ):
            raise ValueError("Object tokens must precede the answer position")
        moved = self._move_inputs(inputs)
        with _SelectedPostBlockRecorder(
            self.blocks[selected_layer], mentions, self.spec.hidden
        ) as recorder:
            with torch.inference_mode():
                outputs = self.model(
                    **moved,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=1,
                )
            if not hasattr(outputs, "logits") or int(outputs.logits.shape[0]) != 1:
                raise ValueError("Full model forward did not return batch-one logits")
            if recorder.states is None:
                raise RuntimeError("Selected zero-based post-block hook did not fire")
            states = recorder.states
        hidden_states: dict[str, list[float]] = {}
        for object_id, state in states.items():
            values = state.tolist()
            if len(values) != self.spec.hidden or not all(
                math.isfinite(float(value)) for value in values
            ):
                raise FloatingPointError(
                    f"Invalid hidden state returned for {object_id!r}"
                )
            hidden_states[object_id] = [float(value) for value in values]
        return {
            "hidden_states": hidden_states,
            "mentions": mentions,
            "source_layer": selected_layer,
        }

    def dispatch(self, request: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        if not isinstance(request, Mapping):
            raise WorkerProtocolError("Each request must be a JSON object")
        if "request_id" not in request:
            raise WorkerProtocolError("request_id is required")
        request_id = request["request_id"]
        action = request.get("action")
        if action == "generate_mentions":
            result = self.generate_mentions(
                request.get("image_path"), request.get("prompt")
            )
            return {"request_id": request_id, "ok": True, "result": result}, False
        if action == "extract_coordinates":
            result = self.extract_coordinates(
                request.get("image_path"),
                request.get("prompt"),
                request.get("spans"),
                request.get("layer"),
            )
            return {"request_id": request_id, "ok": True, "result": result}, False
        if action == "shutdown":
            return {
                "request_id": request_id,
                "ok": True,
                "result": {"shutdown": True},
            }, True
        raise WorkerProtocolError(f"Unknown worker action {action!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional isolated Hugging Face hub directory overriding the registry root.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args(argv)


def _error_response(request_id: Any, exc: BaseException) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "ok": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        spec = get_model_spec(args.model_id)
        checkpoint = checkpoint_path_for(spec, args.cache_dir)
        # Third-party loaders occasionally print progress to stdout.  Redirect
        # it so stdout remains a valid JSON-lines transport.
        with contextlib.redirect_stdout(sys.stderr):
            worker = ModelWorker(
                spec,
                checkpoint_path=checkpoint,
                device=args.device,
                mention_max_new_tokens=args.max_new_tokens,
            )
    except BaseException as exc:
        LOGGER.exception("Worker startup failed for %s", args.model_id)
        _json_line(
            {
                "ready": False,
                "model_id": args.model_id,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        return 1

    _json_line(
        {
            "ready": True,
            "model_id": spec.id,
            "default_layer": spec.default_layer,
            "available_layers": list(worker.available_layers),
        }
    )
    for raw_line in sys.stdin:
        request_id: Any = None
        shutdown = False
        try:
            request = json.loads(
                raw_line, parse_constant=_reject_nonstandard_json_constant
            )
            if isinstance(request, Mapping):
                request_id = request.get("request_id")
            with contextlib.redirect_stdout(sys.stderr):
                response, shutdown = worker.dispatch(request)
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            response = _error_response(request_id, exc)
        _json_line(response)
        if shutdown:
            break
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    raise SystemExit(main())
