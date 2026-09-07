"""Execute model-independent post-block states and fixed contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ..prompts.templates import AXIS_ORDER


def move_model_inputs(
    inputs: Mapping[str, torch.Tensor], model: torch.nn.Module
) -> dict[str, torch.Tensor]:
    """Move processor tensors to the complete model's device without mutation."""
    return {name: value.to(model.device) for name, value in inputs.items()}


def _select_named_states(
    value: torch.Tensor,
    positions: Sequence[Mapping[str, int]],
    roles: Sequence[str],
) -> torch.Tensor:
    return torch.stack(
        [
            torch.stack([value[batch, int(position[role]), :] for role in roles])
            for batch, position in enumerate(positions)
        ]
    )


class LayerwiseFixedContrastGradient:
    """Read all fixed logits and one task-group role gradient at every layer.

    For task group ``g`` and post-block layer ``l``, this class computes

    ``gradient = (grad_query z_g - grad_reference z_g) / 2``

    where ``z_g`` is right-minus-left, below-minus-above, or close-minus-far.
    It also stores the un-subtracted query and reference post-block states.
    The class implements TRN-02D and never reads endpoint labels.
    """

    def __init__(
        self, model: torch.nn.Module, blocks: Sequence[torch.nn.Module]
    ) -> None:
        if not blocks:
            raise ValueError("A model adapter must expose at least one text block")
        self.model = model
        self.blocks = tuple(blocks)
        self.layer_ids = tuple(range(len(self.blocks)))
        self.sources: dict[int, torch.Tensor] = {}
        self.handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            value = output[0] if isinstance(output, tuple) else output
            value.requires_grad_(True)
            self.sources[layer] = value

        return capture

    def __enter__(self) -> "LayerwiseFixedContrastGradient":
        self.handles = [
            block.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(self.blocks)
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.sources = {}

    def extract(
        self,
        inputs: Mapping[str, torch.Tensor],
        positions: Sequence[Mapping[str, int]],
        endpoint_token_ids: Mapping[str, tuple[int, int]],
        task_group: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return contrasts, role gradients, and separate object states.

        Args:
            inputs: Complete processor batch on the model device.
            positions: Query/reference text-token indices for each batch item.
            endpoint_token_ids: Negative/positive token IDs for all H/V/D groups.
            task_group: Group whose fixed contrast is differentiated.

        Returns:
            Float32 CPU tensors: all contrasts ``[B,3]`` in H/V/D order,
            task gradients ``[B,L,D]``, and object states ``[B,L,2,D]`` with
            object role order query, reference.

        Raises:
            ValueError: Group, token, batch, or tensor shapes are invalid.
            RuntimeError: A declared post-block hook does not fire.
            FloatingPointError: Any returned value is non-finite.

        Side effects:
            Executes one forward and one backward through the task contrast.
        """
        if task_group not in AXIS_ORDER:
            raise ValueError(f"Unknown task group {task_group!r}")
        if set(endpoint_token_ids) != set(AXIS_ORDER):
            raise ValueError("Endpoint token IDs must cover H/V/D exactly")
        self.sources = {}
        with torch.enable_grad():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            missing = set(self.layer_ids) - set(self.sources)
            if missing:
                raise RuntimeError(
                    f"Text-layer hooks did not fire for {sorted(missing)}"
                )
            logits = outputs.logits[:, -1, :]
            if len(positions) != logits.shape[0]:
                raise ValueError("Object positions do not match the prompt batch")
            contrasts = torch.stack(
                [
                    logits[:, endpoint_token_ids[group][1]]
                    - logits[:, endpoint_token_ids[group][0]]
                    for group in AXIS_ORDER
                ],
                dim=-1,
            )
            gradients = torch.autograd.grad(
                contrasts[:, AXIS_ORDER.index(task_group)].sum(),
                tuple(self.sources[layer] for layer in self.layer_ids),
            )
            role_gradients = torch.stack(
                [
                    torch.stack(
                        [
                            (
                                gradient[batch, int(position["query"]), :]
                                - gradient[batch, int(position["reference"]), :]
                            )
                            / 2.0
                            for batch, position in enumerate(positions)
                        ]
                    )
                    for gradient in gradients
                ],
                dim=1,
            )
            object_states = torch.stack(
                [
                    _select_named_states(source, positions, ("query", "reference"))
                    for source in (self.sources[layer] for layer in self.layer_ids)
                ],
                dim=1,
            )
        result = tuple(
            value.detach().float().cpu()
            for value in (contrasts, role_gradients, object_states)
        )
        self.sources = {}
        if not all(torch.isfinite(value).all() for value in result):
            raise FloatingPointError(
                "Contrast, role gradient, or object state is non-finite"
            )
        expected = (
            (len(positions), len(AXIS_ORDER)),
            (len(positions), len(self.layer_ids), role_gradients.shape[-1]),
            (len(positions), len(self.layer_ids), 2, object_states.shape[-1]),
        )
        if tuple(value.shape for value in result) != expected:
            raise ValueError(
                "Fixed-contrast extraction returned unexpected tensor shapes"
            )
        return result  # type: ignore[return-value]


class LayerwisePostBlockStates:
    """Read query-minus-reference states at every declared text post-block."""

    def __init__(
        self, model: torch.nn.Module, blocks: Sequence[torch.nn.Module]
    ) -> None:
        if not blocks:
            raise ValueError("A model adapter must expose at least one text block")
        self.model = model
        self.blocks = tuple(blocks)
        self.layer_ids = tuple(range(len(self.blocks)))
        self.sources: dict[int, torch.Tensor] = {}
        self.positions: Sequence[Mapping[str, int]] | None = None
        self.handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            if self.positions is None:
                raise RuntimeError("Layerwise object positions were not initialized")
            value = output[0] if isinstance(output, tuple) else output
            self.sources[layer] = _select_named_states(
                value, self.positions, ("query", "reference")
            )

        return capture

    def __enter__(self) -> "LayerwisePostBlockStates":
        self.handles = [
            block.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(self.blocks)
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.sources = {}
        self.positions = None

    def extract(
        self,
        inputs: Mapping[str, torch.Tensor],
        positions: Sequence[Mapping[str, int]],
    ) -> torch.Tensor:
        """Return finite float32 CPU differences ``[B,L,D]`` without generation."""
        if not positions or any(
            set(position) != {"query", "reference"} for position in positions
        ):
            raise ValueError("Object positions must cover query and reference exactly")
        self.sources = {}
        self.positions = positions
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            if len(positions) != int(outputs.logits.shape[0]):
                raise ValueError("Object positions do not match the prompt batch")
            missing = set(self.layer_ids) - set(self.sources)
            if missing:
                raise RuntimeError(
                    f"Text-layer hooks did not fire for {sorted(missing)}"
                )
            states = torch.stack(
                [self.sources[layer] for layer in self.layer_ids], dim=1
            )
            differences = states[:, :, 0, :] - states[:, :, 1, :]
        result = differences.detach().float().cpu()
        self.sources = {}
        self.positions = None
        if not torch.isfinite(result).all():
            raise FloatingPointError("Layerwise object difference is non-finite")
        return result


class LayerwiseNamedObjectStates:
    """Read named object states from every declared post-block."""

    def __init__(
        self, model: torch.nn.Module, blocks: Sequence[torch.nn.Module]
    ) -> None:
        if not blocks:
            raise ValueError("A model adapter must expose at least one text block")
        self.model = model
        self.blocks = tuple(blocks)
        self.layer_ids = tuple(range(len(self.blocks)))
        self.sources: dict[int, torch.Tensor] = {}
        self.positions: Sequence[Mapping[str, int]] | None = None
        self.object_order: tuple[str, ...] = ()
        self.handles: list[Any] = []

    def _hook(self, layer: int):
        def capture(module: Any, inputs: Any, output: Any) -> None:
            del module, inputs
            if self.positions is None or not self.object_order:
                raise RuntimeError("Layerwise object positions were not initialized")
            value = output[0] if isinstance(output, tuple) else output
            self.sources[layer] = _select_named_states(
                value, self.positions, self.object_order
            )

        return capture

    def __enter__(self) -> "LayerwiseNamedObjectStates":
        self.handles = [
            block.register_forward_hook(self._hook(layer))
            for layer, block in enumerate(self.blocks)
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.sources = {}
        self.positions = None
        self.object_order = ()

    def extract(
        self,
        inputs: Mapping[str, torch.Tensor],
        positions: Sequence[Mapping[str, int]],
        object_order: Sequence[str],
    ) -> torch.Tensor:
        """Return finite float32 CPU states ``[B,L,O,D]``.

        Each hook selects only the declared object-token positions, so the
        runner does not retain full image-text sequences for every layer.
        One inference forward supplies every layer.
        """
        roles = tuple(object_order)
        if not positions or not roles or len(set(roles)) != len(roles):
            raise ValueError("Layerwise object positions and roles must be non-empty")
        if any(set(position) != set(roles) for position in positions):
            raise ValueError("Layerwise object positions must cover every role exactly")
        self.sources = {}
        self.positions = positions
        self.object_order = roles
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            if len(positions) != int(outputs.logits.shape[0]):
                raise ValueError("Object positions do not match the prompt batch")
            missing = set(self.layer_ids) - set(self.sources)
            if missing:
                raise RuntimeError(
                    f"Text-layer hooks did not fire for {sorted(missing)}"
                )
            states = torch.stack(
                [self.sources[layer] for layer in self.layer_ids], dim=1
            )
        result = states.detach().float().cpu()
        self.sources = {}
        self.positions = None
        self.object_order = ()
        expected = (len(positions), len(self.layer_ids), len(roles), result.shape[-1])
        if tuple(result.shape) != expected:
            raise ValueError(
                f"Layerwise object states {tuple(result.shape)} != {expected}"
            )
        if not torch.isfinite(result).all():
            raise FloatingPointError("Layerwise object states are non-finite")
        return result


class SelectedLayerObjectStates:
    """Read named object states from one frozen post-block layer (EVAL-08/10)."""

    def __init__(self, model: torch.nn.Module, block: torch.nn.Module) -> None:
        self.model = model
        self.block = block
        self.source: torch.Tensor | None = None
        self.handle: Any | None = None

    def _capture(self, module: Any, inputs: Any, output: Any) -> None:
        del module, inputs
        self.source = output[0] if isinstance(output, tuple) else output

    def __enter__(self) -> "SelectedLayerObjectStates":
        self.handle = self.block.register_forward_hook(self._capture)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.source = None

    def extract(
        self,
        inputs: Mapping[str, torch.Tensor],
        positions: Sequence[Mapping[str, int]],
        object_order: Sequence[str],
    ) -> torch.Tensor:
        """Return finite float32 states ``[B,O,D]`` in declared role order.

        Args:
            inputs: Complete processor batch on the model device.
            positions: One named role-to-token mapping per batch item.
            object_order: Stable role order shared across every prompt.

        Returns:
            Selected post-block states with shape ``[batch, objects, hidden]``.

        Raises:
            ValueError: Batch, role, or output shapes are inconsistent.
            RuntimeError: The selected block hook does not fire.
            FloatingPointError: A selected state is non-finite.

        Side effects:
            Executes one inference forward and temporarily stores one block.
        """
        if not object_order or len(set(object_order)) != len(object_order):
            raise ValueError("Object order must contain unique named roles")
        self.source = None
        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
            if self.source is None:
                raise RuntimeError("Selected text-layer hook did not fire")
            if len(positions) != int(outputs.logits.shape[0]):
                raise ValueError("Object positions do not match the prompt batch")
            values = torch.stack(
                [
                    torch.stack(
                        [
                            self.source[batch, int(position[role]), :]
                            for role in object_order
                        ]
                    )
                    for batch, position in enumerate(positions)
                ]
            )
        result = values.detach().float().cpu()
        self.source = None
        if not torch.isfinite(result).all():
            raise FloatingPointError("Selected-layer object state is non-finite")
        return result
