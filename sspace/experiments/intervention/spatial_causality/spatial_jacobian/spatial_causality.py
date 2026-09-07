"""Final-logit axes and one-shot coordinate edits for EmbSpatial."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from sspace.core.artifacts import SSpaceArtifact

from .constants import MODEL_ID, MODEL_REVISION


ARTIFACT_ID = "molmo2_er_coco6000_final_logit_axes_v2"
AXIS_ORDER = ("horizontal", "vertical", "distance")
AXIS_INDEX = {name: index for index, name in enumerate(AXIS_ORDER)}


@dataclass(frozen=True)
class AxesArtifact:
    path: Path
    layer_ids: tuple[int, ...]
    axes: torch.Tensor

    def factors(self, layer: int) -> torch.Tensor:
        try:
            row = self.layer_ids.index(int(layer))
        except ValueError as exc:
            raise ValueError(f"Axes artifact has no L{layer}") from exc
        return self.axes[row]


def load_axes_artifact(path: Path) -> AxesArtifact:
    """Load the model-owned axes through the canonical checksum boundary."""
    path = Path(path)
    directory = path if path.is_dir() else path.parent
    if not path.is_dir() and path.name != "tensors.safetensors":
        raise ValueError(f"Expected tensors.safetensors, got {path.name!r}")
    resolved = directory.resolve()
    artifact = SSpaceArtifact.load(resolved)
    manifest = artifact.manifest
    model = manifest.model
    if manifest.artifact_id != ARTIFACT_ID or resolved.name != ARTIFACT_ID:
        raise ValueError("Unexpected Spatial Causality artifact identity")
    if (
        model.model_id != MODEL_ID
        or model.revision != MODEL_REVISION
        or model.layer_numbering != "zero_based_post_block"
    ):
        raise ValueError("Axes artifact and causal model do not match")
    values = np.asarray(artifact.axes_tensor, dtype=np.float32)
    return AxesArtifact(
        resolved / "tensors.safetensors",
        model.layer_ids,
        torch.from_numpy(values.copy()),
    )


def coordinate_right_inverse(factors: torch.Tensor) -> torch.Tensor:
    if factors.ndim != 2 or factors.shape[0] != len(AXIS_ORDER):
        raise ValueError(f"Expected [3,d_model] factors, got {tuple(factors.shape)}")
    if int(torch.linalg.matrix_rank(factors)) != len(AXIS_ORDER):
        raise ValueError("Coordinate factors must have rank three")
    basis = factors.T @ torch.linalg.inv(factors @ factors.T)
    identity = torch.eye(len(AXIS_ORDER), dtype=factors.dtype, device=factors.device)
    error = float((factors @ basis - identity).abs().max())
    if error > 1e-4:
        raise ValueError(f"Coordinate right-inverse error is {error:.6g}")
    return basis


class CoordinateEditHook:
    """Apply one batched edit to post-block prefill states, then disarm."""

    MODES = {"relative_swap", "relative_common_shift", "single_shift"}

    def __init__(self, model: torch.nn.Module, layer: int):
        blocks = model.model.transformer.blocks
        if not 0 <= layer < len(blocks):
            raise ValueError(f"Layer {layer} outside [0,{len(blocks) - 1}]")
        self.block = blocks[layer]
        self.handle: Any = None
        self.config: Any = None
        self.armed = False
        self.applied_count = 0
        self.before: torch.Tensor | None = None
        self.after: torch.Tensor | None = None
        self.target: torch.Tensor | None = None
        self.edit_norm_ratio: torch.Tensor | None = None

    def configure(
        self,
        mode: str,
        positions: Sequence[Mapping[str, int]] | Sequence[int],
        axes: Sequence[int],
        value: float,
        factors: torch.Tensor,
        basis: torch.Tensor,
    ) -> None:
        if mode not in self.MODES:
            raise ValueError(f"Unknown coordinate edit {mode!r}")
        if not positions or len(positions) != len(axes):
            raise ValueError("Every row requires positions and an H/V axis")
        if any(int(axis) not in {0, 1} for axis in axes):
            raise ValueError("EmbSpatial causal edits only support H/V axes")
        if not math.isfinite(value):
            raise ValueError("Causal edit value must be finite")
        self.config = (
            mode,
            list(positions),
            list(map(int, axes)),
            float(value),
            factors,
            basis,
        )
        self.armed = True
        self.applied_count = 0
        self.before = self.after = self.target = self.edit_norm_ratio = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        del module, inputs
        if not self.armed:
            return output
        if self.config is None:
            raise RuntimeError("Coordinate edit is not configured")
        tensor = output[0] if isinstance(output, tuple) else output
        mode, positions, axes, value, factors, basis = self.config
        if tensor.ndim != 3 or tensor.shape[0] != len(positions):
            raise RuntimeError(f"Unexpected prefill activation {tuple(tensor.shape)}")

        activation = tensor.detach().float()
        modified = tensor.clone()
        before_rows: list[torch.Tensor] = []
        target_rows: list[torch.Tensor] = []
        ratios: list[torch.Tensor] = []
        token_positions_by_row: list[list[int]] = []
        for row, (position, axis) in enumerate(zip(positions, axes, strict=True)):
            if mode == "single_shift":
                token_positions = [int(position)]
            else:
                if not isinstance(position, Mapping):
                    raise TypeError("Relative edits require obj1/obj2 positions")
                token_positions = [int(position["obj1"]), int(position["obj2"])]
            if min(token_positions) < 0 or max(token_positions) >= tensor.shape[1]:
                raise RuntimeError("Object token is absent from the prefill sequence")

            before = torch.stack(
                [
                    factors @ activation[row, token_position]
                    for token_position in token_positions
                ]
            )
            target = before.clone()
            if mode == "relative_swap":
                target[0, axis] += value * (before[1, axis] - before[0, axis])
                target[1, axis] += value * (before[0, axis] - before[1, axis])
            else:
                target[:, axis] += value
            row_ratios = []
            for token_position, coordinate_delta in zip(
                token_positions,
                target - before,
                strict=True,
            ):
                hidden_delta = basis @ coordinate_delta
                modified[row, token_position] += hidden_delta.to(modified.dtype)
                row_ratios.append(
                    hidden_delta.norm() / activation[row, token_position].norm()
                )
            token_positions_by_row.append(token_positions)
            before_rows.append(before)
            target_rows.append(target)
            ratios.append(torch.stack(row_ratios))

        after_rows = [
            torch.stack(
                [
                    factors @ modified[row, token_position].float()
                    for token_position in token_positions
                ]
            )
            for row, token_positions in enumerate(token_positions_by_row)
        ]
        self.before = torch.stack(before_rows).detach().cpu()
        self.after = torch.stack(after_rows).detach().cpu()
        self.target = torch.stack(target_rows).detach().cpu()
        self.edit_norm_ratio = torch.stack(ratios).detach().cpu()
        error = float((self.after - self.target).abs().max())
        if error > 1e-4:
            raise ValueError(f"Causal coordinate-edit error is {error:.6g}")
        self.armed = False
        self.applied_count += 1
        return (modified, *output[1:]) if isinstance(output, tuple) else modified

    def mechanics(self, row: int, axis: int) -> dict[str, Any]:
        if any(
            value is None
            for value in (self.before, self.after, self.target, self.edit_norm_ratio)
        ):
            raise RuntimeError("Coordinate mechanics are unavailable")
        assert self.before is not None and self.after is not None
        assert self.target is not None and self.edit_norm_ratio is not None
        before, after, target = self.before[row], self.after[row], self.target[row]
        offaxes = [index for index in range(len(AXIS_ORDER)) if index != axis]
        delta = after - before
        result = {
            "coordinates_before_query": before[0].tolist(),
            "coordinates_after_query": after[0].tolist(),
            "coordinate_before_query": float(before[0, axis]),
            "coordinate_after_query": float(after[0, axis]),
            "coordinate_target_query": float(target[0, axis]),
            "achieved_change_query": float(delta[0, axis]),
            "coordinate_target_error": float((after - target).abs().max()),
            "offaxis_max_change": float(delta[:, offaxes].abs().max()),
            "edit_norm_ratio": float(self.edit_norm_ratio[row].max()),
        }
        if before.shape[0] == 2:
            result.update(
                {
                    "coordinates_before_reference": before[1].tolist(),
                    "coordinates_after_reference": after[1].tolist(),
                    "coordinate_before_reference": float(before[1, axis]),
                    "coordinate_after_reference": float(after[1, axis]),
                    "coordinate_target_reference": float(target[1, axis]),
                    "achieved_change_reference": float(delta[1, axis]),
                }
            )
        return result

    def __enter__(self) -> "CoordinateEditHook":
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
