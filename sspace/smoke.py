"""Run the deterministic CPU fixture through every core numerical contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .experiments.common.pairwise.decision import classify_pairwise_margin
from .experiments.intervention.spatial_causality.spatial_jacobian.spatial_causality import (
    coordinate_right_inverse,
)
from .experiments.multi_view.common.schema import MultiViewSample
from .experiments.multi_view.common.scoring import score_layerwise_multiview_order
from .experiments.spinbench.perspective_taking.adapter import (
    VIEW_ROTATIONS,
    rotate_coordinates,
    spinbench_target_margin,
)
from .experiments.validation.coco1800.selection.layers import (
    build_validation_evidence,
    select_default_layer,
    validation_metrics,
)
from .core.extraction.execution import LayerwiseFixedContrastGradient
from .core.extraction.merge import _project_object_states, _unit_axes
from .core.extraction.storage import ShardLayout, ShardStorage


PROTOCOL = "sspace_cpu_fixture_v1"


class _Scale(torch.nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.factor


class _ToyContrastModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = torch.nn.ModuleList((_Scale(2.0), _Scale(3.0)))
        self.register_buffer("direction", torch.tensor((1.0, -2.0)))

    def forward(
        self,
        hidden: torch.Tensor,
        use_cache: bool,
        return_dict: bool,
        logits_to_keep: int,
    ) -> SimpleNamespace:
        assert not use_cache and return_dict and logits_to_keep == 1
        value = hidden
        for block in self.blocks:
            value = block(value)
        score = (value[:, 0, :] - value[:, 1, :]) @ self.direction
        zero = torch.zeros_like(score)
        logits = torch.stack((zero, score, zero, 2.0 * score, zero, -score), dim=-1)
        return SimpleNamespace(logits=logits[:, None, :])


def toy_extraction() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model = _ToyContrastModel()
    hidden = torch.tensor((((2.0, 1.0), (1.0, 3.0), (0.0, 0.0)),))
    endpoints = {
        "horizontal": (0, 1),
        "vertical": (2, 3),
        "distance": (4, 5),
    }
    with LayerwiseFixedContrastGradient(model, model.blocks) as extractor:
        return extractor.extract(
            {"hidden": hidden},
            ({"query": 0, "reference": 1},),
            endpoints,
            "horizontal",
        )


def orientation_aggregate() -> dict[str, object]:
    gradients = np.asarray(
        (
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ),
        dtype=np.float32,
    )
    with tempfile.TemporaryDirectory() as directory:
        storage = ShardStorage(
            Path(directory),
            {"fixture": PROTOCOL},
            0,
            1,
            np.arange(3, dtype=np.int64),
            np.arange(3, dtype=np.int64),
            ShardLayout(3, 1, 3),
        )
        for group in range(3):
            records = tuple(
                {
                    "global_job_index": group,
                    "orientation": orientation,
                }
                for orientation in ("original", "swapped")
            )
            storage.append(
                np.zeros((2, 1, 2, 3), dtype=np.float32),
                gradients[group, :, None, :],
                np.zeros((2, 3), dtype=np.float32),
                group,
                records,
            )
        axes, layers, means = _unit_axes(storage.gradient_sums, storage.gradient_counts)
        counts = storage.gradient_counts.copy()
    expected_means = gradients.mean(axis=1)[:, None, :]
    np.testing.assert_allclose(means, expected_means, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(counts, np.asarray((2, 2, 2)))
    np.testing.assert_allclose(np.linalg.norm(axes, axis=-1), 1.0, atol=1e-7)
    return {
        "counts": counts.tolist(),
        "layers": layers.tolist(),
        "means": means.tolist(),
        "axes": axes.tolist(),
    }


def projection_contract() -> list[list[list[float]]]:
    states = np.asarray(((((2.0, 3.0), (0.0, 0.0)),),), dtype=np.float32)
    axes = np.asarray((((1.0, 0.0), (0.0, 1.0), (0.5, 0.5)),), dtype=np.float32)
    margins = _project_object_states(states, axes, (0,))
    np.testing.assert_allclose(margins, np.asarray((((2.0, 3.0, 2.5),),)))
    return margins.tolist()


def held_out_selection() -> dict[str, object]:
    specifications = (
        ("horizontal", "right", 1.0),
        ("horizontal", "left", -1.0),
        ("vertical", "below", 1.0),
        ("vertical", "above", -1.0),
        ("distance", "close", 1.0),
        ("distance", "far", -1.0),
    )
    groups: list[str] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    rows: list[tuple[float, float]] = []
    for index, (group, label, sign) in enumerate(specifications):
        for _ in range(5):
            groups.append(group)
            labels.append(label)
            sample_ids.append(f"sample-{index}")
            rows.append((sign, -sign))
    metrics = validation_metrics(np.asarray(rows), groups, labels, sample_ids, (3, 7))
    selected = select_default_layer(metrics)
    if selected != 3:
        raise AssertionError(f"held-out layer selection returned L{selected}")
    evidence = build_validation_evidence(
        "toy-held-out",
        "a" * 64,
        metrics,
        tuple(range(100, 106)),
        tuple(range(200, 206)),
    )
    if evidence["train_image_overlap_count"] != 0:
        raise AssertionError("construction and validation fixture IDs overlap")
    return {
        "selected_layer": selected,
        "per_layer_accuracy": [item.overall_accuracy for item in metrics],
        "overlap": evidence["train_image_overlap_count"],
    }


def pairwise_contract() -> dict[str, str]:
    result = {
        "horizontal_positive": classify_pairwise_margin("horizontal", 1.0),
        "vertical_negative": classify_pairwise_margin("vertical", -1.0),
        "distance_positive": classify_pairwise_margin("distance", 1.0),
        "tie": classify_pairwise_margin("horizontal", 0.0),
    }
    expected = {
        "horizontal_positive": "right",
        "vertical_negative": "above",
        "distance_positive": "close",
        "tie": "tie",
    }
    if result != expected:
        raise AssertionError(result)
    return result


def spinbench_contract() -> dict[str, object]:
    determinants = {}
    for name, rotation in VIEW_ROTATIONS.items():
        np.testing.assert_array_equal(rotation.T @ rotation, np.eye(3))
        determinants[name] = float(np.linalg.det(rotation))
        if round(determinants[name]) != 1:
            raise AssertionError(f"{name} is not a proper rotation")
    left = rotate_coordinates((0.0, 0.0, 2.0), "left")
    back = rotate_coordinates((2.0, 0.0, 0.0), "back")
    if spinbench_target_margin(left, "left") != -2.0:
        raise AssertionError("left-view target sign changed")
    if spinbench_target_margin(back, "left") != 2.0:
        raise AssertionError("back-view target sign changed")
    return {
        "determinants": determinants,
        "left_view": left.tolist(),
        "back_view": back.tolist(),
    }


def hstar_contract() -> dict[str, object]:
    specifications = (
        ("horizontal", "left", (-1.0, 0.0, 0.0)),
        ("horizontal", "right", (1.0, 0.0, 0.0)),
        ("vertical", "up", (0.0, -1.0, 0.0)),
        ("vertical", "down", (0.0, 1.0, 0.0)),
        ("distance", "far", (0.0, 0.0, -1.0)),
        ("distance", "close", (0.0, 0.0, 1.0)),
    )
    with tempfile.TemporaryDirectory() as directory:
        image_a = Path(directory) / "a.jpg"
        image_b = Path(directory) / "b.jpg"
        image_a.touch()
        image_b.touch()
        samples = tuple(
            MultiViewSample(
                sample_id=f"pair-{label}",
                scene="toy",
                object_id=index,
                object_name="object",
                axis=axis,
                split="test",
                view_a_observation_id="a",
                view_b_observation_id="b",
                image_a_path=image_a,
                image_b_path=image_b,
                image_a_sha256="a" * 64,
                image_b_sha256="b" * 64,
                gt_a_hvd=(0.0, 0.0, 0.0),
                gt_b_hvd=delta,
                gt_delta_hvd=delta,
                visibility_a=1.0,
                visibility_b=1.0,
                target_label=label,
            )
            for index, (axis, label, delta) in enumerate(specifications)
        )
        states = np.zeros((6, 2, 1, 3), dtype=np.float32)
        states[:, 1, 0] = np.asarray([item[2] for item in specifications])
        scores, metrics = score_layerwise_multiview_order(
            samples, states, np.eye(3, dtype=np.float32)[None], (0,)
        )
        reverse_scores, reverse_metrics = score_layerwise_multiview_order(
            samples, states[:, ::-1], np.eye(3, dtype=np.float32)[None], (0,)
        )
    if metrics[0]["overall"]["accuracy"] != 1.0:
        raise AssertionError("BA-minus-AB order failed")
    if reverse_metrics[0]["overall"]["accuracy"] != 0.0:
        raise AssertionError("AB-minus-BA reversal was not detected")
    return {
        "scores": scores[:, 0].tolist(),
        "reverse_scores": reverse_scores[:, 0].tolist(),
        "accuracy": metrics[0]["overall"]["accuracy"],
    }


def right_inverse_contract() -> dict[str, object]:
    factors = torch.tensor(
        ((1.0, 1.0, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0), (1.0, 0.0, 1.0, 1.0)),
        dtype=torch.float64,
    )
    basis = coordinate_right_inverse(factors)
    identity = factors @ basis
    torch.testing.assert_close(identity, torch.eye(3, dtype=torch.float64))
    hidden = torch.tensor((0.5, -1.0, 2.0, 0.25), dtype=torch.float64)
    delta = torch.tensor((0.2, -0.3, 0.4), dtype=torch.float64)
    torch.testing.assert_close(
        factors @ (hidden + basis @ delta), factors @ hidden + delta
    )
    return {"max_error": float((identity - torch.eye(3)).abs().max())}


def run_once(seed: int) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    contrasts, gradients, _ = toy_extraction()
    torch.testing.assert_close(contrasts, torch.tensor(((30.0, 60.0, -30.0),)))
    expected_gradient = torch.stack(
        (3.0 * torch.tensor((1.0, -2.0)), torch.tensor((1.0, -2.0)))
    )[None]
    torch.testing.assert_close(gradients, expected_gradient)
    return {
        "protocol": PROTOCOL,
        "seed": seed,
        "random_probe": np.random.uniform(-1.0, 1.0, 4).tolist(),
        "final_logit_contrasts": contrasts.tolist(),
        "role_gradients": gradients.tolist(),
        "orientation_aggregate": orientation_aggregate(),
        "pair_projection": projection_contract(),
        "held_out_selection": held_out_selection(),
        "pairwise": pairwise_contract(),
        "spinbench": spinbench_contract(),
        "hstar": hstar_contract(),
        "spatial_right_inverse": right_inverse_contract(),
    }


def fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 2:
        raise ValueError("--repeat must be at least two")
    payloads = [run_once(args.seed) for _ in range(args.repeat)]
    fingerprints = [fingerprint(payload) for payload in payloads]
    if len(set(fingerprints)) != 1:
        raise AssertionError("same-seed CPU fixture produced different fingerprints")
    result = {
        "protocol": PROTOCOL,
        "seed": args.seed,
        "repeat": args.repeat,
        "fingerprint": fingerprints[0],
        "checks": sorted(
            key
            for key in payloads[0]
            if key not in {"protocol", "seed", "random_probe"}
        ),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
