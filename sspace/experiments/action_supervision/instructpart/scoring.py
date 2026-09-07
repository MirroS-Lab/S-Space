"""Score the InstructPart action-supervision experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t

from sspace.run_records import atomic_json

from .config import DEFAULT_CONFIG, MODEL_ORDER, load_config, project_path
from .prompts import PROMPT_PROTOCOL, PROMPT_TEMPLATES


AXES = ("horizontal", "vertical", "distance")


def _load_ground_truth(path: Path) -> tuple[list[dict[str, Any]], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["rows"]
    ids = [int(row["showcase_id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Pseudo-GT rows require unique non-empty showcase IDs")
    return rows, str(value["review_status"])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float | None:
    if len(first) < 3:
        return None
    first_rank = _rank(first)
    second_rank = _rank(second)
    if first_rank.std() == 0 or second_rank.std() == 0:
        return None
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def _axis_values(rows: list[dict[str, Any]], axis: int) -> np.ndarray:
    key = ("horizontal_delta", "vertical_delta", "depth_gap_normalized")[axis]
    return np.asarray([float(row[key]) for row in rows], dtype=np.float64)


def _valid(rows: list[dict[str, Any]], axis: int) -> np.ndarray:
    key = ("horizontal_label", "vertical_label", "depth_label")[axis]
    return np.asarray(
        [bool(row["score_candidate"]) and row[key] != "ambiguous" for row in rows]
    )


def signed_correct(margins: np.ndarray, ground_truth: np.ndarray) -> np.ndarray:
    """Return sign agreement, counting an exact zero margin as incorrect."""
    return ((margins > 0) & (ground_truth > 0)) | (
        (margins < 0) & (ground_truth < 0)
    )


def _layer_metrics(
    margins: np.ndarray,
    rows: list[dict[str, Any]],
    layer_ids: list[int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if margins.shape != (len(rows), len(layer_ids), 3):
        raise ValueError(f"Unexpected margin shape {margins.shape}")
    correctness = np.zeros_like(margins, dtype=bool)
    metrics = []
    for layer_position, layer_id in enumerate(layer_ids):
        axis_rows = []
        combined = []
        for axis, axis_name in enumerate(AXES):
            values = _axis_values(rows, axis)
            valid = _valid(rows, axis)
            correct = signed_correct(margins[:, layer_position, axis], values)
            correctness[:, layer_position, axis] = correct
            combined.append(correct[valid])
            axis_rows.append(
                {
                    "axis": axis_name,
                    "correct": int(correct[valid].sum()),
                    "total": int(valid.sum()),
                    "accuracy": float(correct[valid].mean()),
                    "spearman": _spearman(
                        margins[valid, layer_position, axis], values[valid]
                    ),
                }
            )
        pooled = np.concatenate(combined)
        metrics.append(
            {
                "layer_id": int(layer_id),
                "overall_correct": int(pooled.sum()),
                "overall_total": int(len(pooled)),
                "overall_accuracy": float(pooled.mean()),
                "worst_axis_accuracy": float(
                    min(row["accuracy"] for row in axis_rows)
                ),
                "per_axis": axis_rows,
            }
        )
    return metrics, correctness


def _best(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        metrics,
        key=lambda row: (
            row["overall_accuracy"],
            row["worst_axis_accuracy"],
            -row["layer_id"],
        ),
    )


def _selection_positions(records_path: Path, rows: list[dict[str, Any]]) -> np.ndarray:
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    positions = {
        int(record["showcase_id"]): int(record["coordinate_index"])
        for record in records
    }
    if len(positions) != len(records):
        raise ValueError(f"Duplicate showcase IDs in {records_path}")
    ground_truth_ids = [int(row["showcase_id"]) for row in rows]
    if set(positions) != set(ground_truth_ids):
        raise ValueError("Projection and pseudo-GT showcase IDs differ")
    return np.asarray([positions[value] for value in ground_truth_ids], dtype=np.int64)


def _aggregate(
    correctness: np.ndarray,
    valid: np.ndarray,
    template_positions: list[int],
    layer_position: int,
) -> dict[str, Any]:
    selected = correctness[template_positions, :, layer_position, :]
    selected_valid = np.broadcast_to(valid, selected.shape)
    per_axis = []
    for axis, axis_name in enumerate(AXES):
        axis_correct = selected[:, :, axis]
        axis_valid = selected_valid[:, :, axis]
        per_axis.append(
            {
                "axis": axis_name,
                "correct": int(axis_correct[axis_valid].sum()),
                "total": int(axis_valid.sum()),
                "accuracy": float(axis_correct[axis_valid].mean()),
            }
        )
    return {
        "template_count": len(template_positions),
        "overall_correct": int(selected[selected_valid].sum()),
        "overall_total": int(selected_valid.sum()),
        "overall_accuracy": float(selected[selected_valid].mean()),
        "worst_axis_accuracy": float(min(row["accuracy"] for row in per_axis)),
        "per_axis": per_axis,
    }


def _prompt_summary(per_template: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize accuracy across prompts, not across sampled images."""
    count = len(per_template)
    critical = float(t.ppf(0.975, df=count - 1))

    def summarize(values: list[float]) -> dict[str, Any]:
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        sem = std / np.sqrt(count)
        half_width = float(critical * sem)
        return {
            "mean_accuracy": mean,
            "sample_std": std,
            "standard_error": float(sem),
            "ci95": [mean - half_width, mean + half_width],
        }

    return {
        "method": "student_t_across_prompts",
        "confidence_level": 0.95,
        "template_count": count,
        "degrees_of_freedom": count - 1,
        "overall": summarize([row["overall_accuracy"] for row in per_template]),
        "per_axis": [
            {
                "axis": axis,
                **summarize([
                    next(item["accuracy"] for item in row["per_axis"] if item["axis"] == axis)
                    for row in per_template
                ]),
            }
            for axis in AXES
        ],
    }


def _paired_bootstrap(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
    seed: int = 42,
    repeats: int = 10_000,
) -> dict[str, Any]:
    eligible = np.flatnonzero(valid.any(axis=1))
    if not len(eligible):
        raise ValueError("Paired bootstrap has no valid image samples")
    reference_accuracy = float(reference[valid].mean())
    candidate_accuracy = float(candidate[valid].mean())
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.choice(eligible, size=len(eligible), replace=True)
        sampled_valid = valid[indices]
        differences[repeat] = (
            candidate[indices][sampled_valid].mean()
            - reference[indices][sampled_valid].mean()
        )
    return {
        "reference_accuracy": reference_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "candidate_minus_reference": candidate_accuracy - reference_accuracy,
        "sample_bootstrap_95_ci": [
            float(np.quantile(differences, 0.025)),
            float(np.quantile(differences, 0.975)),
        ],
        "candidate_only_correct": int((candidate[valid] & ~reference[valid]).sum()),
        "reference_only_correct": int((reference[valid] & ~candidate[valid]).sum()),
        "bootstrap_repeats": repeats,
    }


def _comparison(
    reference: np.ndarray,
    candidate: np.ndarray,
    valid: np.ndarray,
) -> dict[str, Any]:
    sample_count, template_count, axis_count = reference.shape
    expected = (sample_count, template_count, axis_count)
    if candidate.shape != expected or axis_count != len(AXES):
        raise ValueError("Model correctness tensors are incompatible")
    repeated_valid = np.broadcast_to(
        valid[:, None, :], (sample_count, template_count, axis_count)
    )
    output = {
        "overall": _paired_bootstrap(
            reference.reshape(sample_count, -1),
            candidate.reshape(sample_count, -1),
            repeated_valid.reshape(sample_count, -1),
        )
    }
    for axis, axis_name in enumerate(AXES):
        output[axis_name] = _paired_bootstrap(
            reference[:, :, axis],
            candidate[:, :, axis],
            repeated_valid[:, :, axis],
        )
    return output


def score(
    pseudo_ground_truth: Path,
    projection_root: Path,
    output: Path,
    requested_layers: list[int],
    limit: int | None = None,
) -> None:
    """Score all three model directories and write one provisional report."""
    rows, review_status = _load_ground_truth(pseudo_ground_truth)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[:limit]
    valid = np.stack([_valid(rows, axis) for axis in range(3)], axis=1)
    groups = {
        "all_templates": list(range(len(PROMPT_TEMPLATES))),
        "context_free": list(range(5)),
        "action_context": list(range(5, 10)),
    }
    model_results: dict[str, Any] = {}
    model_correctness: dict[str, dict[int, np.ndarray]] = {}
    expected_templates = [
        {
            "template_id": template.template_id,
            "group": template.group,
            "text": template.text,
        }
        for template in PROMPT_TEMPLATES
    ]
    for model_name in MODEL_ORDER:
        model_dir = projection_root / model_name
        summary_path = model_dir / "summary.json"
        margin_path = model_dir / "template_part_minus_object.npy"
        config_path = model_dir / "resolved_config.json"
        records_path = model_dir / "records.jsonl"
        for path in (summary_path, margin_path, config_path, records_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            model_config.get("protocol") != PROMPT_PROTOCOL
            or model_config.get("templates") != expected_templates
        ):
            raise ValueError(f"{model_name} prompt protocol differs")
        layer_ids = [int(value) for value in model_config["layer_ids"]]
        missing_layers = sorted(set(requested_layers) - set(layer_ids))
        if missing_layers:
            raise ValueError(f"{model_name} lacks layers {missing_layers}")
        positions = _selection_positions(records_path, rows)
        margins = np.load(margin_path, mmap_mode="r")[positions]
        expected_shape = (len(rows), len(PROMPT_TEMPLATES), len(layer_ids), 3)
        if margins.shape != expected_shape or not np.isfinite(margins).all():
            raise ValueError(f"{model_name} margin array is invalid")
        template_metrics = []
        correctness_parts = []
        for template_position, metadata in enumerate(expected_templates):
            metrics, correctness = _layer_metrics(
                margins[:, template_position], rows, layer_ids
            )
            template_metrics.append(
                {
                    **metadata,
                    "requested_layers": [
                        row for row in metrics if row["layer_id"] in requested_layers
                    ],
                }
            )
            correctness_parts.append(correctness)
        correctness = np.stack(correctness_parts, axis=0)
        requested_results = []
        model_correctness[model_name] = {}
        for layer_id in requested_layers:
            layer_position = layer_ids.index(layer_id)
            group_results = {
                name: _aggregate(correctness, valid, indices, layer_position)
                for name, indices in groups.items()
            }
            per_template = []
            for position, metadata in enumerate(expected_templates):
                metric = next(
                    row
                    for row in template_metrics[position]["requested_layers"]
                    if row["layer_id"] == layer_id
                )
                per_template.append({**metadata, **metric})
            requested_results.append(
                {
                    "layer_id": layer_id,
                    "groups": group_results,
                    "per_template": per_template,
                    "across_prompt_summary": _prompt_summary(per_template),
                }
            )
            model_correctness[model_name][layer_id] = np.transpose(
                correctness[:, :, layer_position, :], (1, 0, 2)
            )
        diagnostic_rows = [
            {
                "layer_id": row["layer_id"],
                "overall_accuracy": row["groups"]["all_templates"][
                    "overall_accuracy"
                ],
                "worst_axis_accuracy": row["groups"]["all_templates"][
                    "worst_axis_accuracy"
                ],
            }
            for row in requested_results
        ]
        model_results[model_name] = {
            "requested_layers": requested_results,
            "diagnostic_best_within_requested_sweep": _best(diagnostic_rows),
            "template_metrics": template_metrics,
        }
    comparisons = {}
    reference = model_correctness["molmo2_er"]
    for candidate_name in ("molmoact2_pretrain", "molmoact2"):
        comparisons[candidate_name] = {
            str(layer_id): _comparison(
                reference[layer_id], model_correctness[candidate_name][layer_id], valid
            )
            for layer_id in requested_layers
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(
        output,
        {
            "status": "complete" if review_status == "passed" else "provisional",
            "protocol": PROMPT_PROTOCOL,
            "ground_truth_review_status": review_status,
            "sample_count": len(rows),
            "template_count": len(PROMPT_TEMPLATES),
            "requested_layers": requested_layers,
            "axis_valid_counts": {
                AXES[axis]: int(valid[:, axis].sum()) for axis in range(3)
            },
            "model_results": model_results,
            "paired_comparisons_vs_molmo2_er": comparisons,
            "interpretation": {
                "model_margin": "axis @ (part target state - object target state)",
                "ground_truth": "part coordinate - whole-object coordinate",
                "positive_endpoints": ["right", "below", "close"],
                "zero_margin": "incorrect",
                "template_weighting": "each template has equal weight",
                "across_prompt_ci": "per-model mean +/- Student-t(0.975, T-1) * sample_std / sqrt(T); accuracy units; no clipping",
                "bootstrap_unit": "image; all ten templates remain grouped",
                "layer_sweep": "diagnostic only; not validation-selected",
                "distance_ground_truth": "Depth Anything V2 inverse-depth pseudo-label",
            },
        },
    )
    print(
        f"scored {len(model_results)} models at {requested_layers}; "
        f"review={review_status}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--projection-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    dataset_root = project_path(config["dataset"]["dataset_root"])
    projection_root = (
        args.projection_root.resolve()
        if args.projection_root is not None
        else project_path(config["evaluation"]["output_root"])
    )
    output = args.output.resolve() if args.output is not None else projection_root / "results.json"
    score(
        dataset_root / "pseudo_ground_truth/pseudo_ground_truth.json",
        projection_root,
        output,
        [int(layer) for layer in config["evaluation"]["layers"]],
        args.limit,
    )


if __name__ == "__main__":
    main()
