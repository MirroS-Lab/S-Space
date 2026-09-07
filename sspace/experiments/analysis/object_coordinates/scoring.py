"""Model-independent continuous object-coordinate readout utilities."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from sspace.core.prompts.templates import AXIS_ORDER

SINGLE_POINT_COLUMNS = (
    "dataset",
    "model",
    "pair_id",
    "layer",
    "axis",
    "role",
    "ground_truth",
    "projection",
)


def _require_columns(frame: pd.DataFrame, required: Iterable[str]) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Coordinate table is missing columns: {missing}")


def _require_finite(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Coordinate column {column!r} must be finite")


def _require_nonempty_text(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        values = frame[column]
        if (
            values.isna().any()
            or not values.map(
                lambda value: isinstance(value, str) and bool(value.strip())
            ).all()
        ):
            raise ValueError(f"Coordinate column {column!r} must contain text")


def center_unit_interval(values: Iterable[float]) -> np.ndarray:
    """Map normalized image coordinates from ``[0, 1]`` to ``[-1, 1]``.

    Args:
        values: One-dimensional finite normalized image coordinates.

    Returns:
        A new float64 array whose image midpoint is zero.

    Raises:
        ValueError: The input is not one-dimensional, finite, or inside
            ``[0, 1]``.

    Side effects:
        None.
    """
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("Normalized coordinates must be a finite [N] array")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("Normalized coordinates must lie inside [0,1]")
    return 2.0 * array - 1.0


def validate_single_object_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate standardized single-object coordinate records (EVAL-08).

    Every pair/layer/axis group must contain exactly one ``target`` and one
    ``reference`` row. The function deliberately does not infer layers,
    recenter distance coordinates, or introduce a decision boundary.

    Args:
        frame: Table with ``SINGLE_POINT_COLUMNS``.

    Returns:
        A defensive copy sorted by dataset, model, pair, layer, axis, and role.

    Raises:
        ValueError: Schema, axis, role, uniqueness, or finite checks fail.

    Side effects:
        None.
    """
    _require_columns(frame, SINGLE_POINT_COLUMNS)
    result = frame.loc[:, SINGLE_POINT_COLUMNS].copy()
    if result.empty:
        raise ValueError("Single-object coordinate table cannot be empty")
    _require_nonempty_text(
        result,
        ("dataset", "model", "pair_id", "axis", "role"),
    )
    unknown_axes = sorted(set(result["axis"]) - set(AXIS_ORDER))
    if unknown_axes:
        raise ValueError(f"Unknown coordinate axes: {unknown_axes}")
    unknown_roles = sorted(set(result["role"]) - {"target", "reference"})
    if unknown_roles:
        raise ValueError(f"Unknown object roles: {unknown_roles}")
    _require_finite(result, ("layer", "ground_truth", "projection"))
    layers = pd.to_numeric(result["layer"], errors="raise").to_numpy(float)
    if not np.equal(layers, np.floor(layers)).all():
        raise ValueError("Coordinate layer IDs must be integers")
    result["layer"] = layers.astype(int)
    keys = ["dataset", "model", "pair_id", "layer", "axis", "role"]
    if result.duplicated(keys).any():
        raise ValueError("Single-object coordinate keys must be unique")
    counts = result.groupby(keys[:-1], sort=False)["role"].nunique()
    if not counts.eq(2).all():
        raise ValueError("Every pair/layer/axis must contain target and reference")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def derive_pair_difference_points(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute matched target-minus-reference coordinates and projections.

    For every pair, layer, and axis, the returned rows contain
    ``delta_c = c_target - c_reference`` and
    ``delta_m = v_g^T(h_target - h_reference)``. No swapped prompt or
    classification threshold is used.

    Args:
        frame: Valid or unvalidated single-object coordinate records.

    Returns:
        One row per dataset/model/pair/layer/axis group.

    Raises:
        ValueError: Input validation or role matching fails.

    Side effects:
        None.
    """
    points = validate_single_object_points(frame)
    keys = ["dataset", "model", "pair_id", "layer", "axis"]
    target = points[points.role.eq("target")].drop(columns="role")
    reference = points[points.role.eq("reference")].drop(columns="role")
    merged = target.merge(
        reference,
        on=keys,
        suffixes=("_target", "_reference"),
        validate="one_to_one",
    )
    result = merged[keys].copy()
    result["role"] = "target-reference"
    result["ground_truth"] = (
        merged["ground_truth_target"] - merged["ground_truth_reference"]
    )
    result["projection"] = merged["projection_target"] - merged["projection_reference"]
    return result


def _rank_average(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average").to_numpy(float)


def fit_affine_readout(
    coordinates: Iterable[float], projections: Iterable[float]
) -> dict[str, float | int]:
    """Fit and score ``projection = intercept + slope * coordinate``.

    Args:
        coordinates: External ground-truth coordinates with shape ``[N]``.
        projections: Matched model-native projections with shape ``[N]``.

    Returns:
        Sample count, affine slope and intercept, Pearson ``r``, Spearman
        ``rho``, and affine ``R^2``.

    Raises:
        ValueError: Arrays differ, contain fewer than three points, are
            non-finite, or have zero variance.

    Side effects:
        None.
    """
    x = np.asarray(tuple(coordinates), dtype=np.float64)
    y = np.asarray(tuple(projections), dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("Coordinate and projection arrays must match [N]")
    if x.size < 3 or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Affine readout requires at least three finite pairs")
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        raise ValueError("Affine readout requires nonzero variance")
    design = np.column_stack((np.ones(x.size), x))
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = intercept + slope * x
    residual = float(np.sum((y - predicted) ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(_rank_average(x), _rank_average(y))[0, 1])
    return {
        "n": int(x.size),
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "r2": float(1.0 - residual / total),
    }


def summarize_continuous_readouts(
    single_points: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize pooled, role-specific, and pair-difference affine fits.

    The pooled ``single_object`` row is the primary absolute-coordinate
    readout. Role-specific rows remain in the summary for audit purposes and
    use ``single_object_role``. ``pair_difference`` rows report the matched
    target-minus-reference relation.

    Args:
        single_points: Standardized single-object records.

    Returns:
        ``(fit_summary, pair_points)``.

    Raises:
        ValueError: Record validation or affine-fit requirements fail.

    Side effects:
        None.
    """
    points = validate_single_object_points(single_points)
    pair_points = derive_pair_difference_points(points)
    base_keys = ["dataset", "model", "layer", "axis"]
    rows: list[dict[str, object]] = []

    for key, part in points.groupby(base_keys, sort=False):
        rows.append(
            {
                **dict(zip(base_keys, key)),
                "role": "all-objects",
                "readout": "single_object",
                **fit_affine_readout(part.ground_truth, part.projection),
            }
        )

    for key, part in points.groupby(base_keys + ["role"], sort=False):
        rows.append(
            {
                **dict(zip(base_keys + ["role"], key)),
                "readout": "single_object_role",
                **fit_affine_readout(part.ground_truth, part.projection),
            }
        )

    for key, part in pair_points.groupby(base_keys + ["role"], sort=False):
        rows.append(
            {
                **dict(zip(base_keys + ["role"], key)),
                "readout": "pair_difference",
                **fit_affine_readout(part.ground_truth, part.projection),
            }
        )

    return pd.DataFrame(rows), pair_points
