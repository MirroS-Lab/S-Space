"""Pure statistical summaries for contextual S-Space case records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

AXES = ("horizontal", "vertical", "distance")


def load_case_records(path: Path) -> list[dict[str, Any]]:
    """Load unique complete JSONL projection records without mutation."""
    records = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or len({row["case_id"] for row in records}) != len(records):
        raise ValueError("Case records must contain unique non-empty case IDs")
    return records


def flatten_case_records(records: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Flatten role projections and metadata into one auditable table."""
    rows = []
    for record in records:
        row = {
            key: record[key]
            for key in (
                "case_id",
                "theme",
                "family",
                "condition",
                "prompt",
                "selected_layer",
            )
        }
        row.update({f"meta_{key}": value for key, value in record["metadata"].items()})
        for role, values in record["role_projections"].items():
            for axis in AXES:
                row[f"{role}_{axis}"] = float(values[axis])
        pair = record.get("target_minus_reference")
        if pair is not None:
            for axis in AXES:
                row[f"pair_{axis}"] = float(pair[axis])
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(
    values: Sequence[float], seed: int, draws: int
) -> tuple[float, float, float]:
    """Return mean and deterministic paired-bootstrap 95% interval."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2 or not np.isfinite(array).all():
        raise ValueError("Bootstrap requires at least two finite observations")
    if draws <= 0:
        raise ValueError("Bootstrap draws must be positive")
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(draws, array.size), replace=True).mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def _uniform_metadata(part: pd.DataFrame, column: str, default: Any) -> Any:
    """Resolve one family-wide declaration, including legacy defaults."""
    if column not in part:
        return default
    values = [
        default
        if value is None or (isinstance(value, float) and np.isnan(value))
        else value
        for value in part[column]
    ]
    if any(value != values[0] for value in values[1:]):
        raise ValueError(f"{part.family.iloc[0]} has inconsistent {column}")
    return values[0]


def _beyond_visible_groups(frame: pd.DataFrame):
    """Validate complete prompt/condition blocks and their declared contrast."""
    required = {"theme", "family", "condition", "meta_prompt_key", "meta_effect_sign"}
    if not required.issubset(frame):
        raise ValueError(f"Beyond-visible records require {sorted(required)}")
    data = frame[frame.theme.eq("beyond_visible")].copy()
    if data.empty:
        raise ValueError("Beyond-visible records are empty")
    for column in ("family", "condition", "meta_prompt_key"):
        if (
            not data[column]
            .map(lambda value: isinstance(value, str) and bool(value.strip()))
            .all()
        ):
            raise ValueError(f"Beyond-visible {column} must contain non-empty text")
    for family, part in data.groupby("family", sort=True):
        axis = _uniform_metadata(part, "meta_effect_axis", "horizontal")
        if axis not in AXES:
            raise ValueError(f"{family} has invalid effect axis {axis!r}")
        condition_groups = []
        for side, default in (("a", "original"), ("b", "mirror")):
            conditions = _uniform_metadata(part, f"meta_effect_conditions_{side}", [])
            if not isinstance(conditions, (list, tuple)):
                raise ValueError(f"{family} condition group {side} must be an array")
            if not conditions:
                conditions = [
                    _uniform_metadata(part, f"meta_effect_condition_{side}", default)
                ]
            if any(
                not isinstance(value, str) or not value.strip() for value in conditions
            ) or len(set(conditions)) != len(conditions):
                raise ValueError(
                    f"{family} condition group {side} must contain unique names"
                )
            condition_groups.append(tuple(conditions))
        conditions_a, conditions_b = condition_groups
        if set(conditions_a) & set(conditions_b):
            raise ValueError(f"{family} contrast condition groups must be disjoint")
        available = set(part.condition)
        if not set(conditions_a + conditions_b).issubset(available):
            raise ValueError(f"{family} lacks declared contrast conditions")
        if part.duplicated(["meta_prompt_key", "condition"]).any():
            raise ValueError(f"{family} has duplicate prompt/condition rows")
        prompt_groups = part.groupby("meta_prompt_key", sort=False)
        if any(set(group.condition) != available for _, group in prompt_groups):
            raise ValueError(
                f"{family} must contain complete matched conditions per prompt"
            )
        if not part.meta_effect_sign.isin((-1, 1)).all():
            raise ValueError(f"{family} effect signs must be -1 or 1")
        if any(group.meta_effect_sign.nunique() != 1 for _, group in prompt_groups):
            raise ValueError(f"{family} prompt pairs must share one effect sign")
        if f"target_{axis}" not in part:
            raise ValueError(f"{family} requires target_{axis} projections")
        for column in (f"target_{axis}", f"pair_{axis}"):
            if column in part and not np.isfinite(part[column].to_numpy(float)).all():
                raise ValueError(f"{family} {column} projections must be finite")
        yield family, part, axis, conditions_a, conditions_b


def summarize_beyond_visible(
    frame: pd.DataFrame, seed: int = 42, draws: int = 20_000
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute signed differences between prompt-matched condition means.

    Legacy records use horizontal ``effect_sign * (original - mirror)``.
    Declared axes and condition groups support additional controls and the
    train distance contrast ``blank - mean(original, mirror)``. Conditions
    are averaged within each prompt before bootstrapping across prompts.
    """
    rows = []
    points = []
    for family, part, axis, conditions_a, conditions_b in _beyond_visible_groups(frame):
        for readout, column in (
            ("raw_target", f"target_{axis}"),
            ("target_minus_reference", f"pair_{axis}"),
        ):
            if column not in part:
                continue
            table = part.pivot(
                index="meta_prompt_key", columns="condition", values=column
            )
            signs = (
                part.drop_duplicates("meta_prompt_key")
                .set_index("meta_prompt_key")["meta_effect_sign"]
                .loc[table.index]
                .to_numpy(float)
            )
            effects = signs * (
                table.loc[:, list(conditions_a)].mean(axis=1).to_numpy()
                - table.loc[:, list(conditions_b)].mean(axis=1).to_numpy()
            )
            contrast = {
                "effect_axis": axis,
                "conditions_a": "+".join(conditions_a),
                "conditions_b": "+".join(conditions_b),
            }
            mean, low, high = bootstrap_mean_ci(effects, seed, draws)
            rows.append(
                {
                    "family": family,
                    "readout": readout,
                    **contrast,
                    "n_prompts": len(effects),
                    "expected_effect_mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                    "expected_direction_count": int((effects > 0).sum()),
                }
            )
            points.extend(
                {
                    "family": family,
                    "readout": readout,
                    "prompt_key": prompt_key,
                    **contrast,
                    "effect": float(effect),
                }
                for prompt_key, effect in zip(table.index, effects, strict=True)
            )
    return pd.DataFrame(rows), pd.DataFrame(points)


def summarize_beyond_visible_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    """Report each condition's mean, SD, and normal-approximation 95% half-width."""
    rows = []
    for family, part, axis, _, _ in _beyond_visible_groups(frame):
        for readout, column in (
            ("raw_target", f"target_{axis}"),
            ("target_minus_reference", f"pair_{axis}"),
        ):
            if column not in part:
                continue
            for condition, values in part.groupby("condition", sort=True)[column]:
                array = values.to_numpy(float)
                std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
                rows.append(
                    {
                        "family": family,
                        "readout": readout,
                        "condition": condition,
                        "effect_axis": axis,
                        "n": len(array),
                        "mean": float(array.mean()),
                        "std": std,
                        "ci95_half_width": 1.96 * std / np.sqrt(len(array)),
                    }
                )
    return pd.DataFrame(rows)


def summarize_web_political(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize the web's three fixed-prompt identity readouts within one model."""
    words = ["socialist", "moderate", "conservative"]
    data = frame[frame.family.eq("political_semantics")].copy()
    if (
        len(data) != 3
        or set(data.meta_target_word) != set(words)
        or data.selected_layer.nunique() != 1
        or not data.condition.eq("parliament").all()
    ):
        raise ValueError("Web political analysis requires exactly three identities at one layer")
    data = data.set_index("meta_target_word").loc[words].reset_index()
    values = data.target_horizontal.to_numpy(float)
    if not np.isfinite(values).all() or values.std(ddof=0) == 0:
        raise ValueError("Web political readouts must be finite with nonzero variance")
    data["within_model_zscore"] = (values - values.mean()) / values.std(ddof=0)
    return data


def summarize_political(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute socialist-minus-control effects within matched templates."""
    data = frame[
        frame.theme.eq("language_supervision") & frame.family.eq("political_semantics")
    ].copy()
    required = {
        "meta_target_word",
        "meta_template_index",
        "meta_scene_condition",
        "target_horizontal",
        "pair_horizontal",
    }
    if data.empty or not required.issubset(data):
        raise ValueError(f"Political records require {sorted(required)}")
    single = (
        data.groupby(["meta_scene_condition", "meta_target_word"], as_index=False)
        .agg(
            n=("case_id", "size"),
            mean=("target_horizontal", "mean"),
            std=("target_horizontal", "std"),
        )
        .rename(
            columns={
                "meta_scene_condition": "condition",
                "meta_target_word": "word",
            }
        )
    )
    single["ci95_half_width"] = 1.96 * single["std"] / np.sqrt(single["n"])
    rows = []
    for condition, part in data.groupby("meta_scene_condition", sort=True):
        for readout, column in (
            ("single_token", "target_horizontal"),
            ("vs_centrist", "pair_horizontal"),
        ):
            table = part.pivot(
                index="meta_template_index", columns="meta_target_word", values=column
            )
            for control in ("conservative", "journalist"):
                effect = table.socialist - table[control]
                rows.append(
                    {
                        "condition": condition,
                        "readout": readout,
                        "contrast": f"socialist_minus_{control}",
                        "n": len(effect),
                        "mean": float(effect.mean()),
                        "ci95_half_width": float(
                            1.96 * effect.std(ddof=1) / np.sqrt(len(effect))
                        ),
                        "left_fraction": float((effect < 0).mean()),
                    }
                )
    return single, pd.DataFrame(rows)
