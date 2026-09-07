"""Render release-style single-object and pair-difference scatter plots.

The utility consumes standardized EVAL-08 point records. It never runs a
model, fits an S-Space axis, selects a layer, or classifies a pairwise margin.
Every displayed model/layer pair must be declared explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FuncFormatter

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sspace.experiments.analysis.object_coordinates.scoring import (  # noqa: E402
    fit_affine_readout,
    summarize_continuous_readouts,
    validate_single_object_points,
)
from sspace.run_records import file_sha256  # noqa: E402
from sspace.core.prompts.templates import AXIS_ORDER  # noqa: E402

AXIS_DISPLAY = {
    "horizontal": ("Horizontal", "H", "right"),
    "vertical": ("Vertical", "V", "below"),
    "distance": ("Distance", "D", "close"),
}
POINT_COLOR = "#ffa985"
PAIR_POINT_COLOR = "#ffa985"
FIT_COLOR = "#8d70ff"
BACKGROUND_COLOR = "#f7f6fd"
MAJOR_GRID_COLOR = "#d8cff6"
MINOR_GRID_COLOR = "#ece7fa"
AXIS_COLOR = "#29272e"


def load_points(
    path: Path, input_format: str
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load a standard point table or the frozen historical COCO export.

    The standard schema is assumed to contain analysis-ready coordinates.
    The explicit legacy adapter adds ``dataset=COCO-2017-val``, maps H/V from
    ``[0,1]`` to ``[-1,1]``, and median-centers distance proximity. These
    transformations are recorded in the returned metadata.
    """
    points = pd.read_csv(path)
    metadata: dict[str, object] = {"input_format": input_format}
    if input_format == "coco_absolute_object_readouts_v1":
        if "dataset" in points.columns:
            raise ValueError(
                "Legacy COCO input unexpectedly contains dataset; use standard_v1"
            )
        points = points.assign(dataset="COCO-2017-val")
        horizontal_vertical = points["axis"].isin(("horizontal", "vertical"))
        points.loc[horizontal_vertical, "ground_truth"] = (
            2.0 * points.loc[horizontal_vertical, "ground_truth"] - 1.0
        )
        distance = points["axis"].eq("distance")
        if not distance.any():
            raise ValueError("Legacy COCO export must contain distance points")
        unique_distance = points.loc[
            distance, ["pair_id", "role", "ground_truth"]
        ].drop_duplicates()
        distance_median = float(unique_distance["ground_truth"].median())
        points.loc[distance, "ground_truth"] -= distance_median
        metadata["coordinate_transform"] = {
            "horizontal_vertical": "2*c-1",
            "distance": "d-median(d)",
            "distance_median": distance_median,
        }
    elif input_format == "standard_v1":
        metadata["coordinate_transform"] = "none; input is analysis-ready"
    else:
        raise ValueError(f"Unsupported input format: {input_format!r}")
    return validate_single_object_points(points), metadata


def parse_selection(values: list[str]) -> tuple[tuple[str, int], ...]:
    """Parse repeated ``MODEL=LAYER`` declarations without inferred defaults."""
    selections: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError("Each --selection must use MODEL=LAYER")
        model, layer_text = value.rsplit("=", 1)
        if not model.strip():
            raise ValueError("Selection model name cannot be empty")
        try:
            layer = int(layer_text)
        except ValueError as error:
            raise ValueError(f"Invalid selected layer in {value!r}") from error
        selections.append((model, layer))
    if len(set(selections)) != len(selections):
        raise ValueError("Selected model/layer pairs must be unique")
    return tuple(selections)


def select_points(
    points: pd.DataFrame, selections: tuple[tuple[str, int], ...]
) -> pd.DataFrame:
    """Return only explicitly selected model/layer points, failing closed."""
    available = set(zip(points.model, points.layer, strict=True))
    missing = sorted(set(selections) - available)
    if missing:
        raise ValueError(f"Selected model/layer pairs are absent: {missing}")
    selected = pd.concat(
        [
            points[points.model.eq(model) & points.layer.eq(layer)]
            for model, layer in selections
        ],
        ignore_index=True,
    )
    datasets = sorted(selected.dataset.unique())
    if len(datasets) != 1:
        raise ValueError(
            "Each plotting run must contain exactly one dataset; found "
            f"{datasets}. Run the tool once per model/dataset comparison."
        )
    for model, layer in selections:
        available_axes = set(
            selected[selected.model.eq(model) & selected.layer.eq(layer)].axis
        )
        if available_axes != set(AXIS_ORDER):
            raise ValueError(
                f"Selection {model}=L{layer} must contain exactly {AXIS_ORDER}; "
                f"found {sorted(available_axes)}"
            )
    return selected


def _format_tick(value: float, _position: float | None = None) -> str:
    """Format chart ticks with at most two decimals and no trailing zeros."""
    if np.isclose(value, 0.0, atol=5e-12):
        return "0"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def _nice_symmetric_scale(
    values: np.ndarray,
) -> tuple[tuple[float, float], np.ndarray]:
    """Return symmetric limits and five presentation-friendly major ticks."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Plot limits require finite values")
    max_abs = max(float(np.max(np.abs(finite))), 1e-12)
    magnitude = 10.0 ** np.floor(np.log10(max_abs))
    normalized = max_abs / magnitude
    candidates = np.array((1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0))
    bound = float(candidates[np.searchsorted(candidates, normalized)] * magnitude)
    limits = (-bound, bound)
    return limits, np.linspace(limits[0], limits[1], 5)


def _plot_grid(
    points: pd.DataFrame,
    selections: tuple[tuple[str, int], ...],
    readout: str,
    output: Path,
    svg_output: Path | None = None,
) -> None:
    """Plot one row per selected model and one column per spatial axis."""
    if readout not in {"single_object", "pair_difference"}:
        raise ValueError(f"Unsupported readout: {readout!r}")
    fig, panels = plt.subplots(
        len(selections),
        len(AXIS_ORDER),
        figsize=(15.5, 5.25 * len(selections)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    figure_letter = "a" if readout == "single_object" else "b"
    fig.text(
        0.012,
        0.978,
        figure_letter,
        ha="left",
        va="top",
        fontsize=28,
        fontweight="bold",
        color="#111111",
    )
    y_scale_by_model = {
        (model, layer): _nice_symmetric_scale(
            points.loc[
                points.model.eq(model) & points.layer.eq(layer), "projection"
            ].to_numpy(float)
        )
        for model, layer in selections
    }

    for row_index, (model, layer) in enumerate(selections):
        for column_index, axis in enumerate(AXIS_ORDER):
            panel = panels[row_index, column_index]
            subset = points[
                points.model.eq(model) & points.layer.eq(layer) & points.axis.eq(axis)
            ]
            if subset.empty:
                raise ValueError(
                    f"No points for model={model!r}, layer={layer}, axis={axis}"
                )
            stats = fit_affine_readout(subset.ground_truth, subset.projection)
            x_limits, x_major_ticks = _nice_symmetric_scale(
                subset.ground_truth.to_numpy(float)
            )
            y_limits, y_major_ticks = y_scale_by_model[(model, layer)]
            point_color = (
                POINT_COLOR if readout == "single_object" else PAIR_POINT_COLOR
            )
            panel.scatter(
                subset.ground_truth,
                subset.projection,
                s=5,
                alpha=0.70,
                color=point_color,
                edgecolors="none",
                rasterized=True,
                zorder=3,
            )
            x_line = np.linspace(
                x_limits[0],
                x_limits[1],
                240,
            )
            panel.plot(
                x_line,
                stats["intercept"] + stats["slope"] * x_line,
                color=FIT_COLOR,
                linewidth=2.3,
                zorder=4,
            )
            panel.set_xlim(*x_limits)
            panel.set_ylim(*y_limits)
            panel.set_box_aspect(1)
            panel.set_facecolor(BACKGROUND_COLOR)
            panel.set_axisbelow(True)
            panel.xaxis.set_major_locator(FixedLocator(x_major_ticks))
            panel.yaxis.set_major_locator(FixedLocator(y_major_ticks))
            panel.xaxis.set_major_formatter(FuncFormatter(_format_tick))
            panel.yaxis.set_major_formatter(FuncFormatter(_format_tick))
            panel.xaxis.set_minor_locator(
                FixedLocator(np.linspace(x_limits[0], x_limits[1], 9)[1::2])
            )
            panel.yaxis.set_minor_locator(
                FixedLocator(np.linspace(y_limits[0], y_limits[1], 9)[1::2])
            )
            panel.grid(
                which="major",
                color=MAJOR_GRID_COLOR,
                linewidth=0.8,
                alpha=0.78,
            )
            panel.grid(
                which="minor",
                color=MINOR_GRID_COLOR,
                linewidth=0.55,
                alpha=0.82,
            )
            arrow_style = {
                "arrowstyle": "-|>",
                "color": AXIS_COLOR,
                "linewidth": 1.9,
                "alpha": 0.94,
                "mutation_scale": 11.5,
                "shrinkA": 0,
                "shrinkB": 0,
            }
            panel.annotate(
                "",
                xy=(x_limits[1] * 0.995, 0.0),
                xytext=(x_limits[0], 0.0),
                arrowprops=arrow_style,
                annotation_clip=False,
                zorder=2,
            )
            panel.annotate(
                "",
                xy=(0.0, y_limits[1] * 0.995),
                xytext=(0.0, y_limits[0]),
                arrowprops=arrow_style,
                annotation_clip=False,
                zorder=2,
            )
            for spine in panel.spines.values():
                spine.set_visible(False)
            axis_name = AXIS_DISPLAY[axis][0]
            title_prefix = f"{model} L{layer}\n" if len(selections) > 1 else ""
            panel.set_title(
                title_prefix
                + rf"{axis_name}   $\mathbf{{\rho}}$={stats['spearman_rho']:.3f}",
                fontsize=16.5,
                fontweight="bold",
                color="#18171c",
                pad=13,
            )
            if readout == "single_object":
                x_label = "Ground-Truth Coordinate"
                y_label = "S-Space Projection"
            else:
                x_label = "Ground-Truth Coordinate Difference"
                y_label = "S-Space Projection Difference"
            panel.set_xlabel(
                x_label,
                fontsize=12.5,
                fontweight="bold",
                color="#29272e",
                labelpad=10,
            )
            panel.set_ylabel(
                y_label,
                fontsize=12.5,
                fontweight="bold",
                color="#29272e",
                labelpad=10,
            )
            panel.tick_params(
                axis="both",
                which="major",
                labelsize=9.5,
                colors="#55515b",
                length=0,
                pad=5,
            )
            panel.tick_params(axis="both", which="minor", length=0)

    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.16,
        top=0.90,
        wspace=0.43,
        hspace=0.58,
    )
    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor=BACKGROUND_COLOR,
        transparent=False,
    )
    if svg_output is not None:
        fig.savefig(
            svg_output,
            dpi=220,
            bbox_inches="tight",
            pad_inches=0.08,
            facecolor=BACKGROUND_COLOR,
            transparent=False,
            metadata={"Date": None},
        )
    plt.close(fig)


def _verify_reference_manifest(
    manifest_path: Path,
    input_path: Path,
    input_format: str,
    selections: tuple[tuple[str, int], ...],
    points: pd.DataFrame,
    pair_points: pd.DataFrame,
    summary: pd.DataFrame,
) -> dict[str, object]:
    """Check frozen assets and recomputed tables before writing any output."""
    reference = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_path, expected_sha256 in reference["files"].items():
        path = (ROOT / relative_path).resolve()
        if not path.is_relative_to(ROOT) or file_sha256(path) != expected_sha256:
            raise ValueError(f"Reference asset checksum mismatch: {relative_path}")
    expected_selections = tuple(
        (item["model"], item["layer"]) for item in reference["selections"]
    )
    if (
        input_path.resolve() != (ROOT / reference["input"]).resolve()
        or input_format != reference["input_format"]
        or selections != expected_selections
        or str(points.dataset.iloc[0]) != reference["dataset"]
    ):
        raise ValueError("Input, format, dataset, or selection differs from reference")
    counts = reference["counts"]
    if (len(points), len(pair_points), len(summary)) != (
        counts["single_object_rows"],
        counts["pair_difference_rows"],
        counts["fit_summary_rows"],
    ):
        raise ValueError("Recomputed row counts differ from reference")
    for name, actual, keys in (
        (
            "pair_difference_points.csv",
            pair_points,
            ["dataset", "model", "pair_id", "layer", "axis", "role"],
        ),
        (
            "fit_summary.csv",
            summary,
            ["dataset", "model", "layer", "axis", "role", "readout"],
        ),
    ):
        expected = pd.read_csv(manifest_path.parent / name)
        pd.testing.assert_frame_equal(
            actual.sort_values(keys).reset_index(drop=True),
            expected.sort_values(keys).reset_index(drop=True),
            check_exact=False,
            rtol=1e-12,
            atol=1e-12,
            obj=f"Recomputed {name}",
        )
    return {
        "reference_manifest_sha256": file_sha256(manifest_path),
        "reference_protocol": reference["protocol"],
        "reference_verified": True,
        "reference_numeric_tolerance": {"rtol": 1e-12, "atol": 1e-12},
        "coordinate_conventions": reference["coordinate_conventions"],
    }


def _write_manifest(
    output_dir: Path,
    input_path: Path,
    input_metadata: dict[str, object],
    selections: tuple[tuple[str, int], ...],
    dataset: str,
    input_rows: int,
    pair_rows: int,
    outputs: list[str],
) -> None:
    manifest = {
        "protocol": "continuous_object_coordinate_audit_v1",
        "input": str(input_path),
        "input_sha256": file_sha256(input_path),
        "tool_sha256": file_sha256(Path(__file__)),
        "scoring_sha256": file_sha256(Path(__file__).with_name("scoring.py")),
        **input_metadata,
        "dataset": dataset,
        "selections": [{"model": model, "layer": layer} for model, layer in selections],
        "input_rows": input_rows,
        "pair_rows": pair_rows,
        "outputs": outputs,
        "output_sha256": {name: file_sha256(output_dir / name) for name in outputs},
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--input-format",
        choices=("standard_v1", "coco_absolute_object_readouts_v1"),
        required=True,
        help="Explicit schema/coordinate adapter; no format is inferred.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="Optional frozen reference: verify checksums and recomputed results.",
    )
    parser.add_argument(
        "--selection",
        action="append",
        required=True,
        help="Repeated explicit MODEL=LAYER display selection.",
    )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)

    selections = parse_selection(args.selection)
    loaded, input_metadata = load_points(args.input, args.input_format)
    points = select_points(loaded, selections)
    summary, pair_points = summarize_continuous_readouts(points)
    if args.reference_manifest is not None:
        input_metadata.update(
            _verify_reference_manifest(
                args.reference_manifest,
                args.input,
                args.input_format,
                selections,
                points,
                pair_points,
                summary,
            )
        )

    font_paths = [
        Path(__file__).parent / "resources" / "fonts" / f"InstrumentSans-{weight}.ttf"
        for weight in ("Regular", "Bold")
    ]
    for path in font_paths:
        font_manager.fontManager.addfont(path)
    font_family = font_manager.FontProperties(fname=font_paths[0]).get_name()
    input_metadata["font_sha256"] = {
        path.name: file_sha256(path) for path in font_paths
    }
    input_metadata["matplotlib_version"] = matplotlib.__version__

    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = [
        "single_object_points.csv",
        "pair_difference_points.csv",
        "fit_summary.csv",
        "single_object_scatters.png",
        "pair_difference_scatters.png",
        "single_object_scatters.svg",
        "pair_difference_scatters.svg",
    ]
    points.to_csv(args.output_dir / outputs[0], index=False)
    pair_points.to_csv(args.output_dir / outputs[1], index=False)
    summary.to_csv(args.output_dir / outputs[2], index=False)
    with matplotlib.rc_context(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family],
            "svg.hashsalt": "sspace-object-coordinates-v1",
        }
    ):
        _plot_grid(
            points,
            selections,
            "single_object",
            args.output_dir / outputs[3],
            args.output_dir / outputs[5],
        )
        _plot_grid(
            pair_points,
            selections,
            "pair_difference",
            args.output_dir / outputs[4],
            args.output_dir / outputs[6],
        )
    _write_manifest(
        args.output_dir,
        args.input,
        input_metadata,
        selections,
        str(points.dataset.iloc[0]),
        len(points),
        len(pair_points),
        outputs,
    )


if __name__ == "__main__":
    main()
