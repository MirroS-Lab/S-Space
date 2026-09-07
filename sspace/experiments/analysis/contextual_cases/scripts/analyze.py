"""Write tables and plots for the two contextual S-Space case-study themes."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sspace.experiments.analysis.contextual_cases.scoring import (
    bootstrap_mean_ci,
    flatten_case_records,
    load_case_records,
    summarize_beyond_visible,
    summarize_beyond_visible_conditions,
    summarize_political,
    summarize_web_political,
)
from sspace.run_records import atomic_json, file_sha256


def write_analysis_checksums(output: Path) -> None:
    """Seal every generated plot and plot-source table."""
    checksums = {
        path.name: file_sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "analysis_checksums.json"
    }
    if not checksums:
        raise ValueError("Contextual analysis produced no files")
    atomic_json(output / "analysis_checksums.json", checksums)


def plot_beyond(points, output: Path, seed: int, draws: int) -> None:
    families = tuple(points.family.unique())
    fig, charts = plt.subplots(
        1, len(families), figsize=(6.2 * len(families), 4.6), squeeze=False
    )
    rng = np.random.default_rng(seed)
    for chart, family in zip(charts[0], families, strict=True):
        subset = points[points.family.eq(family)]
        labels = tuple(subset.readout.unique())
        for index, label in enumerate(labels):
            values = subset[subset.readout.eq(label)].effect.to_numpy(float)
            mean, low, high = bootstrap_mean_ci(values, seed, draws)
            chart.scatter(
                np.full(len(values), index) + rng.normal(0, 0.035, len(values)),
                values,
                alpha=0.35,
                s=24,
            )
            chart.errorbar(
                index, mean, [[mean - low], [high - mean]], fmt="o", capsize=6
            )
        chart.axhline(0.0, color="#444444", linewidth=1)
        chart.set_xticks(range(len(labels)), labels)
        chart.set_title(family.replace("_", " "))
        axis = str(subset.effect_axis.iloc[0])
        chart.set_ylabel(f"Signed condition effect ({axis})")
        chart.grid(axis="y", alpha=0.2)
    fig.suptitle("Beyond the Visible: prompt-matched condition effects")
    fig.tight_layout()
    fig.savefig(output / "beyond_visible_tbars.png", dpi=200)
    plt.close(fig)


def plot_political(single, matched, output: Path) -> None:
    fig, charts = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True)
    for condition, marker in (("blank", "o"), ("parliament", "s")):
        part = single[single.condition.eq(condition)].set_index("word")
        if part.empty:
            continue
        words = [
            word
            for word in ("socialist", "conservative", "journalist")
            if word in part.index
        ]
        x = np.arange(len(words)) + (-0.1 if condition == "blank" else 0.1)
        charts[0].errorbar(
            x,
            part.loc[words, "mean"],
            yerr=part.loc[words, "ci95_half_width"],
            fmt=marker,
            capsize=4,
            label=condition,
        )
    charts[0].axhline(0.0, color="#444444", linewidth=1)
    charts[0].set_xticks(range(3), ("socialist", "conservative", "journalist"))
    charts[0].set_ylabel("horizontal token projection (+ = viewer-right)")
    charts[0].legend(frameon=False)
    view = matched[matched.readout.eq("single_token")].reset_index(drop=True)
    charts[1].errorbar(
        np.arange(len(view)),
        view["mean"],
        yerr=view["ci95_half_width"],
        fmt="o",
        capsize=4,
    )
    charts[1].axhline(0.0, color="#444444", linewidth=1)
    charts[1].set_xticks(
        np.arange(len(view)),
        [
            f"{row.condition}\n{row.contrast.replace('socialist_minus_', 'vs ')}"
            for row in view.itertuples()
        ],
        rotation=20,
    )
    charts[1].set_ylabel("matched socialist-minus-control effect")
    fig.suptitle("Language supervision: political left on the physical H axis")
    fig.savefig(output / "political_semantics.png", dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        choices=("beyond_visible", "political", "political_web", "web_showcase"),
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-draws", type=int, default=20_000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = flatten_case_records(load_case_records(args.records))
    if args.protocol == "political_web":
        summary = summarize_web_political(frame)
        summary.to_csv(args.output_dir / "political_web_readouts.csv", index=False)
        fig, chart = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        chart.scatter(summary.within_model_zscore, np.arange(len(summary)))
        chart.set_yticks(np.arange(len(summary)), summary.meta_target_word)
        chart.set_xlabel("Within-model horizontal z-score (left → right)")
        chart.axvline(0, color="gray", linewidth=0.7)
        fig.savefig(args.output_dir / "political_web.png", dpi=200)
        plt.close(fig)
    elif args.protocol == "web_showcase":
        if not frame.condition.eq("web_showcase").all():
            raise ValueError("web_showcase requires the representative web case manifest")
        frame.to_csv(args.output_dir / "web_showcase_readouts.csv", index=False)
    elif args.protocol == "beyond_visible":
        summary, points = summarize_beyond_visible(
            frame, args.seed, args.bootstrap_draws
        )
        conditions = summarize_beyond_visible_conditions(frame)
        summary.to_csv(args.output_dir / "beyond_visible_summary.csv", index=False)
        points.to_csv(
            args.output_dir / "beyond_visible_prompt_effects.csv", index=False
        )
        conditions.to_csv(
            args.output_dir / "beyond_visible_condition_summary.csv", index=False
        )
        plot_beyond(points, args.output_dir, args.seed, args.bootstrap_draws)
    elif args.protocol == "political":
        single, matched = summarize_political(frame)
        single.to_csv(
            args.output_dir / "political_single_token_summary.csv", index=False
        )
        matched.to_csv(args.output_dir / "political_matched_effects.csv", index=False)
        plot_political(single, matched, args.output_dir)
    write_analysis_checksums(args.output_dir)


if __name__ == "__main__":
    main()
