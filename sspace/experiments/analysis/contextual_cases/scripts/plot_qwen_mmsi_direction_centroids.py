#!/usr/bin/env python3
"""Plot occurrence- and question-balanced means for MMSI direction tokens."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = (
    "north",
    "northeast",
    "east",
    "southeast",
    "south",
    "southwest",
    "west",
    "northwest",
)
COLORS = {
    "north": "#2878b5",
    "northeast": "#5b8fd1",
    "east": "#e39b27",
    "southeast": "#e58a57",
    "south": "#d94f45",
    "southwest": "#b45f75",
    "west": "#519e5b",
    "northwest": "#6f6ab1",
}


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"mmsi_id", "word", "horizontal", "vertical"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Direction-token table is missing columns: {missing}")
    cardinal = frame[frame["word"].isin(ORDER)].copy()
    missing_words = sorted(set(ORDER) - set(cardinal["word"]))
    if missing_words:
        raise ValueError(f"Direction-token table is missing words: {missing_words}")
    per_question = cardinal.groupby(["mmsi_id", "word"], as_index=False)[
        ["horizontal", "vertical"]
    ].mean()
    rows = []
    for word in ORDER:
        occurrences = cardinal[cardinal["word"].eq(word)]
        questions = per_question[per_question["word"].eq(word)]
        rows.append(
            {
                "word": word,
                "occurrence_count": len(occurrences),
                "question_count": len(questions),
                "occurrence_mean_H": occurrences["horizontal"].mean(),
                "occurrence_mean_V": occurrences["vertical"].mean(),
                "question_balanced_mean_H": questions["horizontal"].mean(),
                "question_balanced_mean_V": questions["vertical"].mean(),
                "question_sem_H": questions["horizontal"].sem(),
                "question_sem_V": questions["vertical"].sem(),
            }
        )
    return pd.DataFrame(rows)


def plot_occurrence_scatter(frame: pd.DataFrame, output: Path) -> None:
    """Render every geographic direction-token occurrence with up-positive V."""

    cardinal = frame[frame["word"].isin(ORDER)].copy()
    cardinal["vertical_up"] = -cardinal["vertical"]
    fig, ax = plt.subplots(figsize=(12, 9))
    for word in ORDER:
        group = cardinal[cardinal["word"].eq(word)]
        ax.scatter(
            group["horizontal"],
            group["vertical_up"],
            s=42,
            alpha=0.7,
            label=f"{word} (n={len(group)})",
            color=COLORS[word],
            edgecolors="none",
            rasterized=True,
        )
    ax.axhline(0, color="#888888", linewidth=1)
    ax.axvline(0, color="#888888", linewidth=1)
    ax.grid(alpha=0.12)
    ax.set_xlabel("L43 horizontal projection (right +)")
    ax.set_ylabel("L43 vertical projection (up +; − original V)")
    ax.set_title(
        "Qwen3.6 visible-CoT eight-direction tokens on released COCO axes\n"
        "display: right +, up +"
    )
    ax.legend(ncol=4, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def draw_panel(
    ax: plt.Axes,
    summary: pd.DataFrame,
    *,
    balanced: bool,
    up_positive: bool = False,
) -> None:
    prefix = "question_balanced" if balanced else "occurrence"
    for row in summary.itertuples(index=False):
        x = getattr(row, f"{prefix}_mean_H")
        raw_y = getattr(row, f"{prefix}_mean_V")
        y = -raw_y if up_positive else raw_y
        color = COLORS[row.word]
        ax.plot([0, x], [0, y], color=color, alpha=0.35, linewidth=2)
        if balanced:
            ax.errorbar(
                x,
                y,
                xerr=1.96 * row.question_sem_H,
                yerr=1.96 * row.question_sem_V,
                color=color,
                capsize=3,
                linewidth=1.5,
                fmt="none",
                alpha=0.8,
            )
        ax.scatter(x, y, s=115, color=color, edgecolor="white", linewidth=1.2, zorder=3)
        count = row.question_count if balanced else row.occurrence_count
        ax.annotate(
            f"{row.word}\n(n={count})",
            (x, y),
            xytext=(6, 5),
            textcoords="offset points",
            fontsize=9,
            color=color,
            weight="bold",
        )
    ax.axhline(0, color="#888", linewidth=1)
    ax.axvline(0, color="#888", linewidth=1)
    ax.grid(alpha=0.18)
    ax.set_xlabel("L43 horizontal projection (right +)")
    ax.set_ylabel(
        "L43 vertical projection (up +; − original V)"
        if up_positive
        else "L43 vertical projection (below +)"
    )
    ax.set_title(
        "Question-balanced means (95% CI)" if balanced else "Occurrence-weighted means"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-token-questions",
        type=int,
        help="Fail unless this many unique MMSI IDs contain a supported token.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.tokens)
    token_questions = int(frame["mmsi_id"].nunique())
    if (
        args.expected_token_questions is not None
        and token_questions != args.expected_token_questions
    ):
        raise ValueError(
            f"Found {token_questions} token-bearing questions, expected "
            f"{args.expected_token_questions}"
        )
    summary = summarize(frame)
    plot_occurrence_scatter(
        frame, args.output_dir / "eight_direction_token_hv_scatter_up_positive.png"
    )
    summary.to_csv(args.output_dir / "eight_direction_mean_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    draw_panel(axes[0], summary, balanced=False)
    draw_panel(axes[1], summary, balanced=True)
    axes[0].set_xlim(-2.8, 2.8)
    axes[0].set_ylim(-4.7, 3.0)
    fig.suptitle(
        "Qwen3.6 MMSI visible-CoT direction-token centroids at L43", fontsize=17
    )
    fig.tight_layout()
    fig.savefig(args.output_dir / "eight_direction_token_means.png", dpi=200)
    plt.close(fig)

    up_summary = summary.copy()
    up_summary["occurrence_mean_Y_up"] = -up_summary["occurrence_mean_V"]
    up_summary["question_balanced_mean_Y_up"] = -up_summary["question_balanced_mean_V"]
    up_summary.to_csv(
        args.output_dir / "eight_direction_mean_summary_up_positive.csv",
        index=False,
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    draw_panel(axes[0], summary, balanced=False, up_positive=True)
    draw_panel(axes[1], summary, balanced=True, up_positive=True)
    axes[0].set_xlim(-2.8, 2.8)
    axes[0].set_ylim(-3.0, 4.7)
    fig.suptitle(
        "Qwen3.6 MMSI visible-CoT direction-token centroids at L43 "
        "(display: right +, up +)",
        fontsize=17,
    )
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "eight_direction_token_means_up_positive.png", dpi=200
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
