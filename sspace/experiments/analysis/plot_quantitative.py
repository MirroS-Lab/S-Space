"""Plot quantitative tables produced by the released experiment runners."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sspace.run_records import atomic_json, file_sha256
from sspace.experiments.spinbench.evolving_cot import STAGE_SCORE_PROTOCOL


def retention_table(path: Path) -> pd.DataFrame:
    """Convert per-strength prediction changes to answer retention."""
    frame = pd.read_csv(path)
    layer_required = {
        "experiment", "source_layer", "edit_scale", "scope", "n",
        "effect_count", "effect_rate",
    }
    if layer_required.issubset(frame):
        frame = frame[frame.scope.eq("overall")].copy()
        if frame.empty or (frame.n <= 0).any():
            raise ValueError("Retention needs nonempty overall observations")
        if frame.duplicated(["experiment", "source_layer", "edit_scale"]).any():
            raise ValueError("Duplicate intervention strengths")
        if (frame.effect_count < 0).any() or (frame.effect_count > frame.n).any():
            raise ValueError("Effect counts are outside the sample count")
        if not np.allclose(frame.effect_rate, frame.effect_count / frame.n):
            raise ValueError("Effect counts and rates disagree")
        frame["answer_retention"] = 1 - frame.effect_rate
        return frame
    required = {"experiment", "source_layer", "parameter", "value", "scope",
                "n", "changed_count", "changed_rate"}
    if not required.issubset(frame):
        raise ValueError("Use summary_all.csv with per-strength changed counts")
    frame = frame[frame.scope.eq("overall")].copy()
    if frame.empty or (frame.n <= 0).any():
        raise ValueError("Retention needs nonempty overall observations")
    if frame.duplicated(["experiment", "source_layer", "parameter", "value"]).any():
        raise ValueError("Duplicate intervention strengths")
    if (frame.changed_count < 0).any() or (frame.changed_count > frame.n).any():
        raise ValueError("Changed counts are outside the sample count")
    if not np.allclose(frame.changed_rate, frame.changed_count / frame.n):
        raise ValueError("Changed counts and rates disagree")
    frame["answer_retention"] = 1 - frame.changed_count / frame.n
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=("causality", "cot", "context"), required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if len(args.inputs) != (2 if args.protocol == "context" else 1):
        raise ValueError("Context needs WITHOUT then WITH metrics; other protocols need one input")
    if args.protocol == "causality":
        table = retention_table(args.inputs[0])
        if "edit_scale" in table:
            groups = list(table.groupby("source_layer", sort=True))
        else:
            groups = list(table.groupby(["source_layer", "parameter"], sort=True))
        fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
        labels = {
            "absolute_shift": "Absolute position (shift)",
            "relative_common_shift": "Relative position (shift)",
            "nonspatial_shift": "Object color (shift)",
            "relative_swap": "Relative position (swap)",
        }
        for ax, (key, group) in zip(axes[0], groups, strict=True):
            if "edit_scale" in table:
                layer, parameter, x = key, "S-Space edit scale", "edit_scale"
            else:
                layer, parameter, x = key[0], key[1], "value"
            for experiment, rows in group.groupby("experiment", sort=True):
                rows = rows.sort_values(x)
                ax.plot(
                    rows[x], rows.answer_retention, marker="o",
                    label=labels.get(experiment, experiment),
                )
            ax.axhline(.5, color="gray", linestyle="--")
            ax.set(xlabel=parameter, ylabel="Answer retention vs baseline", title=f"L{layer}", ylim=(0, 1.02))
            ax.legend(fontsize=8)
    elif args.protocol == "cot":
        summary = json.loads(args.inputs[0].read_text())
        if summary["protocol"] != STAGE_SCORE_PROTOCOL:
            raise ValueError("Expected the fixed-template three-stage scorer output")
        rows = [{"stage": stage, "readout": readout,
                 **summary["metrics"][stage][readout]}
                for stage in ("early", "middle", "late") for readout in ("direct", "rotated")]
        table = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(6, 4))
        for readout in ("direct", "rotated"):
            part = table[table.readout.eq(readout)]
            ax.plot(part.stage, part.accuracy, marker="o", label=readout)
        ax.set(ylabel="Readout accuracy", ylim=(0, 1.02), title=f"Fixed CoT · L{summary['selected_layer']}")
        ax.legend()
    else:
        frames = []
        for mode, path in zip(("without", "with"), args.inputs, strict=True):
            frame = pd.read_csv(path)
            if not {"layer_id", "overall_accuracy"}.issubset(frame) or frame.empty:
                raise ValueError("Expected layerwise_metrics.csv")
            if frame.layer_id.duplicated().any():
                raise ValueError("Duplicate layer IDs")
            frame["premise_mode"] = mode
            frames.append(frame.sort_values("layer_id"))
        if frames[0].layer_id.tolist() != frames[1].layer_id.tolist():
            raise ValueError("Premise conditions must cover identical layer IDs")
        table = pd.concat(frames, ignore_index=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for mode, frame in zip(("without", "with"), frames, strict=True):
            ax.plot(frame.layer_id, frame.overall_accuracy, label=f"{mode} premise")
        ax.set(xlabel="Zero-based post-block layer", ylabel="Rotation readout accuracy", ylim=(0, 1.02))
        ax.legend()
    metric = "answer_retention" if args.protocol == "causality" else (
        "accuracy" if args.protocol == "cot" else "overall_accuracy")
    if not np.isfinite(table[metric]).all() or not table[metric].between(0, 1).all():
        plt.close(fig)
        raise ValueError("Accuracy/retention must be finite and between zero and one")
    args.output_dir.mkdir(parents=True)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(args.output_dir / f"{args.protocol}.{suffix}", dpi=200)
    plt.close(fig)
    table.to_csv(args.output_dir / "plot_source.csv", index=False)
    atomic_json(args.output_dir / "provenance.json", {
        "protocol": args.protocol,
        "inputs": {str(p.resolve()): file_sha256(p) for p in args.inputs},
        "outputs": {p.name: file_sha256(p) for p in args.output_dir.iterdir()},
    })


if __name__ == "__main__":
    main()
