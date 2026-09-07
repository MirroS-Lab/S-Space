"""List or launch individual experiment stages."""

from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINTS = {
    "quantitative-plot": (
        "analysis",
        "sspace.experiments.analysis.plot_quantitative",
        "Plot causal retention, fixed-CoT stages, or premise layer sweeps.",
    ),
    "mmsi-prepare": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.scripts.prepare_mmsi_eightway",
        "Prepare the exhaustive 191-question geographic-answer MMSI subset.",
    ),
    "mmsi-extract": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.scripts.extract_qwen_mmsi_direction_tokens",
        "Generate Qwen3.6 reasoning and replay exact direction token IDs at L43.",
    ),
    "mmsi-plot": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.scripts.plot_qwen_mmsi_direction_centroids",
        "Plot per-token and question-balanced MMSI direction readouts.",
    ),
    "spinbench-layerwise": (
        "analysis",
        "sspace.experiments.spinbench.perspective_taking.tools.run_spinbench_layerwise_sweep",
        "Sweep matching axes at every layer with and without the premise.",
    ),
    "spatial-causality": (
        "analysis",
        "sspace.experiments.intervention.spatial_causality.cli",
        "Layer intervention and answer-change analysis; with baseline-relative answer retention.",
    ),
    "object-lens": (
        "analysis",
        "sspace.experiments.analysis.object_coordinate_lens.cli",
        "Single-image coordinate extraction CLI; Web UI and HTTP service removed.",
    ),
    "continuous-coordinates": (
        "analysis",
        "sspace.experiments.analysis.object_coordinates.plot",
        "Continuous-coordinate affine fits and plots.",
    ),
    "contextual-run": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.cli",
        "Model projection for one checksum-bound contextual case manifest.",
    ),
    "contextual-build": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.scripts.build_manifest",
        "Build a contextual case manifest from a compact specification.",
    ),
    "contextual-analyze": (
        "analysis",
        "sspace.experiments.analysis.contextual_cases.scripts.analyze",
        "Summarize and plot contextual case records.",
    ),
    "evolving-cot-extract": (
        "analysis",
        "sspace.experiments.spinbench.evolving_cot.scripts.run_qwen36_prefilled_cot_activations",
        "Gold-conditioned prefilled-CoT activation extraction; for staged readout analysis.",
    ),
    "evolving-cot-stages": (
        "analysis",
        "sspace.experiments.spinbench.evolving_cot.scripts.score_prefilled_spinbench_cot_stages",
        "Summarize frozen prefilled-CoT stage activations.",
    ),
}


def _list() -> None:
    for name, (classification, module, description) in ENTRYPOINTS.items():
        print(f"{name}\t{classification}\t{module}\t{description}")


def main() -> None:
    arguments = sys.argv[1:]
    if not arguments or arguments == ["list"]:
        _list()
        return
    dry_run = False
    if arguments[0] == "--dry-run":
        dry_run = True
        arguments = arguments[1:]
    if not arguments or arguments[0] not in ENTRYPOINTS:
        allowed = ", ".join(ENTRYPOINTS)
        raise SystemExit(
            f"usage: python -m sspace.experiment_steps [--dry-run] NAME [ARGS]; NAME: {allowed}"
        )
    name, *extra_arguments = arguments
    if extra_arguments[:1] == ["--"]:
        extra_arguments = extra_arguments[1:]
    module = ENTRYPOINTS[name][1]
    command = (sys.executable, "-m", module, *extra_arguments)
    if dry_run:
        print(shlex.join(command))
        return
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
