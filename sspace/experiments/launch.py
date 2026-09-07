"""Launch one experiment through the evaluator declared by its config."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sspace.determinism import deterministic_cuda_environment
from sspace.experiments.config import validate_launch_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PYTHONS = {
    "molmo2_er": Path(".envs/spatialmqa/bin/python"),
    "molmoact2": Path(".envs/molmoact/bin/python"),
    "molmoact2_pretrain": Path(".envs/molmoact/bin/python"),
    "qwen35_4b": Path(".envs/qwen36/bin/python"),
    "qwen36_27b": Path(".envs/qwen36/bin/python"),
}
RUNNER_MODULES = {
    "pairwise_projection": "sspace.experiments.common.pairwise.projection_cli",
    "spinbench_direct": (
        "sspace.experiments.spinbench.perspective_taking.direct_cli"
    ),
    "spinbench_projection": (
        "sspace.experiments.spinbench.perspective_taking.projection_cli"
    ),
    "hstar_multiview": "sspace.experiments.multi_view.hstar.hstar_cli",
    "prompt_where": "sspace.experiments.prompt.where.run_sweep",
    "prompt_color": "sspace.experiments.prompt.color.run_sweep",
    "contextual_case": "sspace.experiments.analysis.contextual_cases.cli",
}


@dataclass(frozen=True)
class ExperimentDescriptor:
    """Hold only the fields needed before the evaluator validates the config."""

    path: Path
    runner: str
    suites: tuple[str, ...]
    model_adapter: str
    output_dir: Path
    shard_count: int


def load_experiment_descriptor(
    path: Path, project_root: Path = PROJECT_ROOT
) -> ExperimentDescriptor:
    """Load explicit routing, runtime, suite, output, and sharding metadata.

    The selected evaluator remains responsible for validating every
    experiment-specific field before it loads a model or writes output.
    """
    resolved_path = path if path.is_absolute() else project_root / path
    values: dict[str, Any] = json.loads(resolved_path.read_text(encoding="utf-8"))
    runner = values.get("runner")
    if not isinstance(runner, str):
        raise ValueError("Experiment config requires a string runner")
    if runner not in {*RUNNER_MODULES, "direct_generation"}:
        raise ValueError(f"Unknown experiment runner {runner!r}")
    suites = validate_launch_metadata(values, runner)
    model_adapter = values.get("model_adapter")
    if model_adapter not in DEFAULT_MODEL_PYTHONS:
        raise ValueError(f"Unknown experiment model_adapter {model_adapter!r}")
    output_value = values.get("output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("Experiment config requires a non-empty output_dir")
    output_dir = Path(output_value)
    output_dir = output_dir if output_dir.is_absolute() else project_root / output_dir
    shard_count = values.get("shard_count", 1)
    if isinstance(shard_count, bool) or not isinstance(shard_count, int):
        raise ValueError("shard_count must be an integer")
    if runner != "direct_generation" and shard_count != 1:
        raise ValueError("Only direct_generation accepts shard_count")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    return ExperimentDescriptor(
        path=resolved_path.resolve(),
        runner=runner,
        suites=suites,
        model_adapter=model_adapter,
        output_dir=output_dir.resolve(),
        shard_count=shard_count,
    )


def experiment_commands(
    descriptor: ExperimentDescriptor,
    python: Path,
    limit: int | None = None,
) -> tuple[tuple[str, ...], ...]:
    """Build the exact subprocess sequence for one declared evaluator route."""
    base = (str(python), "-m")
    config = str(descriptor.path)
    if descriptor.runner == "direct_generation":
        worker = "sspace.experiments.common.direct.cli"
        merge = "sspace.experiments.common.direct.merge_cli"
        suffix = () if limit is None else ("--limit", str(limit))
        return (
            (*base, worker, "--config", config, *suffix),
            (*base, merge, "--config", config, *suffix),
        )
    module = RUNNER_MODULES[descriptor.runner]
    command = (*base, module, "--config", config)
    if limit is not None:
        command = (*command, "--limit", str(limit))
    return (command,)


def resolve_python(
    descriptor: ExperimentDescriptor,
    override: Path | None,
    *,
    require_exists: bool,
) -> Path:
    """Resolve the configured model's project environment or one override."""
    python = override or PROJECT_ROOT / DEFAULT_MODEL_PYTHONS[descriptor.model_adapter]
    python = python if python.is_absolute() else PROJECT_ROOT / python
    if require_exists and not python.is_file():
        raise FileNotFoundError(python)
    return python


def run_commands(commands: Sequence[Sequence[str]], dry_run: bool) -> None:
    """Print every command and execute sequentially unless this is a dry run."""
    environment = deterministic_cuda_environment()
    for index, command in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    """Parse one config, select its declared evaluator, and launch it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    descriptor = load_experiment_descriptor(args.config)
    python = resolve_python(
        descriptor,
        args.python,
        require_exists=not args.dry_run,
    )
    run_commands(experiment_commands(descriptor, python, args.limit), args.dry_run)


if __name__ == "__main__":
    main()
