"""Launch config-declared experiment suites with explicit model runtimes."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from sspace.config_paths import as_project_relative
from sspace.experiments.launch import (
    DEFAULT_MODEL_PYTHONS,
    PROJECT_ROOT,
    experiment_commands,
    load_experiment_descriptor,
    run_commands,
)
from sspace.experiments.identity import experiment_launch_fingerprint
from sspace.run_records import file_sha256
from sspace.core.artifacts import SSpaceArtifact


CONFIG_ROOT = PROJECT_ROOT / "configs/experiments"


@dataclass(frozen=True)
class _EvaluationJob:
    """Bind one checked-in config to its declared model and suites."""

    model_adapter: str
    config_path: Path
    suites: tuple[str, ...]


def _registered_jobs() -> tuple[_EvaluationJob, ...]:
    """Read every checked-in runnable config in deterministic path order."""
    jobs = []
    for path in sorted(CONFIG_ROOT.rglob("*.json")):
        if path.name.endswith(".example.json"):
            continue
        values = json.loads(path.read_text(encoding="utf-8"))
        if "runner" not in values:
            continue
        descriptor = load_experiment_descriptor(path)
        jobs.append(
            _EvaluationJob(
                descriptor.model_adapter,
                descriptor.path,
                descriptor.suites,
            )
        )
    return tuple(jobs)


ALL_JOBS = _registered_jobs()
SUITES = {
    suite: tuple(job for job in ALL_JOBS if suite in job.suites)
    for suite in sorted({suite for job in ALL_JOBS for suite in job.suites})
}


def _completed_and_valid(output_dir: Path, launch_fingerprint: str) -> bool:
    """Return true only for a sealed output matching the current launch inputs."""
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    checksums_path = output_dir / "checksums.json"
    if not checksums_path.is_file():
        raise FileNotFoundError(checksums_path)
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = output_dir / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed experiment checksum changed: {path}")
    resolved_path = output_dir / "resolved_config.json"
    if not resolved_path.is_file():
        return False
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return resolved.get("launch_fingerprint") == launch_fingerprint


def _parse_python_overrides(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated ``MODEL=PYTHON`` overrides without guessing aliases."""
    overrides: dict[str, Path] = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if not separator or model not in DEFAULT_MODEL_PYTHONS or not raw_path:
            raise ValueError(f"Invalid --python override {value!r}")
        if model in overrides:
            raise ValueError(f"Duplicate --python override for {model}")
        path = Path(raw_path)
        overrides[model] = path if path.is_absolute() else PROJECT_ROOT / path
    return overrides


def _selected_jobs(
    suite_name: str | None, models: set[str]
) -> tuple[_EvaluationJob, ...]:
    """Select one declared suite, or every config once, then filter models."""
    unknown_models = models - set(DEFAULT_MODEL_PYTHONS)
    if unknown_models:
        raise ValueError(f"Unknown model filters: {sorted(unknown_models)}")
    jobs = ALL_JOBS if suite_name is None else SUITES[suite_name]
    selected = tuple(job for job in jobs if not models or job.model_adapter in models)
    if not selected:
        raise ValueError("Model filter removed every job from the suite")
    return selected


def _registered_config_job(config_path: Path) -> _EvaluationJob:
    """Resolve one CLI path and reject configs outside the formal registry."""
    registry = json.loads(
        (PROJECT_ROOT / "configs/registry.json").read_text(encoding="utf-8")
    )
    formal_entries = {
        (PROJECT_ROOT / entry["path"]).resolve(): entry
        for entry in registry.get("formal_configs", [])
    }
    candidates = (
        (config_path if config_path.is_absolute() else PROJECT_ROOT / config_path),
        (config_path if config_path.is_absolute() else CONFIG_ROOT / config_path),
    )
    registered = {job.config_path.resolve(): job for job in ALL_JOBS}
    for candidate in candidates:
        resolved = candidate.resolve()
        entry = formal_entries.get(resolved)
        if entry is None:
            continue
        if file_sha256(resolved) != entry.get("sha256"):
            raise ValueError(f"Registered Config digest changed: {resolved}")
        job = registered.get(resolved)
        if job is not None:
            return job
    raise ValueError(f"Config is not registered for formal execution: {config_path}")


def _with_output_root(
    jobs: Sequence[_EvaluationJob],
    output_root: Path | None,
    artifact_overrides: dict[str, Path] | None = None,
) -> tuple[_EvaluationJob, ...]:
    """Materialize configs with isolated outputs and optional trained artifacts."""
    overrides = artifact_overrides or {}
    resolved_root = None
    if output_root is not None:
        resolved_root = (
            output_root if output_root.is_absolute() else PROJECT_ROOT / output_root
        ).resolve()
    cache_key = json.dumps(
        {
            "output_root": None if resolved_root is None else str(resolved_root),
            "artifacts": {key: str(value) for key, value in sorted(overrides.items())},
        },
        sort_keys=True,
    )
    key = hashlib.sha256(cache_key.encode()).hexdigest()[:12]
    config_root = PROJECT_ROOT / ".tmp/experiment-configs" / key

    def apply_artifact(values: dict, model: str) -> None:
        if model not in overrides or "artifact_dir" not in values:
            return
        artifact_dir = overrides[model].resolve()
        values["artifact_dir"] = as_project_relative(
            artifact_dir, PROJECT_ROOT, "artifact_dir"
        )
        if not (artifact_dir / "manifest.json").is_file():
            return
        selected = SSpaceArtifact.load(artifact_dir).manifest.model.selected_layer
        if "layer_id" in values and values["layer_id"] is not None:
            values["layer_id"] = selected
        if "selected_layer" in values:
            values["selected_layer"] = selected

    def rewrite_source(source: Path, model: str) -> Path:
        values = json.loads(source.read_text(encoding="utf-8"))
        apply_artifact(values, model)
        relative = source.relative_to(CONFIG_ROOT)
        destination = config_root / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return destination.resolve()

    rewritten = []
    for job in jobs:
        relative = job.config_path.relative_to(CONFIG_ROOT).with_suffix("")
        destination = config_root / relative.with_suffix(".json")
        destination.parent.mkdir(parents=True, exist_ok=True)
        values = json.loads(job.config_path.read_text(encoding="utf-8"))
        if resolved_root is not None:
            values["output_dir"] = as_project_relative(
                resolved_root / relative, PROJECT_ROOT, "output_dir"
            )
        if job.model_adapter in overrides:
            apply_artifact(values, job.model_adapter)
            if "source_configs" in values:
                values["source_configs"] = [
                    as_project_relative(
                        rewrite_source(
                            (PROJECT_ROOT / source).resolve(), job.model_adapter
                        ),
                        PROJECT_ROOT,
                        "source_configs",
                    )
                    for source in values["source_configs"]
                ]
        destination.write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        rewritten.append(
            _EvaluationJob(job.model_adapter, destination.resolve(), job.suites)
        )
    return tuple(rewritten)


def _parse_artifact_overrides(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeated MODEL=ARTIFACT_DIR overrides."""
    overrides: dict[str, Path] = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if not separator or model not in DEFAULT_MODEL_PYTHONS or not raw_path:
            raise ValueError(f"Invalid --artifact-dir override {value!r}")
        if model in overrides:
            raise ValueError(f"Duplicate artifact override for {model}")
        path = Path(raw_path)
        overrides[model] = path if path.is_absolute() else PROJECT_ROOT / path
    return overrides


def _job_commands(
    jobs: Sequence[_EvaluationJob],
    python_overrides: dict[str, Path],
    *,
    require_python: bool,
    limit: int | None = None,
    skip_completed: bool = True,
) -> tuple[tuple[str, ...], ...]:
    """Build commands and batch compatible direct configs per loaded model."""
    commands: list[tuple[str, ...]] = []
    direct_groups: dict[tuple[object, ...], list[tuple[_EvaluationJob, Path]]] = {}
    for job in jobs:
        descriptor = load_experiment_descriptor(job.config_path)
        if skip_completed:
            launch_fingerprint = experiment_launch_fingerprint(
                job.config_path, PROJECT_ROOT, limit
            )
            completion_dir = descriptor.output_dir
            if descriptor.runner == "direct_generation":
                completion_dir = descriptor.output_dir / "merged"
            if _completed_and_valid(completion_dir, launch_fingerprint):
                print(f"skip completed experiment: {descriptor.output_dir}")
                continue
        python = python_overrides.get(job.model_adapter)
        if python is None:
            python = PROJECT_ROOT / DEFAULT_MODEL_PYTHONS[job.model_adapter]
        if require_python and not python.is_file():
            raise FileNotFoundError(python)
        if descriptor.runner != "direct_generation":
            commands.extend(experiment_commands(descriptor, python, limit))
            continue
        values = json.loads(job.config_path.read_text(encoding="utf-8"))
        key = (
            job.model_adapter,
            str(python),
            values["model_path"],
            values["device"],
            values["shard_count"],
            values["max_new_tokens"],
            values["seed"],
        )
        direct_groups.setdefault(key, []).append((job, python))
    worker = "sspace.experiments.common.direct.cli"
    merge = "sspace.experiments.common.direct.merge_cli"
    limit_args = () if limit is None else ("--limit", str(limit))
    for group in direct_groups.values():
        python = group[0][1]
        config_args = tuple(
            value for job, _ in group for value in ("--config", str(job.config_path))
        )
        commands.append((str(python), "-m", worker, *config_args, *limit_args))
        commands.extend(
            (
                str(python),
                "-m",
                merge,
                "--config",
                str(job.config_path),
                *limit_args,
            )
            for job, _ in group
        )
    return tuple(commands)


def main() -> None:
    """List suites or launch one filtered suite or every config exactly once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", nargs="?", choices=tuple(SUITES))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--config",
        type=Path,
        help="Run one checked-in Config from the formal registry",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--artifact-dir", action="append", default=[], metavar="MODEL=DIR"
    )
    parser.add_argument(
        "--python",
        action="append",
        default=[],
        metavar="MODEL=PATH",
        help="Override one registered project's Python interpreter",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.list:
        for name, jobs in SUITES.items():
            print(f"{name}: {len(jobs)} configs")
        return
    selectors = (args.suite is not None, args.all, args.config is not None)
    if sum(selectors) != 1:
        parser.error("provide exactly one suite, --all, or --config")
    overrides = _parse_python_overrides(args.python)
    artifact_overrides = _parse_artifact_overrides(args.artifact_dir)
    models = set(args.model)
    if args.config is not None:
        job = _registered_config_job(args.config)
        unknown_models = models - set(DEFAULT_MODEL_PYTHONS)
        if unknown_models:
            raise ValueError(f"Unknown model filters: {sorted(unknown_models)}")
        if models and job.model_adapter not in models:
            raise ValueError("Model filter removed the registered Config")
        jobs = (job,)
    else:
        jobs = _selected_jobs(None if args.all else args.suite, models)
    if args.output_root is not None or artifact_overrides:
        jobs = _with_output_root(jobs, args.output_root, artifact_overrides)
    commands = _job_commands(
        jobs,
        overrides,
        require_python=not args.dry_run,
        limit=args.limit,
    )
    run_commands(commands, args.dry_run)


if __name__ == "__main__":
    main()
