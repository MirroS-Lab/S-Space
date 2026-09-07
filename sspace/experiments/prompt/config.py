"""Define the shared strict configuration for prompt-control sweeps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sspace.config_paths import resolve_project_path
from sspace.experiments.config import LAUNCH_FIELDS, validate_launch_metadata
from sspace.run_records import file_sha256
from sspace.core.models.runtime import RUNTIME_SPECS


@dataclass(frozen=True)
class PromptSweepConfig:
    """Declare one model, three source benchmarks, and one output root."""

    runner: str
    suites: tuple[str, ...]
    evaluation_id: str
    model_adapter: str
    source_configs: tuple[Path, ...]
    output_dir: Path


def completed_prompt_output(directory: Path, launch_fingerprint: str) -> bool:
    """Verify one immutable output matches the current launch inputs."""
    manifest_path = directory / "run_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        return False
    checksums_path = directory / "checksums.json"
    if not checksums_path.is_file():
        raise FileNotFoundError(checksums_path)
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = directory / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Completed prompt output checksum changed: {path}")
    resolved_path = directory / "resolved_config.json"
    if not resolved_path.is_file():
        return False
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    return resolved.get("launch_fingerprint") == launch_fingerprint


def load_prompt_sweep_config(
    path: Path, project_root: Path, expected_runner: str
) -> PromptSweepConfig:
    """Load one exact prompt-control config and resolve project-relative paths.

    Raises:
        ValueError: Fields, runner, model, or source count are invalid.
        FileNotFoundError: A declared source configuration is absent.
    """
    values = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "evaluation_id",
        "model_adapter",
        "source_configs",
        "output_dir",
    } | LAUNCH_FIELDS
    if set(values) != expected:
        raise ValueError(f"Prompt config fields {set(values)} != {expected}")
    suites = validate_launch_metadata(values, expected_runner)
    if not str(values["evaluation_id"]).strip():
        raise ValueError("Prompt evaluation_id must be non-empty")
    model_adapter = str(values["model_adapter"])
    if model_adapter not in RUNTIME_SPECS:
        raise ValueError(f"Unknown prompt model_adapter {model_adapter!r}")
    sources = values["source_configs"]
    if not isinstance(sources, list) or len(sources) != 3:
        raise ValueError("Prompt sweep requires exactly three source_configs")

    source_configs = tuple(
        resolve_project_path(value, project_root, "source_configs") for value in sources
    )
    for source in source_configs:
        if not source.is_file():
            raise FileNotFoundError(source)
    return PromptSweepConfig(
        runner=expected_runner,
        suites=suites,
        evaluation_id=str(values["evaluation_id"]),
        model_adapter=model_adapter,
        source_configs=source_configs,
        output_dir=resolve_project_path(
            values["output_dir"], project_root, "output_dir"
        ),
    )
