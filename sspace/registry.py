"""Validate the explicit formal experiment registry and command graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from .experiments.launch import DEFAULT_MODEL_PYTHONS, load_experiment_descriptor
from .experiments.scripts.run_suite import ALL_JOBS, _job_commands
from .experiment_steps import ENTRYPOINTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = PROJECT_ROOT / "configs/registry.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "1.1.0":
        raise ValueError("Unsupported experiment registry schema")
    entries = value.get("formal_configs")
    if not isinstance(entries, list) or not entries or len(entries) != value.get("expected_config_count"):
        raise ValueError("Registry Config count differs from declared coverage")
    paths = [entry.get("path") for entry in entries]
    if len(set(paths)) != len(paths):
        raise ValueError("Registry Config paths must be unique")
    discovered = {
        job.config_path.relative_to(PROJECT_ROOT).as_posix() for job in ALL_JOBS
    }
    if set(paths) != discovered:
        raise ValueError("Registry paths differ from the runnable Config set")
    for entry in entries:
        config_path = PROJECT_ROOT / entry["path"]
        if file_sha256(config_path) != entry.get("sha256"):
            raise ValueError(f"Config digest changed: {entry['path']}")
        descriptor = load_experiment_descriptor(config_path)
        expected = {
            "runner": descriptor.runner,
            "model_adapter": descriptor.model_adapter,
            "suites": list(descriptor.suites),
            "output_dir": json.loads(config_path.read_text(encoding="utf-8"))[
                "output_dir"
            ],
        }
        if any(entry.get(key) != item for key, item in expected.items()):
            raise ValueError(f"Registry metadata changed: {entry['path']}")
    overrides = {model: Path(sys.executable) for model in DEFAULT_MODEL_PYTHONS}
    commands = _job_commands(
        ALL_JOBS,
        overrides,
        require_python=True,
        skip_completed=False,
    )
    if len(commands) != value.get("expected_command_count"):
        raise ValueError("Config command count differs from declared coverage")
    workflows = value.get("formal_workflows")
    if not isinstance(workflows, list) or not workflows:
        raise ValueError("Registry must declare experiment workflows")
    if value.get("expected_workflow_count") != len(workflows):
        raise ValueError("Registry workflow count differs")
    if len({w["id"] for w in workflows}) != len(workflows):
        raise ValueError("Workflow IDs must be unique")
    recipe_paths = set()
    for workflow in workflows:
        if set(workflow) != {"id", "classification", "config", "runner", "sha256"}:
            raise ValueError("Workflow fields differ from schema")
        if workflow["classification"] not in {"quantitative", "qualitative", "construction"}:
            raise ValueError("Unknown experiment classification")
        path = PROJECT_ROOT / workflow["config"]
        if file_sha256(path) != workflow["sha256"]:
            raise ValueError(f"Workflow Config digest changed: {path}")
        if not (PROJECT_ROOT / workflow["runner"]).is_file():
            raise FileNotFoundError(workflow["runner"])
        if workflow["runner"].endswith("workflows.py"):
            from .experiments.workflows import validate_recipe
            recipe = validate_recipe(path)
            if recipe["id"] != workflow["id"] or recipe["classification"] != workflow["classification"]:
                raise ValueError("Workflow metadata differs from its Config")
            recipe_paths.add(path.resolve())
    discovered_recipes = {p.resolve() for p in (PROJECT_ROOT / "configs/experiments/workflows").glob("*.json")}
    if recipe_paths != discovered_recipes:
        raise ValueError("Workflow registry differs from configured experiments")
    expected_steps = [
        dict(id=name, classification=kind, module=module, description=description)
        for name, (kind, module, description) in ENTRYPOINTS.items()
    ]
    if value.get("steps") != expected_steps:
        raise ValueError("Registry step entry points differ")
    return value


def sync_registry(path: Path = REGISTRY_PATH) -> dict[str, Any]:
    """Discover configs, refresh their metadata, and validate before replacing the index."""
    from .experiments.workflows import validate_recipe

    value = json.loads(path.read_text(encoding="utf-8"))
    previous = {entry["path"]: entry for entry in value["formal_configs"]}
    configs = []
    for job in ALL_JOBS:
        relative = job.config_path.relative_to(PROJECT_ROOT).as_posix()
        descriptor = load_experiment_descriptor(job.config_path)
        configs.append(dict(previous.get(relative, {}), path=relative,
                            sha256=file_sha256(job.config_path), runner=descriptor.runner,
                            model_adapter=descriptor.model_adapter, suites=list(descriptor.suites),
                            output_dir=json.loads(job.config_path.read_text())["output_dir"]))
    workflows = []
    for entry in value["formal_workflows"]:
        if not entry["runner"].endswith("workflows.py"):
            workflows.append(dict(entry, sha256=file_sha256(PROJECT_ROOT / entry["config"])))
    for recipe_path in sorted((PROJECT_ROOT / "configs/experiments/workflows").glob("*.json")):
        recipe = validate_recipe(recipe_path)
        workflows.append(dict(id=recipe["id"], classification=recipe["classification"],
                              config=recipe_path.relative_to(PROJECT_ROOT).as_posix(),
                              runner="sspace/experiments/workflows.py",
                              sha256=file_sha256(recipe_path)))
    commands = _job_commands(ALL_JOBS, {}, require_python=False, skip_completed=False)
    value.update(formal_configs=configs, formal_workflows=workflows,
                 expected_config_count=len(configs), expected_workflow_count=len(workflows),
                 expected_command_count=len(commands))
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json", delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    try:
        load_registry(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Print the registry")
    parser.add_argument("--sync", action="store_true", help="Register added configs and refresh checked digests")
    args = parser.parse_args()
    value = sync_registry() if args.sync else load_registry()
    if args.json:
        print(json.dumps(value, indent=2, sort_keys=True))
        return
    print(
        f"registry ok: {len(value['formal_configs'])} Configs, "
        f"{value['expected_workflow_count']} experiment workflows, "
        f"{value['expected_command_count']} Config commands"
    )
    for workflow in value["formal_workflows"]:
        print(
            f"{workflow['id']}: {workflow['classification']} "
            f"workflow ({workflow['runner']})"
        )


if __name__ == "__main__":
    main()
