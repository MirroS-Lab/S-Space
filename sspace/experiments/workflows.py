"""Plan and run registered multi-stage experiments and the complete release."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shlex
import subprocess
from pathlib import Path

from sspace.experiments.launch import DEFAULT_MODEL_PYTHONS, PROJECT_ROOT
from sspace.determinism import deterministic_cuda_environment


def validate_recipe(path: Path) -> dict:
    recipe = json.loads(path.read_text())
    if recipe.get("schema_version") != "1.0.0" or not recipe.get("id"):
        raise ValueError(f"Invalid experiment recipe: {path}")
    if recipe.get("classification") not in {"construction", "quantitative", "qualitative"}:
        raise ValueError(f"Invalid experiment classification: {path}")
    if not isinstance(recipe.get("steps"), list):
        raise ValueError(f"Recipe needs an ordered step list: {path}")
    if not recipe["steps"]:
        raise ValueError(f"Recipe has no steps: {path}")
    for step in recipe["steps"]:
        if ("module" in step) == ("script" in step):
            raise ValueError("Each step must declare one module or script")
        if not isinstance(step.get("args"), list) or not all(isinstance(a, str) for a in step["args"]):
            raise ValueError("Step arguments must be a string array")
        if "module" in step:
            if step.get("model") not in DEFAULT_MODEL_PYTHONS or importlib.util.find_spec(step["module"]) is None:
                raise ValueError(f"Invalid module or model: {step}")
        elif not (PROJECT_ROOT / step["script"]).is_file():
            raise FileNotFoundError(step["script"])
    return recipe


def case_commands(path: Path, dry_run: bool) -> list[list[str]]:
    from sspace.experiments.analysis.contextual_cases.config import load_case_config
    from sspace.experiments.analysis.contextual_cases.schema import load_case_manifest

    config = load_case_config(path, PROJECT_ROOT)
    commands = [[str(PROJECT_ROOT / DEFAULT_MODEL_PYTHONS[config.model_adapter]), "-m",
                 "sspace.experiments.analysis.contextual_cases.cli", "--config", str(path.resolve())]]
    if config.manifest_path.exists():
        samples = load_case_manifest(config.manifest_path, config.image_root)
        if all(s.condition == "web_showcase" for s in samples):
            protocol = "web_showcase"
        elif all(s.family == "political_semantics" for s in samples):
            words = {s.metadata["target_word"] for s in samples}
            protocol = "political_web" if words == {"socialist", "moderate", "conservative"} and len(samples) == 3 else "political"
        else:
            protocol = "beyond_visible"
    elif dry_run:
        protocol = "<PROTOCOL_FROM_CASE_MANIFEST>"
    else:
        raise FileNotFoundError(config.manifest_path)
    commands.append([str(PROJECT_ROOT / DEFAULT_MODEL_PYTHONS["molmo2_er"]), "-m",
                     "sspace.experiments.analysis.contextual_cases.scripts.analyze",
                     "--records", str(config.output_dir / "records.jsonl"),
                     "--output-dir", str(config.output_dir / "analysis"), "--protocol", protocol])
    return commands


def workflow_commands(entry: dict, case_configs: list[Path], dry_run: bool) -> list[list[str]]:
    if entry["runner"].endswith(".sh"):
        return [[str(PROJECT_ROOT / entry["runner"]), "--config", entry["config"]]]
    recipe = validate_recipe(PROJECT_ROOT / entry["config"])
    if recipe["id"] == "contextual-cases" and case_configs:
        return [command for path in case_configs for command in case_commands(path, dry_run)]
    commands = []
    for step in recipe["steps"]:
        prefix = ([str(PROJECT_ROOT / step["script"])] if "script" in step else
                  [str(PROJECT_ROOT / DEFAULT_MODEL_PYTHONS[step["model"]]), "-m", step["module"]])
        commands.append(prefix + step["args"])
    return commands


def plan(all_experiments: bool, experiment: str | None, case_configs: list[Path], dry_run: bool):
    from sspace.registry import load_registry

    registry = load_registry()
    if experiment == "action_supervision":
        experiment = "action_supervision/instructpart_prompt_ensemble"
    entries = registry["formal_workflows"]
    if not all_experiments:
        entries = [e for e in entries if e["id"] == experiment or e["id"].split("/")[-1] == experiment]
        if not entries:
            raise ValueError(f"Unknown experiment: {experiment}")
    # Resolve required external inputs before any expensive stage executes.
    groups = [(e["id"], workflow_commands(e, case_configs, dry_run)) for e in entries]
    if all_experiments:
        from sspace.experiments.scripts.run_suite import ALL_JOBS, _job_commands
        # Contextual manifests are built by their workflow before inference.
        jobs = [job for job in ALL_JOBS if "contextual-cases" not in job.suites]
        commands = _job_commands(jobs, {}, require_python=not dry_run, skip_completed=not dry_run)
        groups.insert(0, ("benchmark-configs", commands))
        groups.sort(key=lambda pair: 0 if pair[0] == "sspace-construction" else 1)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--experiment")
    parser.add_argument("--case-config", action="append", type=Path, default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    groups = plan(args.all, args.experiment, args.case_config, args.dry_run)
    environment = deterministic_cuda_environment()
    for name, commands in groups:
        print(f"[{name}]", flush=True)
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


if __name__ == "__main__":
    main()
