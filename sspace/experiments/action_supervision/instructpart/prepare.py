"""Prepare the frozen inputs for the InstructPart quantitative experiment."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from sspace.run_records import canonical_fingerprint, file_sha256

from .config import DEFAULT_CONFIG, load_config, project_path


def _value_fingerprint(path: Path) -> str:
    value = json.loads(path.read_text(encoding="utf-8"))
    core = {
        "rows": [
            {name: item for name, item in row.items() if name != "depth_map"}
            for row in value["rows"]
        ],
    }
    return canonical_fingerprint(core)


def _run(command: list[str], dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    dataset = config["dataset"]
    ground_truth = config["pseudo_ground_truth"]
    device = args.device or config["evaluation"]["device"]
    root = project_path(dataset["dataset_root"])
    source = root / "source"
    filtered = root / "filtered"
    pseudo = root / "pseudo_ground_truth"
    source_command = [
        sys.executable,
        "-m",
        "sspace.data.instructpart.acquisition",
        "--output-dir",
        str(source),
        "--count",
        str(dataset["candidate_count"]),
        "--seed",
        str(dataset["seed"]),
        "--workers",
        str(args.workers),
    ]
    _run(source_command, args.dry_run)
    if args.dry_run:
        prepared = False
    else:
        required = {
            pseudo / "retained_selection.json": ground_truth[
                "expected_retained_selection_sha256"
            ],
            filtered / "selection.json": ground_truth[
                "expected_filtered_selection_sha256"
            ],
            filtered / "object_boxes.json": ground_truth[
                "expected_object_boxes_sha256"
            ],
        }
        prepared = all(
            path.is_file() and file_sha256(path) == expected
            for path, expected in required.items()
        )
        pseudo_path = pseudo / "pseudo_ground_truth.json"
        prepared = (
            prepared
            and pseudo_path.is_file()
            and _value_fingerprint(pseudo_path)
            == ground_truth["expected_ground_truth_values_sha256"]
        )
    if prepared:
        print("verified existing InstructPart boxes and pseudo-GT", flush=True)
        return
    box_command = [
        sys.executable,
        "-m",
        "sspace.data.instructpart.object_boxes",
        "--selection",
        str(source / "selection.json"),
        "--assets-dir",
        str(source / "assets"),
        "--model-path",
        str(project_path(ground_truth["grounding_dino_path"])),
        "--output-dir",
        str(filtered),
        "--device",
        device,
        "--box-threshold",
        str(ground_truth["box_threshold"]),
        "--text-threshold",
        str(ground_truth["text_threshold"]),
        "--minimum-coverage",
        str(ground_truth["minimum_part_mask_coverage"]),
        "--minimum-count",
        str(ground_truth["retained_count"]),
        "--expected-weights-sha256",
        ground_truth["grounding_dino_weights_sha256"],
        "--expected-selection-sha256",
        ground_truth["expected_filtered_selection_sha256"],
        "--expected-boxes-sha256",
        ground_truth["expected_object_boxes_sha256"],
    ]
    _run(box_command, args.dry_run)
    pseudo_command = [
        sys.executable,
        "-m",
        "sspace.data.instructpart.pseudo_ground_truth",
        "--selection",
        str(filtered / "selection.json"),
        "--boxes",
        str(filtered / "object_boxes.json"),
        "--assets-dir",
        str(source / "assets"),
        "--sam-model",
        str(project_path(ground_truth["slimsam_path"])),
        "--depth-model",
        str(project_path(ground_truth["depth_model_path"])),
        "--output-dir",
        str(pseudo),
        "--device",
        device,
        "--target-count",
        str(ground_truth["retained_count"]),
        "--hv-dead-zone",
        str(ground_truth["horizontal_vertical_dead_zone"]),
        "--depth-dead-zone",
        str(ground_truth["depth_dead_zone"]),
        "--sam-weights-sha256",
        ground_truth["slimsam_weights_sha256"],
        "--depth-weights-sha256",
        ground_truth["depth_weights_sha256"],
        "--expected-retained-sha256",
        ground_truth["expected_retained_selection_sha256"],
        "--expected-values-sha256",
        ground_truth["expected_ground_truth_values_sha256"],
    ]
    _run(pseudo_command, args.dry_run)


if __name__ == "__main__":
    main()
