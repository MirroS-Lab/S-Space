"""Load the formal InstructPart action-supervision configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/action_supervision/instructpart_prompt_ensemble.json"
)
MODEL_ORDER = ("molmo2_er", "molmoact2_pretrain", "molmoact2")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate the complete quantitative protocol declaration."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "classification",
        "protocol",
        "dataset",
        "pseudo_ground_truth",
        "evaluation",
    }
    if set(value) != required:
        raise ValueError("InstructPart config fields differ from schema 1")
    if (
        value["schema_version"] != "1.0.0"
        or value["classification"] != "quantitative"
    ):
        raise ValueError("InstructPart config identity differs")
    if value["protocol"] != "instructpart_named_part_prompt_ensemble_10":
        raise ValueError("InstructPart prompt protocol differs")
    dataset = value["dataset"]
    expected_dataset = {
        "repo_id": "IffYuan/InstructPart",
        "revision": "bcd06969e32582ceeba3e1841183106027b379fb",
        "split": "train",
        "candidate_count": 360,
        "seed": 42,
        "expected_selection_sha256": (
            "cb6d668645f7e5bbf40bb34f0ee00af95940141c89fbfee6e9dacf9c13d08c71"
        ),
    }
    for name, expected in expected_dataset.items():
        if dataset.get(name) != expected:
            raise ValueError(f"InstructPart dataset {name} differs")
    if set(value["evaluation"]["models"]) != set(MODEL_ORDER):
        raise ValueError("InstructPart config must declare exactly three models")
    layers = value["evaluation"]["layers"]
    if layers != [17, 18, 19, 20, 21, 22]:
        raise ValueError("Historical comparison requires diagnostic layers 17--22")
    return value


def project_path(value: str) -> Path:
    """Resolve one config path against the independent publication root."""
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
