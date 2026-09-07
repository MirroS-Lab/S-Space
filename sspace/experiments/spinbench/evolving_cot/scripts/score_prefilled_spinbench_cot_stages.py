"""Score three fixed-CoT stages with direct and rotated SpinBench readouts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file

from sspace.core.artifacts import SSpaceArtifact
from sspace.experiments.spinbench.evolving_cot import STAGE_SCORE_PROTOCOL
from sspace.experiments.spinbench.perspective_taking.adapter import (
    rotate_coordinates,
    spinbench_target_margin,
)
from sspace.experiments.spinbench.evolving_cot.template import TEMPLATE_ID
from sspace.run_records import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parents[5]
INPUT_DIR = (
    PROJECT_ROOT / "outputs/reproduction/cot_evolution/activations/full"
)
ARTIFACT_DIR = PROJECT_ROOT / (
    "sspace/core/artifacts/pretrained/qwen36_27b_coco6000_final_logit_axes_v2"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "outputs/reproduction/cot_evolution/analysis"
)
STAGES = ("early", "middle", "late")
MAPPINGS = ("direct", "rotated")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _answer(score: float) -> str:
    return "A" if score > 0.0 else "B" if score < 0.0 else "tie"


def _wilson(correct: int, total: int) -> list[float]:
    z = 1.959963984540054
    center = (correct / total + z * z / (2 * total)) / (1 + z * z / total)
    radius = (
        z
        * math.sqrt(correct * (total - correct) / total**3 + z * z / (4 * total**2))
        / (1 + z * z / total)
    )
    return [center - radius, center + radius]


def _paired_exact(
    rows: list[dict[str, Any]], favored: str, comparison: str
) -> dict[str, Any]:
    favored_only = sum(row[favored] and not row[comparison] for row in rows)
    comparison_only = sum(row[comparison] and not row[favored] for row in rows)
    discordant = favored_only + comparison_only
    if discordant == 0:
        one_sided = two_sided = 1.0
    else:
        upper = (
            sum(
                math.comb(discordant, value)
                for value in range(favored_only, discordant + 1)
            )
            / 2**discordant
        )
        lower = (
            sum(math.comb(discordant, value) for value in range(favored_only + 1))
            / 2**discordant
        )
        one_sided = upper
        two_sided = min(1.0, 2.0 * min(lower, upper))
    return {
        "favored": favored,
        "comparison": comparison,
        "favored_only": favored_only,
        "comparison_only": comparison_only,
        "discordant": discordant,
        "one_sided_p": one_sided,
        "two_sided_p": two_sided,
    }


def _score_record(
    record: dict[str, Any], artifact: SSpaceArtifact, axes: np.ndarray
) -> dict[str, Any]:
    tensor_path = INPUT_DIR / record["tensor_file"]
    if _sha256(tensor_path) != record["tensor_sha256"]:
        raise ValueError(f"Tensor checksum mismatch: {record['sample_id']}")
    tensors = load_file(tensor_path)
    states = tensors["post_block_states"].astype(np.float64)
    saved = tensors["projection_hvd"].astype(np.float64)
    if states.shape != (6, artifact.manifest.model.hidden_size):
        raise ValueError(f"Invalid hidden-state shape: {record['sample_id']}")
    if saved.shape != (6, 3) or not np.isfinite(states).all():
        raise ValueError(f"Invalid projection tensors: {record['sample_id']}")
    projected = states @ axes.T.astype(np.float64)
    if not np.allclose(projected, saved, rtol=2e-5, atol=2e-5):
        raise ValueError(f"Saved projection mismatch: {record['sample_id']}")

    mentions = sorted(
        record["mentions"], key=lambda item: item["generated_token_start"]
    )
    columns: dict[str, list[int]] = {"object_a": [], "object_b": []}
    for mention in mentions:
        columns[mention["role"]].append(int(mention["activation_column"]))
    if {role: len(values) for role, values in columns.items()} != {
        "object_a": 3,
        "object_b": 3,
    }:
        raise ValueError(f"Expected three mentions per object: {record['sample_id']}")

    output = {
        key: record[key]
        for key in (
            "sample_index",
            "sample_id",
            "object_a",
            "object_b",
            "gold_answer",
            "target_view",
            "target_property",
        )
    }
    gold_sign = 1.0 if record["gold_answer"] == "A" else -1.0
    for stage_index, stage in enumerate(STAGES):
        coordinate_a = projected[columns["object_a"][stage_index]]
        coordinate_b = projected[columns["object_b"][stage_index]]
        pair = coordinate_a - coordinate_b
        output[f"{stage}_coordinate_a"] = coordinate_a.tolist()
        output[f"{stage}_coordinate_b"] = coordinate_b.tolist()
        output[f"{stage}_pair"] = pair.tolist()
        for mapping in MAPPINGS:
            readout = (
                pair
                if mapping == "direct"
                else rotate_coordinates(pair, record["target_view"])
            )
            score = float(spinbench_target_margin(readout, record["target_property"]))
            answer = _answer(score)
            prefix = f"{stage}_{mapping}"
            output[f"{prefix}_score"] = score
            output[f"{prefix}_gold_aligned_margin"] = gold_sign * score
            output[f"{prefix}_answer"] = answer
            output[f"{prefix}_correct"] = answer == record["gold_answer"]
    return output


def _metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    correct = sum(bool(row[f"{key}_correct"]) for row in rows)
    margins = np.asarray(
        [row[f"{key}_gold_aligned_margin"] for row in rows], dtype=np.float64
    )
    return {
        "correct": correct,
        "total": len(rows),
        "accuracy": correct / len(rows),
        "wilson_95": _wilson(correct, len(rows)),
        "gold_aligned_margin_mean": float(margins.mean()),
        "gold_aligned_margin_median": float(np.median(margins)),
    }


def _metric_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        stage: {mapping: _metric(rows, f"{stage}_{mapping}") for mapping in MAPPINGS}
        for stage in STAGES
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _metric_block(rows)
    direct = [metrics[stage]["direct"]["accuracy"] for stage in STAGES]
    rotated = [metrics[stage]["rotated"]["accuracy"] for stage in STAGES]
    grouped = {}
    for field in ("target_view", "target_property"):
        grouped[field] = {}
        for value in sorted({row[field] for row in rows}):
            subset = [row for row in rows if row[field] == value]
            grouped[field][value] = {
                "sample_count": len(subset),
                "metrics": _metric_block(subset),
            }
    return {
        "protocol": STAGE_SCORE_PROTOCOL,
        "template_id": TEMPLATE_ID,
        "sample_count": len(rows),
        "selected_layer": 43,
        "metrics": metrics,
        "accuracy_deltas": {
            "direct_middle_minus_early": direct[1] - direct[0],
            "direct_late_minus_middle": direct[2] - direct[1],
            "direct_late_minus_early": direct[2] - direct[0],
            "rotated_middle_minus_early": rotated[1] - rotated[0],
            "rotated_late_minus_middle": rotated[2] - rotated[1],
            "rotated_late_minus_early": rotated[2] - rotated[0],
            "crossover_interaction": (direct[2] - direct[0])
            - (rotated[2] - rotated[0]),
        },
        "aggregate_hypothesis": {
            "direct_nondecreasing": direct[0] <= direct[1] <= direct[2],
            "rotated_nonincreasing": rotated[0] >= rotated[1] >= rotated[2],
            "positive_crossover": (direct[2] - direct[0]) - (rotated[2] - rotated[0])
            > 0,
        },
        "paired_exact_tests": {
            "middle_direct_over_early_direct": _paired_exact(
                rows, "middle_direct_correct", "early_direct_correct"
            ),
            "late_direct_over_middle_direct": _paired_exact(
                rows, "late_direct_correct", "middle_direct_correct"
            ),
            "late_direct_over_early_direct": _paired_exact(
                rows, "late_direct_correct", "early_direct_correct"
            ),
            "early_rotated_over_middle_rotated": _paired_exact(
                rows, "early_rotated_correct", "middle_rotated_correct"
            ),
            "middle_rotated_over_late_rotated": _paired_exact(
                rows, "middle_rotated_correct", "late_rotated_correct"
            ),
            "early_rotated_over_late_rotated": _paired_exact(
                rows, "early_rotated_correct", "late_rotated_correct"
            ),
        },
        "grouped": grouped,
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "sample_id",
        "target_view",
        "target_property",
        "gold_answer",
    ]
    for stage in STAGES:
        for mapping in MAPPINGS:
            fields.extend(
                (
                    f"{stage}_{mapping}_score",
                    f"{stage}_{mapping}_gold_aligned_margin",
                    f"{stage}_{mapping}_answer",
                    f"{stage}_{mapping}_correct",
                )
            )
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(summary: dict[str, Any]) -> None:
    lines = [
        "# Qwen3.6-27B 固定 CoT 三阶段空间读出",
        "",
        "该实验是使用正确答案构造固定 CoT 的激活干预，不是独立 benchmark 准确率。",
        "",
        "| 阶段 | 直接映射 | 旋转后映射 |",
        "|---|---:|---:|",
    ]
    for stage in STAGES:
        direct = summary["metrics"][stage]["direct"]
        rotated = summary["metrics"][stage]["rotated"]
        lines.append(
            f"| {stage} | {direct['correct']}/{direct['total']} "
            f"({100 * direct['accuracy']:.1f}%) | {rotated['correct']}/{rotated['total']} "
            f"({100 * rotated['accuracy']:.1f}%) |"
        )
    delta = summary["accuracy_deltas"]
    lines.extend(
        (
            "",
            f"Crossover interaction = {100 * delta['crossover_interaction']:.1f} 个百分点。",
            "",
            "每个阶段都计算 `p = c_A - c_B`。直接映射从 `p` 读取题目方向；"
            "旋转映射先计算 `R_view @ p`，再读取同一方向。全部坐标使用固定的 "
            "COCO-1800 选层 L43 和同一组单位 H/V/D 轴。",
        )
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    if OUTPUT_DIR.exists():
        raise FileExistsError(OUTPUT_DIR)
    artifact = SSpaceArtifact.load(ARTIFACT_DIR)
    layer = artifact.manifest.model.selected_layer
    if layer != 43:
        raise ValueError(f"Expected COCO-1800 selected L43, found L{layer}")
    axes = artifact.axes(layer)
    records_path = INPUT_DIR / "records.jsonl"
    source_summary = json.loads(
        (INPUT_DIR / "summary.json").read_text(encoding="utf-8")
    )
    if source_summary.get("template_id") != TEMPLATE_ID:
        raise ValueError("Expected the final native_exact_winner CoT template")
    records = _read_jsonl(records_path)
    if len(records) != 146 or source_summary["sample_count"] != 146:
        raise ValueError("Expected the complete 146-row fixed-CoT intervention")
    if source_summary["records_sha256"] != _sha256(records_path):
        raise ValueError("Source record checksum mismatch")
    rows = [_score_record(record, artifact, axes) for record in records]
    summary = _summary(rows)
    OUTPUT_DIR.mkdir(parents=True)
    _write_jsonl(OUTPUT_DIR / "records.jsonl", rows)
    _write_csv(OUTPUT_DIR / "records.csv", rows)
    atomic_json(OUTPUT_DIR / "summary.json", summary)
    _write_report(summary)
    outputs = ("records.jsonl", "records.csv", "summary.json", "report.md")
    atomic_json(
        OUTPUT_DIR / "provenance.json",
        {
            "protocol": summary["protocol"],
            "artifact_id": artifact.manifest.artifact_id,
            "selected_layer": layer,
            "source_records": str(records_path.relative_to(PROJECT_ROOT)),
            "source_records_sha256": _sha256(records_path),
            "coordinate_formula": "c = h_object_last_subtoken @ axes.T",
            "pair_formula": "p_stage = c_A_stage - c_B_stage",
            "direct_formula": "margin(target_property, p_stage)",
            "rotated_formula": "margin(target_property, R_view @ p_stage)",
            "checksums": {name: _sha256(OUTPUT_DIR / name) for name in outputs},
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
