#!/usr/bin/env python3
"""Run the four canonical EmbSpatial H/V spatial-causality experiments."""

from __future__ import annotations

import argparse
import gc
import io
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from sspace.experiments.identity import runtime_identity
from sspace.run_records import file_sha256

from .spatial_jacobian.constants import (
    DATASET_REVISIONS,
    DEFAULT_ANNOTATION_PATH,
    DEFAULT_HF_HOME,
    MODEL_ID,
    MODEL_REVISION,
    NEGATIVE_ANSWER,
    OPPOSITE,
    POSITIVE_ANSWER,
    PROJECT_DIR,
    TEMPLATES,
)
from .spatial_jacobian.dataset import Sample, load_samples
from .spatial_jacobian.spatial_causality import (
    ARTIFACT_ID,
    AXIS_INDEX,
    CoordinateEditHook,
    coordinate_right_inverse,
    load_axes_artifact,
)
from .spatial_jacobian.io_utils import atomic_csv, atomic_json
from .spatial_jacobian.modeling import (
    PromptJob,
    load_model,
    locate_object_tokens,
    move_inputs,
    prepare_batch,
    resolve_model_path,
)


EXPERIMENTS = (
    "relative_swap",
    "relative_common_shift",
    "absolute_shift",
    "nonspatial_shift",
)
HORIZONTAL_VERTICAL = ("left", "right", "above", "below")
ABSOLUTE_CONFIG = {
    "horizontal": {
        "negative": "left",
        "positive": "right",
        "template": (
            "Is the {obj} to the left or right of the image center? "
            "Answer with only one word."
        ),
    },
    "vertical": {
        "negative": "above",
        "positive": "below",
        "template": (
            "Is the {obj} above or below the image center? Answer with only one word."
        ),
    },
}
COLOR_TEMPLATE = "What color is the {obj}? Answer with only one word."
DEFAULT_AXES = (
    PROJECT_DIR
    / "sspace"
    / "core"
    / "artifacts"
    / "pretrained"
    / ARTIFACT_ID
    / "tensors.safetensors"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["all"],
        choices=("all", *EXPERIMENTS),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["17"],
        help="Layer ids, inclusive ranges such as 13:21, or all (default: 17).",
    )
    parser.add_argument("--alpha-values", nargs="+", type=float, default=[5.0])
    parser.add_argument("--beta-values", nargs="+", type=float, default=[-30.0, 30.0])
    parser.add_argument("--samples-per-category", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_DIR / "data" / "processed" / "qa_embspatial.sqlite3",
    )
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--axes", type=Path, default=DEFAULT_AXES)
    parser.add_argument(
        "--selection",
        type=Path,
        help="Four-case selection.json; runs a dense showcase sweep and bundle.",
    )
    parser.add_argument("--showcase-points", type=int, default=101)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hf-home", type=Path, default=DEFAULT_HF_HOME)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def parse_layers(values: Sequence[str], available: Sequence[int]) -> list[int]:
    tokens = [part for value in values for part in value.split(",") if part]
    if tokens == ["all"]:
        return list(available)
    if "all" in tokens:
        raise ValueError("Use --layers all by itself")
    layers: list[int] = []
    for token in tokens:
        if ":" in token:
            start_text, stop_text = token.split(":", 1)
            start, stop = int(start_text), int(stop_text)
            if stop < start:
                raise ValueError(f"Descending layer range {token!r}")
            layers.extend(range(start, stop + 1))
        else:
            layers.append(int(token))
    layers = list(dict.fromkeys(layers))
    missing = sorted(set(layers).difference(available))
    if not layers or missing:
        raise ValueError(f"Axes artifact does not publish layers {missing}")
    return layers


def parse_experiments(values: Sequence[str]) -> list[str]:
    if "all" in values:
        if len(values) != 1:
            raise ValueError("Use --experiments all by itself")
        return list(EXPERIMENTS)
    return list(dict.fromkeys(values))


def finite_values(values: Sequence[float], name: str) -> list[float]:
    output = list(dict.fromkeys(float(value) for value in values))
    if not output or not all(math.isfinite(value) for value in output):
        raise ValueError(f"{name} values must be non-empty and finite")
    return output


def default_annotations(hf_home: Path) -> Path:
    if DEFAULT_ANNOTATION_PATH.is_file():
        return DEFAULT_ANNOTATION_PATH
    source = DATASET_REVISIONS["embspatial_annotations"]
    return (
        hf_home
        / "hub"
        / "datasets--FlagEval--EmbSpatial-Bench"
        / "snapshots"
        / source["revision"]
        / source["filename"]
    )


def select_samples(database: Path, count: int, seed: int) -> list[Sample]:
    if count <= 0:
        raise ValueError("samples-per-category must be positive")
    source = load_samples(database, ("embspatial",))
    rng = random.Random(seed)
    selected: list[Sample] = []
    for category in HORIZONTAL_VERTICAL:
        rows = [sample for sample in source if sample.category == category]
        if len(rows) < count:
            raise ValueError(
                f"Need {count} EmbSpatial {category} rows, found {len(rows)}"
            )
        selected.extend(rng.sample(rows, count) if len(rows) > count else rows)
    counts = Counter(sample.category for sample in selected)
    if counts != Counter({category: count for category in HORIZONTAL_VERTICAL}):
        raise ValueError(f"Unexpected EmbSpatial H/V population: {counts}")
    return selected


def base_cases(samples: Sequence[Sample]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "case_index": index,
                "dataset": sample.dataset,
                "idx": sample.index,
                "question_id": sample.question_id,
                "category": sample.category,
                "group": sample.group,
                "query_object": sample.obj1,
                "reference_object": sample.obj2,
                "relative_question": sample.original_question,
            }
            for index, sample in enumerate(samples)
        ]
    )


def add_absolute_labels(
    cases: pd.DataFrame,
    samples: Sequence[Sample],
    annotations: Path,
) -> pd.DataFrame:
    frame = pd.read_parquet(
        annotations,
        columns=["data_source", "question_id", "objects"],
    )
    rows = {str(row.question_id): row for row in frame.itertuples(index=False)}
    additions: list[dict[str, Any]] = []
    for sample in samples:
        row = rows.get(sample.question_id)
        if row is None:
            raise ValueError(f"Missing annotation for {sample.question_id}")
        names = [str(value) for value in row.objects["name"]]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate object names for {sample.question_id}")
        boxes = {
            name: [float(value) for value in box]
            for name, box in zip(names, row.objects["bbox"], strict=True)
        }
        if sample.obj1 not in boxes or sample.obj2 not in boxes:
            raise ValueError(f"Object annotation mismatch for {sample.question_id}")
        with Image.open(io.BytesIO(sample.image_bytes)) as image:
            width, height = image.size
        source = str(row.data_source)

        def center(name: str) -> tuple[float, float]:
            x0, y0, third, fourth = boxes[name]
            if source == "ai2thor":
                x, y = (x0 + third) / 2, (y0 + fourth) / 2
            elif source in {"mp3d", "scannet"}:
                x, y = x0 + third / 2, y0 + fourth / 2
            else:
                raise ValueError(f"Unknown EmbSpatial source {source!r}")
            return x / width, y / height

        query_u, query_v = center(sample.obj1)
        reference_u, reference_v = center(sample.obj2)
        values = (query_u, query_v, reference_u, reference_v)
        if not all(0 <= value <= 1 for value in values):
            raise ValueError(f"Invalid object centers for {sample.question_id}")
        relation = (
            ("left" if query_u < reference_u else "right")
            if sample.group == "horizontal"
            else ("above" if query_v < reference_v else "below")
        )
        if relation != sample.category:
            raise ValueError(f"BBox relation mismatch for {sample.question_id}")
        coordinate = query_u if sample.group == "horizontal" else query_v
        absolute_answer = (
            ("left" if coordinate < 0.5 else "right")
            if sample.group == "horizontal"
            else ("above" if coordinate < 0.5 else "below")
        )
        additions.append(
            {
                "source": source,
                "query_u": query_u,
                "query_v": query_v,
                "reference_u": reference_u,
                "reference_v": reference_v,
                "absolute_answer": absolute_answer,
                "clear_center_0p1": abs(coordinate - 0.5) >= 0.1,
            }
        )
    return pd.concat((cases.reset_index(drop=True), pd.DataFrame(additions)), axis=1)


def apply_selection(
    samples: Sequence[Sample],
    cases: pd.DataFrame,
    path: Path,
) -> tuple[list[Sample], pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = list(payload.get("cases", []))
    if len(selected) != 4 or {str(row.get("category")) for row in selected} != set(
        HORIZONTAL_VERTICAL
    ):
        raise ValueError("Showcase selection must contain one case per H/V category")
    sample_output: list[Sample] = []
    case_output: list[dict[str, Any]] = []
    showcase_ids: set[str] = set()
    indexed = cases.set_index("case_index", verify_integrity=True)
    for order, spec in enumerate(selected):
        index = int(spec["case_index"])
        if not 0 <= index < len(samples) or index not in indexed.index:
            raise ValueError(f"Invalid selected case index {index}")
        sample = samples[index]
        expected = (
            str(spec["question_id"]),
            str(spec["category"]),
            str(spec["group"]),
            str(spec["query_object"]),
            str(spec["reference_object"]),
        )
        actual = (
            sample.question_id,
            sample.category,
            sample.group,
            sample.obj1,
            sample.obj2,
        )
        if actual != expected:
            raise ValueError(f"Selection differs from frozen case {index}")
        showcase_id = str(spec.get("showcase_id", f"case_{index:03d}"))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", showcase_id):
            raise ValueError(f"Invalid showcase id {showcase_id!r}")
        if showcase_id in showcase_ids:
            raise ValueError(f"Duplicate showcase id {showcase_id!r}")
        showcase_ids.add(showcase_id)
        case = indexed.loc[index].to_dict()
        case.update(
            {
                "case_index": index,
                "showcase_order": order,
                "showcase_id": showcase_id,
            }
        )
        sample_output.append(sample)
        case_output.append(case)
    return sample_output, pd.DataFrame(case_output)


def answer_token_ids(tokenizer: Any) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for word in HORIZONTAL_VERTICAL:
        ids = []
        for form in (word, word.title()):
            encoded = tokenizer.encode(form, add_special_tokens=False)
            if len(encoded) != 1:
                raise ValueError(f"Answer form {form!r} is not one token")
            ids.append(int(encoded[0]))
        output[word] = list(dict.fromkeys(ids))
    return output


def binary_scores(
    logits: torch.Tensor,
    groups: Sequence[str],
    token_ids: Mapping[str, Sequence[int]],
) -> list[dict[str, Any]]:
    logits = logits.float()
    normalizer = torch.logsumexp(logits, dim=-1)
    records: list[dict[str, Any]] = []
    for row, group in enumerate(groups):
        negative, positive = NEGATIVE_ANSWER[group], POSITIVE_ANSWER[group]
        negative_score = torch.logsumexp(logits[row, list(token_ids[negative])], dim=-1)
        positive_score = torch.logsumexp(logits[row, list(token_ids[positive])], dim=-1)
        margin = positive_score - negative_score
        negative_vocab = torch.exp(negative_score - normalizer[row])
        positive_vocab = torch.exp(positive_score - normalizer[row])
        records.append(
            {
                "negative_answer": negative,
                "positive_answer": positive,
                "negative_score": float(negative_score),
                "positive_score": float(positive_score),
                "positive_minus_negative_margin": float(margin),
                "p_negative_pair": float(torch.sigmoid(-margin)),
                "p_positive_pair": float(torch.sigmoid(margin)),
                "p_negative_vocab": float(negative_vocab),
                "p_positive_vocab": float(positive_vocab),
                "answer_vocab_mass": float(negative_vocab + positive_vocab),
                "prediction": positive if float(margin) > 0 else negative,
            }
        )
    return records


def prepare_relative(
    processor: Any,
    samples: Sequence[Sample],
) -> tuple[
    dict[str, torch.Tensor], list[dict[str, int]], list[dict[str, str]], list[str]
]:
    jobs = [
        PromptJob(index, "baseline", "original", sample.group, sample.category)
        for index, sample in enumerate(samples)
    ]
    inputs, positions, pieces, questions = prepare_batch(processor, samples, jobs)
    for sample, question in zip(samples, questions, strict=True):
        if question != sample.original_question:
            raise ValueError(f"Relative prompt mismatch for {sample.question_id}")
    return inputs, positions, pieces, questions


def render_single_prompt(
    template: str, object_name: str
) -> tuple[str, tuple[int, int]]:
    if template.count("{obj}") != 1:
        raise ValueError("Single-object template must contain one {obj} field")
    prefix, suffix = template.split("{obj}")
    return f"{prefix}{object_name}{suffix}", (
        len(prefix),
        len(prefix) + len(object_name),
    )


def prepare_single(
    processor: Any,
    samples: Sequence[Sample],
    templates: Sequence[str],
) -> tuple[dict[str, torch.Tensor], list[int], list[str], list[str]]:
    if len(samples) != len(templates):
        raise ValueError("Every sample requires a single-object template")
    conversations = []
    rendered = []
    for sample, template in zip(samples, templates, strict=True):
        question, span = render_single_prompt(template, sample.obj1)
        image = Image.open(io.BytesIO(sample.image_bytes)).convert("RGB")
        conversations.append(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question},
                    ],
                }
            ]
        )
        rendered.append((question, span))
    inputs = processor.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        padding=True,
    )
    positions, pieces = [], []
    for row, (question, span) in enumerate(rendered):
        located, token_pieces = locate_object_tokens(
            processor.tokenizer,
            inputs["input_ids"][row].tolist(),
            question,
            {"obj": span},
        )
        position = int(located["obj"])
        if position >= inputs["input_ids"].shape[1] - 1:
            raise ValueError("Object token must precede the final prompt token")
        positions.append(position)
        pieces.append(str(token_pieces["obj"]))
    return inputs, positions, pieces, [question for question, _ in rendered]


def identity_record(case: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "case_index",
        "dataset",
        "idx",
        "question_id",
        "category",
        "group",
        "query_object",
        "reference_object",
    )
    return {key: case[key] for key in keys}


def templates_for(experiment: str, samples: Sequence[Sample]) -> list[str]:
    if experiment == "absolute_shift":
        return [str(ABSOLUTE_CONFIG[sample.group]["template"]) for sample in samples]
    if experiment == "nonspatial_shift":
        return [COLOR_TEMPLATE] * len(samples)
    raise ValueError(f"No single-object template for {experiment}")


def progress_frame(
    path: Path,
    cases: pd.DataFrame,
    resume: bool,
    expected: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if not resume or not path.is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if len(frame) > len(cases):
        raise ValueError(f"Saved stage has too many rows: {path}")
    required = {
        "case_index",
        "question_id",
        "idx",
        "category",
        "group",
        "query_object",
        "reference_object",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Saved stage is missing columns {missing}: {path}")
    wanted = cases["case_index"].astype(int).tolist()[: len(frame)]
    if frame["case_index"].astype(int).tolist() != wanted:
        raise ValueError(f"Saved stage uses different cases: {path}")
    if (
        frame["question_id"].astype(str).tolist()
        != cases["question_id"].astype(str).tolist()[: len(frame)]
    ):
        raise ValueError(f"Saved stage uses different question ids: {path}")
    for column in (
        "idx",
        "category",
        "group",
        "query_object",
        "reference_object",
    ):
        if frame[column].tolist() != cases[column].tolist()[: len(frame)]:
            raise ValueError(f"Saved stage uses different {column}: {path}")
    for key, value in (expected or {}).items():
        if not len(frame):
            continue
        if key not in frame:
            raise ValueError(f"Saved stage is missing {key}: {path}")
        values = frame[key]
        matches = (
            np.isclose(values.astype(float), value, rtol=1e-15, atol=0.0)
            if isinstance(value, float)
            else values.eq(value)
        )
        if not bool(np.asarray(matches).all()):
            raise ValueError(f"Saved stage has different {key}: {path}")
    return frame


def validate_cases(path: Path, current: pd.DataFrame) -> None:
    saved = pd.read_csv(path)
    if list(saved.columns) != list(current.columns) or len(saved) != len(current):
        raise ValueError(f"Saved run uses a different case table: {path}")
    for column in current.columns:
        left, right = saved[column], current[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            matches = np.isclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
                equal_nan=True,
            )
        else:
            matches = (left.isna() & right.isna()) | left.astype(str).eq(
                right.astype(str)
            )
        if not bool(np.asarray(matches).all()):
            raise ValueError(f"Saved run has different {column}: {path}")


def baseline_path(output_dir: Path, task: str) -> Path:
    return output_dir / "baselines" / f"{task}.csv"


def value_key(value: float) -> str:
    return format(value, ".17g")


def value_label(value: float) -> str:
    return value_key(value).replace("-", "m").replace("+", "").replace(".", "p")


def stage_path(
    output_dir: Path,
    experiment: str,
    layer: int,
    parameter: str,
    value: float,
) -> Path:
    return (
        output_dir
        / experiment
        / f"L{layer}"
        / f"{parameter}{value_label(value)}"
        / "records.csv"
    )


def run_binary_baseline(
    task: str,
    processor: Any,
    model: torch.nn.Module,
    samples: Sequence[Sample],
    cases: pd.DataFrame,
    token_ids: Mapping[str, Sequence[int]],
    batch_size: int,
    path: Path,
    resume: bool,
) -> pd.DataFrame:
    records = progress_frame(path, cases, resume)
    start = len(records)
    rows = records.to_dict("records")
    progress = tqdm(total=len(samples), initial=start, desc=f"baseline/{task}")
    while start < len(samples):
        stop = min(start + batch_size, len(samples))
        batch_samples = samples[start:stop]
        if task == "relative":
            inputs, _, pieces, questions = prepare_relative(processor, batch_samples)
        else:
            inputs, _, single_pieces, questions = prepare_single(
                processor,
                batch_samples,
                templates_for("absolute_shift", batch_samples),
            )
            pieces = [{"obj1": piece} for piece in single_pieces]
        with torch.inference_mode():
            output = model(
                **move_inputs(inputs, model),
                use_cache=False,
                logits_to_keep=1,
                return_dict=True,
            )
        scores = binary_scores(
            output.logits[:, -1],
            [sample.group for sample in batch_samples],
            token_ids,
        )
        for offset, (sample, piece, question, score) in enumerate(
            zip(
                batch_samples,
                pieces,
                questions,
                scores,
                strict=True,
            )
        ):
            case = cases.iloc[start + offset].to_dict()
            answer = (
                sample.category if task == "relative" else str(case["absolute_answer"])
            )
            rows.append(
                {
                    **identity_record(case),
                    "task": task,
                    "question": question,
                    "query_token": piece["obj1"],
                    "reference_token": piece.get("obj2", ""),
                    **score,
                    "correct": score["prediction"] == answer,
                }
            )
        start = stop
        atomic_csv(path, pd.DataFrame(rows))
        progress.update(stop - progress.n)
    progress.close()
    return pd.DataFrame(rows)


def normalize_color(raw: str) -> str:
    value = re.sub(r"\s+", " ", raw.strip()).lower()
    return re.sub(r"^[^a-z]+|[^a-z]+$", "", value)


def color_rows(
    processor: Any,
    generated: Any,
    input_width: int,
) -> list[dict[str, Any]]:
    if not generated.scores:
        raise RuntimeError("Generation returned no scores")
    new_tokens = generated.sequences[:, input_width:]
    decoded = processor.tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    first_ids = new_tokens[:, 0].detach().cpu()
    first_logits = generated.scores[0].float()
    first_log_probs = torch.log_softmax(first_logits, dim=-1).detach().cpu()
    top_scores, top_ids = torch.topk(first_logits, k=5, dim=-1)
    normalizer = torch.logsumexp(first_logits, dim=-1)
    output = []
    for row, (raw_answer, first_id) in enumerate(
        zip(decoded, first_ids.tolist(), strict=True)
    ):
        raw = raw_answer.strip()
        top5 = [
            {
                "token_id": int(token_id),
                "token": processor.tokenizer.decode([int(token_id)]),
                "probability": float(torch.exp(score - normalizer[row])),
            }
            for score, token_id in zip(top_scores[row], top_ids[row], strict=True)
        ]
        logp = float(first_log_probs[row, int(first_id)])
        output.append(
            {
                "raw_answer": raw,
                "normalized_answer": normalize_color(raw),
                "word_count": len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", raw)),
                "first_token_id": int(first_id),
                "first_token": processor.tokenizer.decode([int(first_id)]),
                "logp_generated_first_token": logp,
                "p_generated_first_token": math.exp(logp),
                "top5": json.dumps(top5, ensure_ascii=False),
            }
        )
    return output


def run_color_baseline(
    processor: Any,
    model: torch.nn.Module,
    samples: Sequence[Sample],
    cases: pd.DataFrame,
    batch_size: int,
    max_new_tokens: int,
    path: Path,
    resume: bool,
) -> pd.DataFrame:
    records = progress_frame(path, cases, resume)
    start = len(records)
    rows = records.to_dict("records")
    progress = tqdm(total=len(samples), initial=start, desc="baseline/color")
    while start < len(samples):
        stop = min(start + batch_size, len(samples))
        batch_samples = samples[start:stop]
        inputs, _, pieces, questions = prepare_single(
            processor,
            batch_samples,
            templates_for("nonspatial_shift", batch_samples),
        )
        moved = move_inputs(inputs, model)
        input_width = int(moved["input_ids"].shape[1])
        with torch.inference_mode():
            generated = model.generate(
                **moved,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        scored = color_rows(processor, generated, input_width)
        for offset, (piece, question, score) in enumerate(
            zip(
                pieces,
                questions,
                scored,
                strict=True,
            )
        ):
            case = cases.iloc[start + offset].to_dict()
            rows.append(
                {
                    **identity_record(case),
                    "task": "color",
                    "question": question,
                    "query_token": piece,
                    **score,
                }
            )
        start = stop
        atomic_csv(path, pd.DataFrame(rows))
        progress.update(stop - progress.n)
    progress.close()
    return pd.DataFrame(rows)


def encoded_mechanics(hook: CoordinateEditHook, row: int, axis: int) -> dict[str, Any]:
    values = hook.mechanics(row, axis)
    return {
        key: json.dumps(value) if isinstance(value, list) else value
        for key, value in values.items()
    }


def run_binary_stage(
    experiment: str,
    layer: int,
    parameter: str,
    value: float,
    processor: Any,
    model: torch.nn.Module,
    samples: Sequence[Sample],
    cases: pd.DataFrame,
    baseline: pd.DataFrame,
    token_ids: Mapping[str, Sequence[int]],
    factors: torch.Tensor,
    basis: torch.Tensor,
    batch_size: int,
    path: Path,
    resume: bool,
) -> pd.DataFrame:
    expected = {
        "experiment": experiment,
        "source_layer": layer,
        "parameter": parameter,
        "value": float(value),
        "value_key": value_key(value),
    }
    records = progress_frame(path, cases, resume, expected)
    start = len(records)
    rows = records.to_dict("records")
    mode = "single_shift" if experiment == "absolute_shift" else experiment
    progress = tqdm(
        total=len(samples),
        initial=start,
        desc=f"{experiment}/L{layer}/{parameter}={value:g}",
    )
    block_device = next(model.model.transformer.blocks[layer].parameters()).device
    factors_device, basis_device = factors.to(block_device), basis.to(block_device)
    with CoordinateEditHook(model, layer) as hook:
        while start < len(samples):
            stop = min(start + batch_size, len(samples))
            batch_samples = samples[start:stop]
            if experiment == "absolute_shift":
                inputs, positions, pieces, questions = prepare_single(
                    processor,
                    batch_samples,
                    templates_for(experiment, batch_samples),
                )
                hook_positions: Sequence[Any] = positions
                piece_rows = [{"obj1": piece} for piece in pieces]
            else:
                inputs, positions, piece_rows, questions = prepare_relative(
                    processor,
                    batch_samples,
                )
                hook_positions = positions
            axes = [AXIS_INDEX[sample.group] for sample in batch_samples]
            hook.configure(
                mode,
                hook_positions,
                axes,
                value,
                factors_device,
                basis_device,
            )
            with torch.inference_mode():
                output = model(
                    **move_inputs(inputs, model),
                    use_cache=False,
                    logits_to_keep=1,
                    return_dict=True,
                )
            if hook.applied_count != 1 or hook.armed:
                raise RuntimeError("Prefill causal edit did not apply exactly once")
            scored = binary_scores(
                output.logits[:, -1],
                [sample.group for sample in batch_samples],
                token_ids,
            )
            for offset, (sample, piece, question, score, axis) in enumerate(
                zip(
                    batch_samples,
                    piece_rows,
                    questions,
                    scored,
                    axes,
                    strict=True,
                )
            ):
                absolute_index = start + offset
                case = cases.iloc[absolute_index].to_dict()
                baseline_row = baseline.iloc[absolute_index]
                answer = (
                    str(case["absolute_answer"])
                    if experiment == "absolute_shift"
                    else sample.category
                )
                record = {
                    **identity_record(case),
                    **expected,
                    "axis_index": axis,
                    "question": question,
                    "query_token": piece["obj1"],
                    "reference_token": piece.get("obj2", ""),
                    "baseline_prediction": baseline_row["prediction"],
                    "baseline_correct": bool(baseline_row["correct"]),
                    **score,
                    "correct": score["prediction"] == answer,
                    "changed_vs_baseline": score["prediction"]
                    != baseline_row["prediction"],
                    **encoded_mechanics(hook, offset, axis),
                }
                if experiment == "relative_swap":
                    record["target_opposite_correct"] = (
                        score["prediction"] == OPPOSITE[sample.category]
                    )
                rows.append(record)
            start = stop
            atomic_csv(path, pd.DataFrame(rows))
            progress.update(stop - progress.n)
    progress.close()
    return pd.DataFrame(rows)


def run_color_stage(
    layer: int,
    value: float,
    processor: Any,
    model: torch.nn.Module,
    samples: Sequence[Sample],
    cases: pd.DataFrame,
    baseline: pd.DataFrame,
    factors: torch.Tensor,
    basis: torch.Tensor,
    batch_size: int,
    max_new_tokens: int,
    path: Path,
    resume: bool,
) -> pd.DataFrame:
    expected = {
        "experiment": "nonspatial_shift",
        "source_layer": layer,
        "parameter": "beta",
        "value": float(value),
        "value_key": value_key(value),
    }
    records = progress_frame(path, cases, resume, expected)
    start = len(records)
    rows = records.to_dict("records")
    progress = tqdm(
        total=len(samples),
        initial=start,
        desc=f"nonspatial_shift/L{layer}/beta={value:g}",
    )
    block_device = next(model.model.transformer.blocks[layer].parameters()).device
    factors_device, basis_device = factors.to(block_device), basis.to(block_device)
    with CoordinateEditHook(model, layer) as hook:
        while start < len(samples):
            stop = min(start + batch_size, len(samples))
            batch_samples = samples[start:stop]
            inputs, positions, pieces, questions = prepare_single(
                processor,
                batch_samples,
                templates_for("nonspatial_shift", batch_samples),
            )
            axes = [AXIS_INDEX[sample.group] for sample in batch_samples]
            hook.configure(
                "single_shift",
                positions,
                axes,
                value,
                factors_device,
                basis_device,
            )
            moved = move_inputs(inputs, model)
            input_width = int(moved["input_ids"].shape[1])
            with torch.inference_mode():
                generated = model.generate(
                    **moved,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            if hook.applied_count != 1 or hook.armed:
                raise RuntimeError(
                    "Color prefill causal edit did not apply exactly once"
                )
            scored = color_rows(processor, generated, input_width)
            first_log_probs = (
                torch.log_softmax(
                    generated.scores[0].float(),
                    dim=-1,
                )
                .detach()
                .cpu()
            )
            for offset, (piece, question, score, axis) in enumerate(
                zip(
                    pieces,
                    questions,
                    scored,
                    axes,
                    strict=True,
                )
            ):
                absolute_index = start + offset
                case = cases.iloc[absolute_index].to_dict()
                baseline_row = baseline.iloc[absolute_index]
                baseline_token_id = int(baseline_row["first_token_id"])
                baseline_logp = float(first_log_probs[offset, baseline_token_id])
                rows.append(
                    {
                        **identity_record(case),
                        **expected,
                        "axis_index": axis,
                        "question": question,
                        "query_token": piece,
                        "baseline_answer": baseline_row["normalized_answer"],
                        "baseline_raw_answer": baseline_row["raw_answer"],
                        "baseline_token_id": baseline_token_id,
                        **score,
                        "retained": score["normalized_answer"]
                        == baseline_row["normalized_answer"],
                        "changed_vs_baseline": score["normalized_answer"]
                        != baseline_row["normalized_answer"],
                        "logp_baseline_first_token": baseline_logp,
                        "p_baseline_first_token": math.exp(baseline_logp),
                        **encoded_mechanics(hook, offset, axis),
                    }
                )
            start = stop
            atomic_csv(path, pd.DataFrame(rows))
            progress.update(stop - progress.n)
    progress.close()
    return pd.DataFrame(rows)


def summarize(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        first = frame.iloc[0]
        for scope, part in (
            ("overall", frame),
            ("horizontal", frame[frame["group"].eq("horizontal")]),
            ("vertical", frame[frame["group"].eq("vertical")]),
        ):
            row: dict[str, Any] = {
                "experiment": first["experiment"],
                "source_layer": int(first["source_layer"]),
                "parameter": first["parameter"],
                "value": float(first["value"]),
                "scope": scope,
                "n": len(part),
                "changed_count": int(part["changed_vs_baseline"].sum()),
                "changed_rate": float(part["changed_vs_baseline"].mean()),
                "max_coordinate_target_error": float(
                    part["coordinate_target_error"].max()
                ),
                "max_offaxis_change": float(part["offaxis_max_change"].max()),
                "median_edit_norm_ratio": float(part["edit_norm_ratio"].median()),
            }
            if "baseline_correct" in part:
                row["baseline_accuracy"] = float(part["baseline_correct"].mean())
                row["accuracy"] = float(part["correct"].mean())
            if "target_opposite_correct" in part:
                row["target_opposite_count"] = int(
                    part["target_opposite_correct"].sum()
                )
                row["target_opposite_rate"] = float(
                    part["target_opposite_correct"].mean()
                )
            rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["experiment", "source_layer", "value", "scope"],
        )
        .reset_index(drop=True)
    )


def summarize_layers(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    records = pd.concat(frames, ignore_index=True)
    rows: list[dict[str, Any]] = []
    for (experiment, layer), layer_rows in records.groupby(
        ["experiment", "source_layer"],
        sort=False,
    ):
        if experiment == "relative_swap":
            groups = [
                (
                    [float(value)],
                    float(value),
                    value_rows.set_index("case_index"),
                    "changed_vs_baseline",
                )
                for value, value_rows in layer_rows.groupby("value", sort=True)
            ]
        else:
            groups = []
            magnitudes = layer_rows["value"].abs()
            for magnitude in sorted(float(value) for value in magnitudes.unique()):
                strength_rows = layer_rows[magnitudes.eq(magnitude)]
                values = sorted(float(value) for value in strength_rows["value"].unique())
                counts = strength_rows.groupby("case_index")["value"].nunique()
                if not counts.eq(len(values)).all():
                    raise ValueError(
                        f"Incomplete strength summary for {experiment}/L{layer}/|beta|={magnitude:g}"
                    )
                by_case = strength_rows.drop_duplicates("case_index").set_index("case_index")
                by_case = by_case.assign(
                    any_value_changed=strength_rows.groupby("case_index")[
                        "changed_vs_baseline"
                    ].any()
                )
                groups.append((values, magnitude / 5.0, by_case, "any_value_changed"))
        for values, edit_scale, by_case, metric in groups:
            for scope, part in (
                ("overall", by_case),
                ("horizontal", by_case[by_case["group"].eq("horizontal")]),
                ("vertical", by_case[by_case["group"].eq("vertical")]),
            ):
                effect_count = int(part[metric].sum())
                row = {
                    "experiment": experiment,
                    "source_layer": int(layer),
                    "edit_scale": edit_scale,
                    "scope": scope,
                    "values": json.dumps(values),
                    "effect_metric": (
                        "changed_vs_baseline"
                        if experiment == "relative_swap"
                        else "any_value_changed_vs_baseline"
                    ),
                    "n": len(part),
                    "effect_count": effect_count,
                    "effect_rate": effect_count / len(part),
                }
                if experiment != "relative_swap":
                    row["unchanged_count"] = len(part) - effect_count
                    row["unchanged_rate"] = 1.0 - effect_count / len(part)
                rows.append(row)
    return (
        pd.DataFrame(rows)
        .sort_values(
            ["experiment", "source_layer", "edit_scale", "scope"],
        )
        .reset_index(drop=True)
    )


def build_showcase(
    output_dir: Path,
    cases: pd.DataFrame,
    samples: Sequence[Sample],
    layer: int,
    alpha_values: Sequence[float],
    beta_values: Sequence[float],
    frames: Mapping[tuple[str, float], pd.DataFrame],
) -> None:
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    bundle_cases = []
    for row, sample in zip(cases.to_dict("records"), samples, strict=True):
        case_id = str(row["showcase_id"])
        with Image.open(io.BytesIO(sample.image_bytes)) as image:
            image.convert("RGB").save(image_dir / f"{case_id}.jpg", quality=95)
        experiments: dict[str, Any] = {}
        for experiment in EXPERIMENTS:
            parameter = "alpha" if experiment == "relative_swap" else "beta"
            values = alpha_values if parameter == "alpha" else beta_values
            states = []
            question = ""
            for value in values:
                part = frames[(experiment, float(value))]
                state_row = part[part["case_index"].eq(int(row["case_index"]))].iloc[0]
                question = str(state_row["question"])
                coordinates = {
                    "query": json.loads(state_row["coordinates_after_query"]),
                }
                if experiment.startswith("relative_"):
                    coordinates["reference"] = json.loads(
                        state_row["coordinates_after_reference"]
                    )
                state: dict[str, Any] = {
                    "value": float(value),
                    "answer": (
                        str(state_row["first_token"]).strip()
                        if experiment == "nonspatial_shift"
                        else str(state_row["prediction"])
                    ),
                    "coordinates": coordinates,
                }
                if experiment == "nonspatial_shift":
                    state["top5"] = json.loads(state_row["top5"])
                    state["fullAnswer"] = str(state_row["normalized_answer"])
                else:
                    state["probabilities"] = {
                        str(state_row["negative_answer"]): float(
                            state_row["p_negative_vocab"]
                        ),
                        str(state_row["positive_answer"]): float(
                            state_row["p_positive_vocab"]
                        ),
                    }
                states.append(state)
            experiments[experiment] = {
                "layer": layer,
                "question": question,
                "parameter": parameter,
                "visibleObjectIds": (
                    ["query", "reference"]
                    if experiment.startswith("relative_")
                    else ["query"]
                ),
                "states": states,
            }
        bundle_cases.append(
            {
                "id": case_id,
                "caseIndex": int(row["case_index"]),
                "category": row["category"],
                "axis": "h" if row["group"] == "horizontal" else "v",
                "image": f"images/{case_id}.jpg",
                "objects": [
                    {"id": "query", "label": row["query_object"]},
                    {"id": "reference", "label": row["reference_object"]},
                ],
                "experiments": experiments,
            }
        )
    atomic_json(
        output_dir / "showcase_sweep.json",
        {
            "schemaVersion": 1,
            "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "artifactId": ARTIFACT_ID,
            "sourceLayer": layer,
            "probability": "full-vocabulary softmax; no candidate-set renormalization",
            "grids": {"alpha": list(alpha_values), "beta": list(beta_values)},
            "cases": bundle_cases,
        },
    )


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("Batch size and max-new-tokens must be positive")
    os.environ["HF_HOME"] = str(args.hf_home)
    artifact = load_axes_artifact(args.axes)
    layers = parse_layers(args.layers, artifact.layer_ids)
    experiments = parse_experiments(args.experiments)
    alpha_values = finite_values(args.alpha_values, "alpha")
    beta_values = finite_values(args.beta_values, "beta")

    if args.selection:
        if len(layers) != 1:
            raise ValueError("Showcase selection requires exactly one layer")
        if experiments != list(EXPERIMENTS):
            raise ValueError("Showcase selection requires --experiments all")
        if args.showcase_points < 3 or args.showcase_points % 2 == 0:
            raise ValueError("showcase-points must be an odd number of at least three")
        alpha_values = np.linspace(0.0, 10.0, args.showcase_points).tolist()
        beta_values = np.linspace(-30.0, 30.0, args.showcase_points).tolist()

    output_dir = args.output_dir or (
        PROJECT_DIR
        / "outputs"
        / "spatial_causality"
        / ("showcase" if args.selection else "molmo2_er")
    )
    if not args.resume and output_dir.is_dir() and any(output_dir.iterdir()):
        raise ValueError(
            f"--no-resume requires an empty or new output directory: {output_dir}"
        )
    samples = select_samples(args.database, args.samples_per_category, args.seed)
    cases = base_cases(samples)
    if "absolute_shift" in experiments:
        annotations = args.annotations or default_annotations(args.hf_home)
        cases = add_absolute_labels(cases, samples, annotations)
    else:
        annotations = args.annotations
    if args.selection:
        samples, cases = apply_selection(samples, cases, args.selection)

    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path = output_dir / "cases.csv"
    model_path = args.model_path or resolve_model_path(
        args.hf_home, allow_download=False
    )
    run_meta = {
        "runtime_identity": runtime_identity("molmo2_er", PROJECT_DIR),
        "database_sha256": file_sha256(args.database),
        "annotations_sha256": file_sha256(annotations) if annotations is not None else None,
        "axes_sha256": file_sha256(artifact.path),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_path": str(model_path.resolve()),
        "artifact_id": ARTIFACT_ID,
        "axes": str(artifact.path),
        "available_layers": list(artifact.layer_ids),
        "layers": layers,
        "experiments": experiments,
        "alpha_values": alpha_values,
        "beta_values": beta_values,
        "sample_count": len(samples),
        "samples_per_category": args.samples_per_category,
        "seed": args.seed,
        "database": str(args.database.resolve()),
        "annotations": str(annotations.resolve()) if annotations else None,
        "selection": str(args.selection.resolve()) if args.selection else None,
        "causal_edit_scope": "post_block_prefill_once",
        "batch_size": args.batch_size,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "prompts": {
            "relative": {
                group: TEMPLATES["baseline"][group]
                for group in ("horizontal", "vertical")
            },
            "absolute": {
                group: config["template"] for group, config in ABSOLUTE_CONFIG.items()
            },
            "color": COLOR_TEMPLATE,
        },
    }
    meta_path = output_dir / "run_meta.json"
    if args.resume:
        existing = any(output_dir.iterdir())
        if existing and (not meta_path.is_file() or not cases_path.is_file()):
            raise ValueError(f"Incomplete saved run metadata in {output_dir}")
        if meta_path.is_file():
            saved_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if saved_meta != run_meta:
                raise ValueError(f"Saved run configuration differs: {meta_path}")
            validate_cases(cases_path, cases)
    atomic_csv(cases_path, cases)
    atomic_json(meta_path, run_meta)

    processor, model = load_model(model_path, args.device, require_gradients=False)
    processor.tokenizer.padding_side = "left"
    hidden_size = int(model.config.get_text_config().hidden_size)
    token_ids = answer_token_ids(processor.tokenizer)
    baselines: dict[str, pd.DataFrame] = {}
    stage_frames: list[pd.DataFrame] = []
    showcase_frames: dict[tuple[str, float], pd.DataFrame] = {}
    try:
        if any(name.startswith("relative_") for name in experiments):
            baselines["relative"] = run_binary_baseline(
                "relative",
                processor,
                model,
                samples,
                cases,
                token_ids,
                args.batch_size,
                baseline_path(output_dir, "relative"),
                args.resume,
            )
        if "absolute_shift" in experiments:
            baselines["absolute"] = run_binary_baseline(
                "absolute",
                processor,
                model,
                samples,
                cases,
                token_ids,
                args.batch_size,
                baseline_path(output_dir, "absolute"),
                args.resume,
            )
        if "nonspatial_shift" in experiments:
            baselines["color"] = run_color_baseline(
                processor,
                model,
                samples,
                cases,
                args.batch_size,
                args.max_new_tokens,
                baseline_path(output_dir, "color"),
                args.resume,
            )

        for layer in layers:
            factors = artifact.factors(layer)
            if factors.shape != (3, hidden_size):
                raise ValueError(f"L{layer} axes do not match model hidden size")
            basis = coordinate_right_inverse(factors)
            for experiment in experiments:
                parameter = "alpha" if experiment == "relative_swap" else "beta"
                values = alpha_values if parameter == "alpha" else beta_values
                for value in values:
                    path = stage_path(
                        output_dir,
                        experiment,
                        layer,
                        parameter,
                        float(value),
                    )
                    if experiment == "nonspatial_shift":
                        frame = run_color_stage(
                            layer,
                            float(value),
                            processor,
                            model,
                            samples,
                            cases,
                            baselines["color"],
                            factors,
                            basis,
                            args.batch_size,
                            args.max_new_tokens,
                            path,
                            args.resume,
                        )
                    else:
                        baseline_key = (
                            "absolute" if experiment == "absolute_shift" else "relative"
                        )
                        frame = run_binary_stage(
                            experiment,
                            layer,
                            parameter,
                            float(value),
                            processor,
                            model,
                            samples,
                            cases,
                            baselines[baseline_key],
                            token_ids,
                            factors,
                            basis,
                            args.batch_size,
                            path,
                            args.resume,
                        )
                    if len(frame) != len(samples):
                        raise ValueError(
                            f"Incomplete stage {experiment}/L{layer}/{value:g}"
                        )
                    stage_frames.append(frame)
                    if args.selection:
                        showcase_frames[(experiment, float(value))] = frame
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = summarize(stage_frames)
    atomic_csv(output_dir / "summary_all.csv", summary)
    atomic_csv(output_dir / "layer_summary.csv", summarize_layers(stage_frames))
    if args.selection:
        build_showcase(
            output_dir,
            cases,
            samples,
            layers[0],
            alpha_values,
            beta_values,
            showcase_frames,
        )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
