#!/usr/bin/env python3
"""Generate Qwen3.6 MMSI reasoning and probe naturally occurring direction tokens.

The script deliberately keeps behaviour and mechanism in the same model:

1. Qwen3.6-27B greedily generates the fixed five-step rationale. Native
   thinking is opt-in; the exhaustive reference run keeps it disabled.
2. The exact generated token sequence is replayed with the same image(s).
3. States at naturally generated direction words are projected at the
   artifact-selected post-block layer onto the released Qwen3.6 COCO
   Final-logit H/V/D axes.

The input is the exhaustive eight-geographic-direction MMSI manifest produced
by ``prepare_mmsi_eightway.py``. Rows are deduplicated by ``mmsi_id``. The
reference run keeps all 191 rows and does not filter by model correctness,
question type, or generated rationale.
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import time
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256
from sspace.core.artifacts import ModelSpec, SSpaceArtifact
from sspace.core.models.runtime import LoadedModelRuntime, load_model_runtime

from ..token_matching import find_direction_occurrences
from sspace.experiments.identity import runtime_identity

AXIS_ORDER = ("horizontal", "vertical", "distance")
ANSWER_RE = re.compile(
    r"\b(?:answer|option)\s*(?:is\b|:|：)\s*\(?([A-D])(?!\w)\)?",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()


def canonical_samples(
    rows: Iterable[dict[str, Any]], question_type: str | None
) -> list[dict[str, Any]]:
    samples: dict[int, dict[str, Any]] = {}
    for row in rows:
        if question_type is not None and row.get("question_type") != question_type:
            continue
        sample_id = int(row["mmsi_id"])
        if sample_id in samples:
            continue
        prompt = row["prompt"].split("\n\nReasoning:", 1)[0].strip()
        samples[sample_id] = {
            "mmsi_id": sample_id,
            "question_type": row["question_type"],
            "difficulty": row.get("difficulty"),
            "answer_letter": str(row["answer_letter"]).upper(),
            "answer_text": row.get("answer_text"),
            "gold_direction": row.get("gold_direction"),
            "image_paths": list(row["image_paths"]),
            "image_sha256": list(row["image_sha256"]),
            "prompt": prompt,
        }
    return [samples[key] for key in sorted(samples)]


def runtime_model_spec(
    runtime: LoadedModelRuntime, artifact: SSpaceArtifact
) -> ModelSpec:
    """Convert the frozen runtime declaration to the artifact identity contract."""

    spec = runtime.spec
    return ModelSpec(
        model_id=spec.model_id,
        revision=spec.revision,
        architecture=spec.architecture,
        tokenizer_id=spec.model_id,
        tokenizer_revision=spec.revision,
        hidden_size=spec.hidden_size,
        layer_ids=tuple(range(spec.layer_count)),
        selected_layer=artifact.manifest.model.selected_layer,
        layer_numbering="zero_based_post_block",
    )


def resolve_images(manifest: Path, sample: dict[str, Any]) -> list[Path]:
    """Resolve and checksum every manifest-relative MMSI image."""

    paths = sample["image_paths"]
    hashes = sample["image_sha256"]
    if len(paths) != len(hashes) or not paths:
        raise ValueError("MMSI rows require aligned image paths and SHA-256 values")
    root = manifest.parent.resolve()
    resolved = []
    for relative, expected in zip(paths, hashes, strict=True):
        path = Path(str(relative))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("MMSI image paths must stay below the manifest directory")
        candidate = (root / path).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise FileNotFoundError(candidate)
        if file_sha256(candidate) != str(expected):
            raise ValueError(f"MMSI image checksum mismatch: {relative}")
        resolved.append(candidate)
    return resolved


def prepare_inputs(
    processor: Any,
    images: list[Image.Image],
    prompt: str,
    *,
    native_thinking: bool = False,
) -> Any:
    content = [{"type": "image", "image": image} for image in images]
    content.append(
        {
            "type": "text",
            "text": (
                prompt
                + "\n\nGive a concise step-by-step spatial explanation in at most "
                "five steps. End exactly with `Answer: X`, where X is the option letter."
            ),
        }
    )
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=native_thinking,
        return_tensors="pt",
        return_dict=True,
    )


def move_inputs(inputs: Any, device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def parse_answer(text: str) -> str | None:
    matches = ANSWER_RE.findall(text)
    if matches:
        return matches[-1].upper()
    # Accept a bare option letter only when it is the complete final line.
    # Searching every standalone A-D near the tail can misclassify a response
    # truncated mid-reasoning (for example, the article "A" in prose).
    trailing = re.search(
        r"(?:^|\n)\s*\(?([A-D])\)?[.!]?\s*(?:<\|im_end\|>)?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    return trailing.group(1).upper() if trailing else None


def extend_for_replay(
    inputs: dict[str, torch.Tensor], full_ids: torch.Tensor
) -> dict[str, torch.Tensor]:
    replay = dict(inputs)
    original_length = int(inputs["input_ids"].shape[1])
    replay["input_ids"] = full_ids
    replay["attention_mask"] = torch.ones_like(
        full_ids, dtype=inputs["attention_mask"].dtype
    )
    # Qwen3.6 names this sequence ``mm_token_type_ids``.  Keep this generic so
    # future processor versions with another text-aligned field remain valid.
    # Generated tokens inherit the final prompt token's (text) type.
    for name, old in list(inputs.items()):
        if name in {"input_ids", "attention_mask"}:
            continue
        if old.ndim != 2 or tuple(old.shape) != tuple(inputs["input_ids"].shape):
            continue
        extension = old[:, -1:].expand(
            old.shape[0], full_ids.shape[1] - original_length
        )
        replay[name] = torch.cat([old, extension], dim=1)
    return replay


def extract_projections(
    model: torch.nn.Module,
    block: torch.nn.Module,
    replay_inputs: dict[str, torch.Tensor],
    input_length: int,
    occurrences: list[dict[str, Any]],
    axes: torch.Tensor,
) -> list[dict[str, Any]]:
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["hidden"] = output[0] if isinstance(output, tuple) else output

    handle = block.register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(
                **replay_inputs,
                use_cache=False,
                return_dict=True,
                logits_to_keep=1,
            )
    finally:
        handle.remove()
    hidden = captured.pop("hidden")[0]
    output: list[dict[str, Any]] = []
    for occurrence in occurrences:
        positions = [
            input_length + position
            for position in range(
                int(occurrence["generated_start"]), int(occurrence["generated_end"])
            )
        ]
        state = hidden[positions].float().mean(dim=0)
        norm = torch.linalg.vector_norm(state)
        scores = axes @ state
        record = dict(occurrence)
        record["sequence_positions"] = positions
        record["hidden_norm"] = float(norm.cpu())
        for index, name in enumerate(AXIS_ORDER):
            record[name] = float(scores[index].cpu())
            record[name + "_cosine"] = float((scores[index] / norm).cpu())
        output.append(record)
    del hidden
    return output


def write_outputs(output_dir: Path, records: list[dict[str, Any]]) -> None:
    flat: list[dict[str, Any]] = []
    for record in records:
        for occurrence_index, occurrence in enumerate(record.get("occurrences", [])):
            flat.append(
                {
                    "mmsi_id": record["mmsi_id"],
                    "question_type": record.get("question_type"),
                    "gold_direction": record.get("gold_direction"),
                    "gold": record["answer_letter"],
                    "prediction": record.get("prediction"),
                    "correct": record.get("correct"),
                    "occurrence_index": occurrence_index,
                    **occurrence,
                }
            )
    pd.DataFrame(flat).to_csv(
        output_dir / "direction_token_projections.csv", index=False
    )
    wrong = [record for record in records if record.get("correct") is False]
    (output_dir / "wrong_cases.json").write_text(
        json.dumps(wrong, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    parsed = [record for record in records if record.get("prediction") is not None]
    summary = {
        "completed": len(records),
        "parsed": len(parsed),
        "correct": sum(record.get("correct") is True for record in parsed),
        "accuracy": (
            sum(record.get("correct") is True for record in parsed) / len(records)
            if records
            else None
        ),
        "parse_failures": len(records) - len(parsed),
        "parsed_accuracy": (
            sum(record.get("correct") is True for record in parsed) / len(parsed)
            if parsed
            else None
        ),
        "wrong": len(wrong),
        "probe_occurrences": len(flat),
        "word_counts": Counter(item["word"] for item in flat),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--layer",
        type=int,
        help="Post-block layer; defaults to the published artifact selection.",
    )
    parser.add_argument(
        "--question-type",
        help=(
            "Optional diagnostic filter. Omit it for the canonical exhaustive "
            "191-question run."
        ),
    )
    parser.add_argument(
        "--mmsi-ids",
        type=int,
        nargs="*",
        help="Optionally keep only these MMSI ids after canonical deduplication.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument(
        "--native-thinking",
        action="store_true",
        help=(
            "Enable the model's native thinking stream while retaining the same "
            "five-step user instruction. The canonical 191-question run leaves "
            "this disabled."
        ),
    )
    parser.add_argument(
        "--skip-projections",
        action="store_true",
        help="Save generation text and answers without replaying the sequence for axis projections.",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        raise ValueError("max-new-tokens must be positive")
    if args.offset < 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("offset must be nonnegative and limit must be positive")

    samples = canonical_samples(read_jsonl(args.manifest), args.question_type)
    if args.mmsi_ids:
        requested_ids = set(args.mmsi_ids)
        samples = [
            sample for sample in samples if int(sample["mmsi_id"]) in requested_ids
        ]
        found_ids = {int(sample["mmsi_id"]) for sample in samples}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(f"Requested MMSI ids not found: {sorted(missing_ids)}")
    if args.offset:
        samples = samples[args.offset :]
    if args.limit is not None:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("No canonical MMSI samples selected")
    artifact = SSpaceArtifact.load(args.artifact_dir)
    layer = artifact.manifest.model.selected_layer if args.layer is None else args.layer
    if layer != artifact.manifest.model.selected_layer:
        raise ValueError(
            f"Requested L{layer}, but the released artifact selects "
            f"L{artifact.manifest.model.selected_layer}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    records = read_jsonl(records_path) if records_path.exists() else []
    completed = {int(record["mmsi_id"]) for record in records}
    selected_ids = {int(sample["mmsi_id"]) for sample in samples}
    if not completed <= selected_ids:
        raise ValueError("Existing records contain MMSI IDs outside this selection")
    config = {
        "runtime_identity": runtime_identity("qwen36_27b", Path(__file__).resolve().parents[5]),
        "manifest": str(args.manifest),
        "manifest_sha256": file_sha256(args.manifest),
        "model_path": str(args.model_path),
        "artifact_dir": str(args.artifact_dir),
        "artifact_id": artifact.manifest.artifact_id,
        "artifact_checksums_sha256": file_sha256(args.artifact_dir / "checksums.json"),
        "layer": layer,
        "question_type": args.question_type,
        "mmsi_ids": args.mmsi_ids,
        "max_new_tokens": args.max_new_tokens,
        "thinking_enabled": args.native_thinking,
        "skip_projections": args.skip_projections,
        "sample_count": len(samples),
        "protocol": (
            "qwen_native_thinking_then_exact_generated_token_replay_v1"
            if args.native_thinking
            else "qwen_visible_cot_then_exact_generated_token_replay_v2"
        ),
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if canonical_fingerprint(existing) != canonical_fingerprint(config):
            raise ValueError("Existing MMSI extraction configuration differs")
    else:
        atomic_json(config_path, config)

    runtime = load_model_runtime("qwen36_27b", args.model_path, args.device)
    artifact.validate_model(runtime_model_spec(runtime, artifact))
    processor = runtime.processor
    model = runtime.model
    device = model.device
    blocks = runtime.blocks
    axes = (
        None
        if args.skip_projections
        else torch.from_numpy(artifact.axes(layer)).to(
            device=device, dtype=torch.float32
        )
    )

    for sample_index, sample in enumerate(samples):
        if int(sample["mmsi_id"]) in completed:
            continue
        opened: list[Image.Image] = []
        try:
            for path in resolve_images(args.manifest, sample):
                image = Image.open(path)
                opened.append(image.convert("RGB"))
            processor_inputs = prepare_inputs(
                processor,
                opened,
                sample["prompt"],
                native_thinking=args.native_thinking,
            )
            inputs = move_inputs(processor_inputs, device)
            input_length = int(inputs["input_ids"].shape[1])
            started = time.perf_counter()
            with torch.inference_mode():
                sequence = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                )
            generated = [int(value) for value in sequence[0, input_length:].tolist()]
            raw_response = processor.tokenizer.decode(
                generated, skip_special_tokens=False
            )
            clean_response = processor.tokenizer.decode(
                generated, skip_special_tokens=True
            )
            prediction = parse_answer(clean_response)
            occurrences = find_direction_occurrences(processor.tokenizer, generated)
            replay = None
            if args.skip_projections:
                projections = []
            else:
                replay = extend_for_replay(inputs, sequence)
                projections = extract_projections(
                    model,
                    blocks[layer],
                    replay,
                    input_length,
                    occurrences,
                    axes,
                )
            record = {
                **sample,
                "sample_index": sample_index,
                "prediction": prediction,
                "correct": prediction == sample["answer_letter"]
                if prediction
                else None,
                "input_tokens": input_length,
                "output_tokens": len(generated),
                "elapsed_seconds": time.perf_counter() - started,
                "raw_response": raw_response,
                "clean_response": clean_response,
                "generated_token_ids": generated,
                "occurrences": projections,
            }
            append_jsonl(records_path, record)
            records.append(record)
            completed.add(int(sample["mmsi_id"]))
            write_outputs(args.output_dir, records)
            print(
                f"[{len(records)}/{len(samples)}] MMSI {sample['mmsi_id']}: "
                f"gold={sample['answer_letter']} pred={prediction} "
                f"correct={record['correct']} probes={len(projections)}",
                flush=True,
            )
            del processor_inputs, inputs, sequence, replay
            torch.cuda.empty_cache()
        finally:
            for image in opened:
                image.close()

    write_outputs(args.output_dir, records)
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
