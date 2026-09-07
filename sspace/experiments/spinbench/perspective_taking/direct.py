"""Run checkpointed SpinBench direct inference."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from PIL import Image

from sspace.run_records import RunRecorder, atomic_json, canonical_fingerprint
from sspace.core.extraction.execution import move_model_inputs
from sspace.core.models.runtime import LoadedModelRuntime
from sspace.core.models.qwen import prepare_direct_generation

from .adapter import SpinBenchSample
from sspace.experiments.common.direct.generation import split_generated_response
from .direct_config import SpinBenchDirectConfig


@dataclass(frozen=True)
class SpinBenchDirectRecord:
    """One complete SpinBench generation and strict A/B decision."""

    evaluation_id: str
    sample_id: str
    global_sample_index: int
    row_index: int
    premise_mode: str
    transform: str
    source_group: str
    target_view: str
    target_property: str
    object_a: str
    object_b: str
    gold_answer: str
    prompt: str
    source_premise: str | None
    image_path: str
    image_sha256: str
    thinking: bool
    output_tokens: int
    ended_with_eos: bool
    thinking_completed: bool | None
    reasoning_response: str
    final_response: str
    raw_response: str
    generated_token_sha256: str
    predicted_answer: str | None
    parse_failure: bool
    correct: bool
    generation_seconds: float
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe record without altering generated text."""
        return asdict(self)


def parse_spinbench_choice(text: str) -> str | None:
    """Parse one unique standalone uppercase A/B from final-answer text.

    Repeated mentions of the same option are unambiguous. Lowercase answers,
    both options, or no option are rejected by the frozen EVAL-06 protocol.
    """
    matches = set(re.findall(r"(?<![A-Za-z])([AB])(?![A-Za-z])", text))
    return next(iter(matches)) if len(matches) == 1 else None


def _distribution(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot summarize an empty numeric distribution")

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "sum": float(sum(ordered)),
        "mean": float(sum(ordered) / len(ordered)),
        "median": percentile(0.5),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def summarize_spinbench_direct_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize complete SpinBench records by frozen semantic groups."""
    if not records:
        raise ValueError("Cannot summarize empty SpinBench records")
    identity: tuple[str, str, bool] | None = None
    seen: set[int] = set()
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    parse_failures = thinking_incomplete = eos_count = 0
    tokens: list[float] = []
    seconds: list[float] = []
    for record in records:
        current = (
            str(record["evaluation_id"]),
            str(record["premise_mode"]),
            bool(record["thinking"]),
        )
        if identity is None:
            identity = current
        elif identity != current:
            raise ValueError("SpinBench summary mixes run identities")
        index = int(record["global_sample_index"])
        if index in seen:
            raise ValueError("SpinBench summary contains duplicate sample indices")
        seen.add(index)
        correct = int(bool(record["correct"]))
        for key in (
            "all",
            f"transform:{record['transform']}",
            f"source_group:{record['source_group']}",
            f"target_view:{record['target_view']}",
            f"target_property:{record['target_property']}",
        ):
            buckets[key][0] += correct
            buckets[key][1] += 1
        parse_failures += int(bool(record["parse_failure"]))
        if current[2]:
            thinking_incomplete += int(record["thinking_completed"] is not True)
        elif record["thinking_completed"] is not None:
            raise ValueError("No-Thinking records must not report Thinking completion")
        eos_count += int(bool(record["ended_with_eos"]))
        tokens.append(float(record["output_tokens"]))
        seconds.append(float(record["generation_seconds"]))

    def score(key: str) -> dict[str, float | int]:
        correct, total = buckets[key]
        return {"correct": correct, "total": total, "accuracy": correct / total}

    def grouped(prefix: str) -> dict[str, dict[str, float | int]]:
        return {
            key.removeprefix(prefix): score(key)
            for key in sorted(buckets)
            if key.startswith(prefix)
        }

    assert identity is not None
    return {
        "evaluation_id": identity[0],
        "premise_mode": identity[1],
        "thinking": identity[2],
        "overall": score("all"),
        "by_transform": grouped("transform:"),
        "by_source_group": grouped("source_group:"),
        "by_target_view": grouped("target_view:"),
        "by_target_property": grouped("target_property:"),
        "parse_failures": parse_failures,
        "thinking_incomplete": thinking_incomplete if identity[2] else None,
        "ended_with_eos": eos_count,
        "output_tokens": _distribution(tokens),
        "generation_seconds": _distribution(seconds),
    }


def _load_image(sample: SpinBenchSample) -> tuple[Image.Image, str]:
    encoded = sample.image_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    image = Image.open(sample.image_path)
    image.load()
    return image.convert("RGB"), digest


def _eos_ids(model: torch.nn.Module) -> set[int]:
    value = model.generation_config.eos_token_id
    if isinstance(value, int):
        return {value}
    if isinstance(value, Sequence) and value:
        return {int(item) for item in value}
    raise ValueError("SpinBench model generation config must declare EOS token IDs")


def run_spinbench_direct(
    runtime: LoadedModelRuntime,
    samples: Sequence[SpinBenchSample],
    dataset_fingerprint: str,
    config: SpinBenchDirectConfig,
    recorder: RunRecorder,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Generate, strictly parse, and checkpoint one full SpinBench condition.

    Implements frozen greedy generation. Qwen Thinking permits 32K new tokens;
    no-Thinking Qwen and Molmo permit 128. This path does not read axes or
    hidden states.
    """
    if runtime.spec.key != config.model_adapter:
        raise ValueError("SpinBench direct runner requires its configured runtime")
    expected_count = sample_limit or config.expected_sample_count
    if len(samples) != expected_count or any(
        sample.premise_mode != config.premise_mode for sample in samples
    ):
        raise ValueError("SpinBench samples differ from the configured subset")
    records_path = recorder.directory / "records.jsonl"
    checkpoint_path = recorder.directory / "checkpoint.json"
    run_identity = {
        **config.to_dict(),
        "dataset_fingerprint": dataset_fingerprint,
        "sample_limit": sample_limit,
    }
    fingerprint = canonical_fingerprint(run_identity)
    if checkpoint_path.exists():
        checkpoint_value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_value["fingerprint"] != fingerprint:
            raise ValueError("SpinBench checkpoint fingerprint mismatch")
        next_index = int(checkpoint_value["next_sample_index"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint_value["records_offset"]))
    else:
        if records_path.exists():
            raise FileExistsError("SpinBench records exist without a checkpoint")
        records_path.touch(exist_ok=False)
        next_index = 0
    eos_ids = _eos_ids(runtime.model)

    def checkpoint(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "next_sample_index": next_index,
                "records_offset": stream.tell(),
            },
        )

    with records_path.open("a", encoding="utf-8") as stream:
        checkpoint(stream)
        while next_index < len(samples):
            sample = samples[next_index]
            image, image_sha256 = _load_image(sample)
            if config.thinking:
                inputs = prepare_direct_generation(
                    runtime.processor,
                    image,
                    sample.direct_prompt,
                    thinking=True,
                )
            else:
                inputs = runtime.prepare_generation(
                    runtime.processor,
                    image,
                    sample.direct_prompt,
                )
            model_inputs = move_model_inputs(inputs, runtime.model)
            input_length = int(model_inputs["input_ids"].shape[1])
            try:
                if runtime.model.device.type == "cuda":
                    torch.cuda.synchronize(runtime.model.device)
                started = time.perf_counter()
                with torch.inference_mode():
                    output = runtime.model.generate(
                        **model_inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=config.max_new_tokens,
                    )
                if runtime.model.device.type == "cuda":
                    torch.cuda.synchronize(runtime.model.device)
                elapsed = time.perf_counter() - started
            except BaseException:
                checkpoint(stream)
                raise
            if (
                output.ndim != 2
                or output.shape[0] != 1
                or output.shape[1] <= input_length
            ):
                raise ValueError("SpinBench generation output has an invalid shape")
            generated = [int(value) for value in output[0, input_length:].tolist()]
            raw, reasoning, final, thinking_completed = split_generated_response(
                generated,
                runtime.processor.tokenizer,
                thinking=config.thinking,
            )
            prediction = parse_spinbench_choice(final)
            token_digest = hashlib.sha256(
                json.dumps(generated, separators=(",", ":")).encode()
            ).hexdigest()
            record = SpinBenchDirectRecord(
                evaluation_id=config.evaluation_id,
                sample_id=sample.sample_id,
                global_sample_index=next_index,
                row_index=sample.row_index,
                premise_mode=sample.premise_mode,
                transform=sample.transform,
                source_group=sample.source_group,
                target_view=sample.target_view,
                target_property=sample.target_property,
                object_a=sample.object_a,
                object_b=sample.object_b,
                gold_answer=sample.answer,
                prompt=sample.direct_prompt,
                source_premise=sample.source_premise,
                image_path=str(sample.image_path),
                image_sha256=image_sha256,
                thinking=config.thinking,
                output_tokens=len(generated),
                ended_with_eos=generated[-1] in eos_ids,
                thinking_completed=thinking_completed,
                reasoning_response=reasoning,
                final_response=final,
                raw_response=raw,
                generated_token_sha256=token_digest,
                predicted_answer=prediction,
                parse_failure=prediction is None,
                correct=prediction == sample.answer,
                generation_seconds=float(elapsed),
                metadata=dict(sample.metadata),
            )
            stream.write(
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            next_index += 1
            if next_index % config.checkpoint_every == 0:
                checkpoint(stream)
                print(
                    f"spinbench {config.premise_mode} {next_index}/{len(samples)}",
                    flush=True,
                )
        checkpoint(stream)
    with records_path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    summary = summarize_spinbench_direct_records(records)
    summary.update(
        dataset_fingerprint=dataset_fingerprint,
        sample_count=len(samples),
        decoding_protocol=config.decoding_protocol,
        max_new_tokens=config.max_new_tokens,
        thinking_budget="unset",
    )
    atomic_json(recorder.directory / "summary.json", summary)
    return summary
