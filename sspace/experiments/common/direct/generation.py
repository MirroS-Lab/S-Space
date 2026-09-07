"""Run checkpointed matched-pair direct generation (EVAL-07)."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from sspace.run_records import RunRecorder, atomic_json, canonical_fingerprint
from sspace.core.extraction.execution import move_model_inputs
from sspace.core.models.runtime import LoadedModelRuntime
from sspace.core.models.qwen import prepare_direct_generation

from ..pairwise.schema import ENDPOINTS
from .config import DirectGenerationConfig
from .schema import DirectGenerationRecord, DirectGenerationSample


def direct_shard_indices(
    sample_count: int, rank: int, world_size: int
) -> tuple[int, ...]:
    """Return global sample indices for the frozen modulo sharding rule."""
    if sample_count <= 0 or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("Invalid direct-generation sample count, rank, or world size")
    return tuple(index for index in range(sample_count) if index % world_size == rank)


def parse_endpoint_response(text: str, group: str) -> str | None:
    """Parse exactly one legal endpoint as a complete case-insensitive word.

    Repeated mentions of the same endpoint remain unambiguous. A response that
    contains both endpoints, neither endpoint, or only a substring is rejected.
    """
    if group not in ENDPOINTS:
        raise ValueError(f"Unknown direct-generation group {group!r}")
    matches = {
        endpoint
        for endpoint in ENDPOINTS[group]
        if re.search(
            rf"(?<![A-Za-z]){re.escape(endpoint)}(?![A-Za-z])",
            text,
            flags=re.IGNORECASE,
        )
    }
    return next(iter(matches)) if len(matches) == 1 else None


def _thinking_end_token_id(tokenizer: Any) -> int:
    ids = tokenizer("</think>", add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"Qwen </think> must be one exact token, found {ids}")
    token_id = int(ids[0])
    if tokenizer.decode([token_id], skip_special_tokens=False) != "</think>":
        raise ValueError("Qwen </think> token does not decode exactly")
    return token_id


def split_generated_response(
    generated_ids: Sequence[int], tokenizer: Any, *, thinking: bool
) -> tuple[str, str, str, bool | None]:
    """Decode raw/reasoning/final text without scoring hidden reasoning text."""
    ids = [int(value) for value in generated_ids]
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    if not thinking:
        final = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return raw, "", final, None
    end_token = _thinking_end_token_id(tokenizer)
    positions = [index for index, value in enumerate(ids) if value == end_token]
    if not positions:
        reasoning = tokenizer.decode(ids, skip_special_tokens=True).strip()
        return raw, reasoning, "", False
    split = positions[-1]
    reasoning = tokenizer.decode(ids[:split], skip_special_tokens=True).strip()
    final = tokenizer.decode(ids[split + 1 :], skip_special_tokens=True).strip()
    return raw, reasoning, final, True


def _image(sample: DirectGenerationSample) -> tuple[Image.Image, str]:
    if sample.image_bytes is not None:
        encoded = sample.image_bytes
    elif sample.image_path is not None:
        encoded = sample.image_path.read_bytes()
    else:
        raise ValueError(f"Sample {sample.sample_id} has no image source")
    digest = hashlib.sha256(encoded).hexdigest()
    if sample.image_sha256 is not None and digest != sample.image_sha256:
        raise ValueError(f"Image checksum changed for sample {sample.sample_id}")
    image = Image.open(io.BytesIO(encoded))
    image.load()
    return image.convert("RGB"), digest


def _eos_ids(model: torch.nn.Module) -> set[int]:
    value = model.generation_config.eos_token_id
    if isinstance(value, int):
        return {value}
    if isinstance(value, Sequence) and value:
        return {int(item) for item in value}
    raise ValueError("Qwen generation config must declare EOS token IDs")


def _record_from_mapping(value: Mapping[str, Any]) -> DirectGenerationRecord:
    data = dict(value)
    data["subset_tags"] = tuple(data["subset_tags"])
    return DirectGenerationRecord(**data)


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
        "max": float(ordered[-1]),
    }


def summarize_direct_generation_records(
    records: Iterable[DirectGenerationRecord],
) -> dict[str, Any]:
    """Aggregate accuracy, parsing, completion, token, and latency evidence."""
    identity: tuple[str, str, str, bool] | None = None
    seen: set[int] = set()
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    tokens: list[float] = []
    seconds: list[float] = []
    parse_failures = eos_count = thinking_incomplete = 0
    count = 0
    for record in records:
        current = (
            record.evaluation_id,
            record.benchmark,
            record.split,
            record.thinking,
        )
        if identity is None:
            identity = current
        elif current != identity:
            raise ValueError("Direct-generation summary mixes run identities")
        if record.global_sample_index in seen:
            raise ValueError("Direct-generation summary contains duplicate indices")
        seen.add(record.global_sample_index)
        count += 1
        correct = int(record.correct)
        for key in (
            "all",
            f"axis:{record.group}",
            f"endpoint:{record.gold_endpoint}",
            *(f"subset:{tag}" for tag in record.subset_tags),
        ):
            buckets[key][0] += correct
            buckets[key][1] += 1
        parse_failures += int(record.parse_failure)
        eos_count += int(record.ended_with_eos)
        thinking_incomplete += int(record.thinking_completed is False)
        tokens.append(float(record.output_tokens))
        seconds.append(float(record.generation_seconds))
    if identity is None or count == 0:
        raise ValueError("Cannot summarize empty direct-generation records")

    def score(key: str) -> dict[str, float | int]:
        correct, total = buckets[key]
        return {"correct": correct, "total": total, "accuracy": correct / total}

    evaluation_id, benchmark, split, thinking = identity
    return {
        "evaluation_id": evaluation_id,
        "benchmark": benchmark,
        "split": split,
        "thinking": thinking,
        "overall": score("all"),
        "by_axis": {
            group: score(f"axis:{group}")
            for group in ENDPOINTS
            if f"axis:{group}" in buckets
        },
        "by_endpoint": {
            endpoint: score(f"endpoint:{endpoint}")
            for endpoints in ENDPOINTS.values()
            for endpoint in endpoints
            if f"endpoint:{endpoint}" in buckets
        },
        "by_subset": {
            key.removeprefix("subset:"): score(key)
            for key in sorted(buckets)
            if key.startswith("subset:")
        },
        "parse_failures": parse_failures,
        "ended_with_eos": eos_count,
        "thinking_incomplete": thinking_incomplete if thinking else None,
        "output_tokens": _distribution(tokens),
        "generation_seconds": _distribution(seconds),
    }


def run_direct_generation_shard(
    runtime: LoadedModelRuntime,
    samples: Sequence[DirectGenerationSample],
    dataset_fingerprint: str,
    config: DirectGenerationConfig,
    rank: int,
    recorder: RunRecorder,
    sample_limit: int | None = None,
    effective_shard_count: int | None = None,
) -> dict[str, Any]:
    """Generate one deterministic modulo shard and persist complete evidence.

    The implemented generation is ``generate(do_sample=False, num_beams=1)``
    with the config's frozen model-specific token limit. No thinking budget,
    sampling control, time limit, or fallback is supplied.
    Runtime failures checkpoint completed rows and are then re-raised.
    """
    if runtime.spec.key != config.model_adapter:
        raise ValueError("Direct-generation runtime differs from the config")
    shard_count = effective_shard_count or min(config.shard_count, len(samples))
    indices = direct_shard_indices(len(samples), rank, shard_count)
    for sample in samples:
        sample.validate()
    output_dir = recorder.directory
    records_path = output_dir / "records.jsonl"
    checkpoint_path = output_dir / "checkpoint.json"
    run_identity = {
        **config.to_dict(),
        "rank": rank,
        "effective_shard_count": shard_count,
        "dataset_fingerprint": dataset_fingerprint,
        "sample_limit": sample_limit,
    }
    fingerprint = canonical_fingerprint(run_identity)
    if checkpoint_path.exists():
        checkpoint_value = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint_value["fingerprint"] != fingerprint:
            raise ValueError("Direct-generation checkpoint fingerprint mismatch")
        next_position = int(checkpoint_value["next_local_position"])
        with records_path.open("r+b") as stream:
            stream.truncate(int(checkpoint_value["records_offset"]))
    else:
        if records_path.exists():
            raise FileExistsError("Direct records exist without a checkpoint")
        records_path.touch(exist_ok=False)
        next_position = 0
    eos_ids = _eos_ids(runtime.model)

    def checkpoint(stream: Any) -> None:
        stream.flush()
        os.fsync(stream.fileno())
        atomic_json(
            checkpoint_path,
            {
                "fingerprint": fingerprint,
                "rank": rank,
                "shard_count": shard_count,
                "next_local_position": next_position,
                "records_offset": stream.tell(),
            },
        )

    with records_path.open("a", encoding="utf-8") as stream:
        checkpoint(stream)
        while next_position < len(indices):
            global_index = indices[next_position]
            sample = samples[global_index]
            image, image_sha256 = _image(sample)
            inputs = (
                prepare_direct_generation(
                    runtime.processor,
                    image,
                    sample.prompt,
                    thinking=True,
                )
                if config.thinking
                else runtime.prepare_generation(
                    runtime.processor,
                    image,
                    sample.prompt,
                )
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
                raise ValueError("Direct generation output has an invalid shape")
            generated = [int(value) for value in output[0, input_length:].tolist()]
            raw, reasoning, final, thinking_completed = split_generated_response(
                generated, runtime.processor.tokenizer, thinking=config.thinking
            )
            prediction = parse_endpoint_response(final, sample.group)
            token_digest = hashlib.sha256(
                json.dumps(generated, separators=(",", ":")).encode()
            ).hexdigest()
            record = DirectGenerationRecord(
                evaluation_id=config.evaluation_id,
                benchmark=sample.benchmark,
                sample_id=sample.sample_id,
                global_sample_index=global_index,
                shard_rank=rank,
                shard_count=shard_count,
                split=sample.split,
                subset_tags=sample.subset_tags,
                group=sample.group,
                query=sample.query,
                reference=sample.reference,
                gold_endpoint=sample.gold_endpoint,
                prompt=sample.prompt,
                image_sha256=image_sha256,
                thinking=config.thinking,
                output_tokens=len(generated),
                ended_with_eos=generated[-1] in eos_ids,
                thinking_completed=thinking_completed,
                reasoning_response=reasoning,
                final_response=final,
                raw_response=raw,
                generated_token_sha256=token_digest,
                predicted_endpoint=prediction,
                parse_failure=prediction is None,
                correct=prediction == sample.gold_endpoint,
                generation_seconds=float(elapsed),
                metadata=dict(sample.metadata),
            )
            stream.write(
                json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            )
            next_position += 1
            if next_position % config.checkpoint_every == 0:
                checkpoint(stream)
                print(
                    f"direct generation rank {rank} {next_position}/{len(indices)}",
                    flush=True,
                )
        checkpoint(stream)
    with records_path.open(encoding="utf-8") as stream:
        summary = summarize_direct_generation_records(
            _record_from_mapping(json.loads(line)) for line in stream
        )
    summary.update(
        dataset_fingerprint=dataset_fingerprint,
        rank=rank,
        shard_count=shard_count,
        shard_sample_count=len(indices),
        max_new_tokens=config.max_new_tokens,
        decoding_protocol=config.decoding_protocol,
        thinking_budget="unset",
    )
    atomic_json(output_dir / "summary.json", summary)
    return summary


def iter_direct_generation_records(path: Path) -> Iterable[DirectGenerationRecord]:
    """Yield typed records from one JSONL file without retaining raw text."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield _record_from_mapping(json.loads(line))
