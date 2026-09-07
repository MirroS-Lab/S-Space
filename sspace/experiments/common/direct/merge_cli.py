"""Verify and merge completed direct-generation shards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, TextIO

from sspace.run_records import RunRecorder, atomic_json, file_sha256
from sspace.experiments.identity import bind_launch_identity

from .config import load_direct_generation_config
from .dataset import load_direct_generation_samples
from .generation import (
    iter_direct_generation_records,
    parse_endpoint_response,
    summarize_direct_generation_records,
)
from .schema import DirectGenerationRecord, DirectGenerationSample


def _verify_checksums(directory: Path) -> None:
    checksums_path = directory / "checksums.json"
    if not checksums_path.is_file():
        raise FileNotFoundError(checksums_path)
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    for relative, expected in checksums.items():
        path = directory / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Direct shard checksum mismatch: {path}")


def _validate_shard_manifest(
    manifest: dict[str, Any],
    dataset_fingerprint: str,
    launch_fingerprint: str,
    rank: int,
    shard_count: int,
) -> None:
    """Require one shard to have been launched from this exact Config identity."""
    if manifest.get("status") != "complete":
        raise ValueError(f"Direct-generation shard {rank} is not complete")
    if manifest.get("dataset_fingerprint") != dataset_fingerprint:
        raise ValueError(f"Direct-generation shard {rank} dataset differs")
    config = manifest.get("config")
    if not isinstance(config, dict) or (
        config.get("launch_fingerprint") != launch_fingerprint
    ):
        raise ValueError(f"Direct-generation shard {rank} launch identity differs")
    if config.get("rank") != rank or config.get("effective_shard_count") != shard_count:
        raise ValueError(f"Direct-generation shard {rank} topology differs")


def _validate_record(
    record: DirectGenerationRecord,
    sample: DirectGenerationSample,
    global_index: int,
    rank: int,
    shard_count: int,
    evaluation_id: str,
    thinking: bool,
) -> None:
    expected = (
        evaluation_id,
        sample.benchmark,
        sample.sample_id,
        global_index,
        rank,
        shard_count,
        sample.split,
        sample.subset_tags,
        sample.group,
        sample.query,
        sample.reference,
        sample.gold_endpoint,
        sample.prompt,
        thinking,
    )
    actual = (
        record.evaluation_id,
        record.benchmark,
        record.sample_id,
        record.global_sample_index,
        record.shard_rank,
        record.shard_count,
        record.split,
        record.subset_tags,
        record.group,
        record.query,
        record.reference,
        record.gold_endpoint,
        record.prompt,
        record.thinking,
    )
    if actual != expected:
        raise ValueError(f"Direct shard record identity differs at {global_index}")
    if sample.image_sha256 is not None and record.image_sha256 != sample.image_sha256:
        raise ValueError(f"Direct shard image checksum differs at {global_index}")
    prediction = parse_endpoint_response(record.final_response, sample.group)
    expected_semantics = (
        prediction,
        prediction is None,
        prediction == sample.gold_endpoint,
    )
    actual_semantics = (
        record.predicted_endpoint,
        record.parse_failure,
        record.correct,
    )
    if actual_semantics != expected_semantics:
        raise ValueError(f"Direct shard parsed semantics differ at {global_index}")


def _merge_records_atomically(
    records_path: Path,
    streams: list[TextIO],
    samples: list[DirectGenerationSample],
    shard_count: int,
    evaluation_id: str,
    thinking: bool,
) -> None:
    """Validate a complete merge before atomically publishing records.jsonl."""
    temporary = records_path.with_suffix(records_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with temporary.open("x", encoding="utf-8") as output:
        for global_index, sample in enumerate(samples):
            rank = global_index % shard_count
            line = streams[rank].readline()
            if not line:
                raise ValueError(
                    f"Direct shard {rank} ended before source index {global_index}"
                )
            value = json.loads(line)
            value["subset_tags"] = tuple(value["subset_tags"])
            record = DirectGenerationRecord(**value)
            _validate_record(
                record,
                sample,
                global_index,
                rank,
                shard_count,
                evaluation_id,
                thinking,
            )
            output.write(line if line.endswith("\n") else line + "\n")
        output.flush()
        os.fsync(output.fileno())
    for rank, stream in enumerate(streams):
        if stream.readline():
            raise ValueError(f"Direct shard {rank} contains extra records")
    os.replace(temporary, records_path)


def main() -> None:
    """Merge source-order records only after strict all-shard verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    project_root = Path(__file__).resolve().parents[4]
    config = load_direct_generation_config(args.config, project_root)
    samples, fingerprint = load_direct_generation_samples(config)
    if args.limit is not None:
        samples = samples[: args.limit]
    output_dir = config.output_dir / "merged"
    effective_shard_count = min(config.shard_count, len(samples))
    resolved: dict[str, Any] = {
        **config.to_dict(),
        "config_path": str(args.config.resolve()),
        "dataset_fingerprint": fingerprint,
        "sample_limit": args.limit,
        "effective_shard_count": effective_shard_count,
        "merge_protocol": "verified_modulo_shards_source_order_v1",
    }
    bind_launch_identity(resolved, args.config, project_root, args.limit)
    recorder = RunRecorder(output_dir, resolved, project_root)
    stage = "direct_generation_merge"
    streams = []
    try:
        for rank in range(effective_shard_count):
            shard = config.output_dir / "shards" / f"rank_{rank:03d}"
            manifest = json.loads(
                (shard / "run_manifest.json").read_text(encoding="utf-8")
            )
            _validate_shard_manifest(
                manifest,
                fingerprint,
                resolved["launch_fingerprint"],
                rank,
                effective_shard_count,
            )
            _verify_checksums(shard)
            streams.append((shard / "records.jsonl").open(encoding="utf-8"))
        recorder.event(stage, "started", sample_count=len(samples))
        records_path = output_dir / "records.jsonl"
        _merge_records_atomically(
            records_path,
            streams,
            samples,
            effective_shard_count,
            config.evaluation_id,
            config.thinking,
        )
        summary = summarize_direct_generation_records(
            iter_direct_generation_records(records_path)
        )
        summary.update(
            dataset_fingerprint=fingerprint,
            shard_count=effective_shard_count,
            sample_count=len(samples),
            max_new_tokens=config.max_new_tokens,
            decoding_protocol=config.decoding_protocol,
            thinking_budget="unset",
        )
        atomic_json(output_dir / "summary.json", summary)
        recorder.event(stage, "complete", **summary["overall"])
        recorder.complete(sample_count=len(samples), dataset_fingerprint=fingerprint)
    except BaseException as error:
        recorder.fail(error, stage)
        raise
    finally:
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
