"""Store a deterministic extraction shard with resume safety."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from sspace.run_records import atomic_json, canonical_fingerprint, file_sha256


SHARD_ARRAY_FILES = (
    "global_sample_indices.npy",
    "global_job_indices.npy",
    "object_states.npy",
    "role_gradients.npy",
    "logit_contrasts.npy",
    "gradient_sums.npy",
    "gradient_counts.npy",
)


@dataclass(frozen=True)
class ShardLayout:
    """Declare exact worker array sizes and dtypes for TRN-02D."""

    job_count: int
    model_layer_count: int
    hidden_size: int

    def validate(self) -> None:
        if self.job_count <= 0 or self.model_layer_count <= 0 or self.hidden_size <= 0:
            raise ValueError("Shard dimensions must all be positive")


class ShardStorage:
    """Append paired jobs to one worker shard and checkpoint committed state.

    Array shapes are ``object_states[J,2,L,2,D]``,
    ``role_gradients[J,2,L,D]``, and ``logit_contrasts[J,2,3]``. A checkpoint
    commits only complete paired jobs after all arrays and JSONL records have
    been flushed. Resume truncates the record file to that committed boundary.
    """

    def __init__(
        self,
        directory: Path,
        run_config: dict[str, Any],
        rank: int,
        world_size: int,
        global_sample_indices: np.ndarray,
        global_job_indices: np.ndarray,
        layout: ShardLayout,
    ) -> None:
        layout.validate()
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("Worker rank must be in [0, world_size)")
        self.layout = layout
        self.run_fingerprint = canonical_fingerprint(run_config)
        self.sample_indices = np.asarray(global_sample_indices, dtype=np.int64)
        self.job_indices = np.asarray(global_job_indices, dtype=np.int64)
        if self.job_indices.shape != (layout.job_count,):
            raise ValueError("Global job IDs do not match the shard job count")
        if not np.array_equal(self.sample_indices, np.unique(self.sample_indices)):
            raise ValueError("Worker sample IDs must be unique and increasing")
        if not np.array_equal(self.job_indices, np.unique(self.job_indices)):
            raise ValueError("Worker job IDs must be unique and increasing")
        if (self.sample_indices.size and int(self.sample_indices[0]) < 0) or (
            self.job_indices.size and int(self.job_indices[0]) < 0
        ):
            raise ValueError("Worker sample and job IDs must be nonnegative")
        if self.sample_indices.size and not np.all(
            self.sample_indices % self.world_size == self.rank
        ):
            raise ValueError("Worker samples violate the frozen modulo assignment")

        self.records_path = self.directory / "records.jsonl"
        self.checkpoint_path = self.directory / "checkpoint.json"
        self.accumulator_path = self.directory / "checkpoint_accumulators.npz"
        self.object_states: np.memmap
        self.role_gradients: np.memmap
        self.logit_contrasts: np.memmap
        self.gradient_sums = np.zeros(
            (3, layout.model_layer_count, layout.hidden_size), dtype=np.float64
        )
        self.gradient_counts = np.zeros(3, dtype=np.int64)
        self.next_job = 0
        self._open_or_create(run_config)

    def _array_shapes(self) -> dict[str, tuple[int, ...]]:
        value = self.layout
        return {
            "object_states.npy": (
                value.job_count,
                2,
                value.model_layer_count,
                2,
                value.hidden_size,
            ),
            "role_gradients.npy": (
                value.job_count,
                2,
                value.model_layer_count,
                value.hidden_size,
            ),
            "logit_contrasts.npy": (value.job_count, 2, 3),
        }

    def _create_float_memmap(self, name: str, shape: tuple[int, ...]) -> np.memmap:
        path = self.directory / name
        if path.exists():
            raise FileExistsError(path)
        array = open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
        array[:] = np.nan
        array.flush()
        return array

    def _open_or_create(self, run_config: dict[str, Any]) -> None:
        manifest_path = self.directory / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("status") == "complete":
                raise FileExistsError(
                    f"Completed worker shard is immutable: {self.directory}"
                )
        if self.checkpoint_path.exists():
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
            if checkpoint.get("schema_version", "1.0.0") not in {"1.0.0", "2.0.0"}:
                raise ValueError("Unsupported worker checkpoint schema version")
            if checkpoint["run_fingerprint"] != self.run_fingerprint:
                raise ValueError("Worker checkpoint run fingerprint mismatch")
            if (
                checkpoint["rank"] != self.rank
                or checkpoint["world_size"] != self.world_size
            ):
                raise ValueError("Worker checkpoint rank identity mismatch")
            for name, expected_shape in self._array_shapes().items():
                value = np.load(self.directory / name, mmap_mode="r+")
                if value.shape != expected_shape or value.dtype != np.float32:
                    raise ValueError(f"Checkpoint array {name} has the wrong layout")
                setattr(self, name.removesuffix(".npy"), value)
            accumulator_name = checkpoint.get(
                "accumulator_file", "checkpoint_accumulators.npz"
            )
            if (
                Path(accumulator_name).name != accumulator_name
                or not accumulator_name.startswith("checkpoint_accumulators")
                or not accumulator_name.endswith(".npz")
            ):
                raise ValueError("Checkpoint accumulator filename is invalid")
            self.accumulator_path = self.directory / accumulator_name
            if not self.accumulator_path.is_file():
                raise FileNotFoundError(self.accumulator_path)
            if file_sha256(self.accumulator_path) != checkpoint["accumulator_sha256"]:
                raise ValueError("Checkpoint accumulator checksum mismatch")
            with np.load(self.accumulator_path, allow_pickle=False) as values:
                if set(values.files) != {"gradient_sums", "gradient_counts"}:
                    raise ValueError("Checkpoint accumulator fields differ")
                self.gradient_sums = values["gradient_sums"].astype(
                    np.float64, copy=True
                )
                self.gradient_counts = values["gradient_counts"].astype(
                    np.int64, copy=True
                )
            expected_sum = (3, self.layout.model_layer_count, self.layout.hidden_size)
            if (
                self.gradient_sums.shape != expected_sum
                or self.gradient_counts.shape != (3,)
            ):
                raise ValueError("Checkpoint accumulators have incompatible shapes")
            if not np.isfinite(self.gradient_sums).all() or np.any(
                self.gradient_counts < 0
            ):
                raise ValueError("Checkpoint accumulators contain invalid values")
            self.next_job = int(checkpoint["next_job"])
            if not 0 <= self.next_job <= self.layout.job_count:
                raise ValueError("Checkpoint next_job is outside the worker layout")
            if int(self.gradient_counts.sum()) != self.next_job * 2:
                raise ValueError(
                    "Checkpoint gradient counts differ from committed jobs"
                )
            records_offset = int(checkpoint["records_offset"])
            if not 0 <= records_offset <= self.records_path.stat().st_size:
                raise ValueError("Checkpoint record offset is invalid")
            with self.records_path.open("r+b") as stream:
                stream.truncate(records_offset)
            if len(self.records_path.read_text(encoding="utf-8").splitlines()) != (
                self.next_job * 2
            ):
                raise ValueError("Checkpoint record count differs from committed jobs")
            if self.next_job and not all(
                np.isfinite(value[: self.next_job]).all()
                for value in (
                    self.object_states,
                    self.role_gradients,
                    self.logit_contrasts,
                )
            ):
                raise FloatingPointError("Committed worker arrays are non-finite")
            stored_samples = np.load(self.directory / "global_sample_indices.npy")
            stored_jobs = np.load(self.directory / "global_job_indices.npy")
            if not np.array_equal(
                stored_samples, self.sample_indices
            ) or not np.array_equal(stored_jobs, self.job_indices):
                raise ValueError("Checkpoint global sample/job IDs differ")
            return

        reserved = [
            self.records_path,
            *(self.directory / name for name in SHARD_ARRAY_FILES),
        ]
        if any(path.exists() for path in reserved) or any(
            self.directory.glob("checkpoint_accumulators*.npz")
        ):
            raise FileExistsError("Incomplete shard files exist without a checkpoint")
        np.save(self.directory / "global_sample_indices.npy", self.sample_indices)
        np.save(self.directory / "global_job_indices.npy", self.job_indices)
        for name, shape in self._array_shapes().items():
            setattr(
                self, name.removesuffix(".npy"), self._create_float_memmap(name, shape)
            )
        self.records_path.touch(exist_ok=False)
        atomic_json(
            self.directory / "resolved_config.json",
            {
                "run": run_config,
                "rank": self.rank,
                "world_size": self.world_size,
                "run_fingerprint": self.run_fingerprint,
                "layout": {
                    "job_count": self.layout.job_count,
                    "model_layer_count": self.layout.model_layer_count,
                    "hidden_size": self.layout.hidden_size,
                },
            },
        )
        self.checkpoint()

    def append(
        self,
        object_states: np.ndarray,
        role_gradients: np.ndarray,
        logit_contrasts: np.ndarray,
        group_index: int,
        records: Sequence[dict[str, Any]],
    ) -> None:
        """Append one complete paired job at the next local position."""
        if self.next_job >= self.layout.job_count:
            raise IndexError("Worker shard already contains every declared job")
        expected_states = (2, self.layout.model_layer_count, 2, self.layout.hidden_size)
        expected_gradients = (2, self.layout.model_layer_count, self.layout.hidden_size)
        if np.asarray(object_states).shape != expected_states:
            raise ValueError("Object-state result has the wrong worker shape")
        if np.asarray(role_gradients).shape != expected_gradients:
            raise ValueError("Role-gradient result has the wrong worker shape")
        if np.asarray(logit_contrasts).shape != (2, 3):
            raise ValueError("Logit-contrast result has the wrong worker shape")
        if group_index not in range(3):
            raise ValueError("Gradient group index must be H, V, or D")
        values = (object_states, role_gradients, logit_contrasts)
        if not all(np.isfinite(np.asarray(value)).all() for value in values):
            raise FloatingPointError("Worker result contains non-finite values")

        local_job = self.next_job
        expected_global_job = int(self.job_indices[local_job])
        if len(records) != 2 or [record.get("orientation") for record in records] != [
            "original",
            "swapped",
        ]:
            raise ValueError("Each paired job requires original and swapped records")
        if any(
            int(record.get("global_job_index", -1)) != expected_global_job
            for record in records
        ):
            raise ValueError("Worker record global job ID differs from the layout")
        self.object_states[local_job] = np.asarray(object_states, dtype=np.float32)
        self.role_gradients[local_job] = np.asarray(role_gradients, dtype=np.float32)
        self.logit_contrasts[local_job] = np.asarray(logit_contrasts, dtype=np.float32)
        self.gradient_sums[group_index] += np.asarray(
            role_gradients, dtype=np.float64
        ).sum(axis=0)
        self.gradient_counts[group_index] += 2
        with self.records_path.open("a", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.next_job += 1

    def checkpoint(self) -> None:
        """Commit all complete jobs and accumulator state atomically."""
        self.object_states.flush()
        self.role_gradients.flush()
        self.logit_contrasts.flush()
        with self.records_path.open("ab") as stream:
            stream.flush()
            os.fsync(stream.fileno())
            records_offset = stream.tell()
        accumulator_path = self.directory / (
            f"checkpoint_accumulators_{self.next_job:08d}.npz"
        )
        if accumulator_path != self.accumulator_path or not accumulator_path.exists():
            temporary = accumulator_path.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez(
                    stream,
                    gradient_sums=self.gradient_sums,
                    gradient_counts=self.gradient_counts,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, accumulator_path)
        accumulator_sha256 = file_sha256(accumulator_path)
        previous_accumulator = self.accumulator_path
        atomic_json(
            self.checkpoint_path,
            {
                "schema_version": "2.0.0",
                "run_fingerprint": self.run_fingerprint,
                "rank": self.rank,
                "world_size": self.world_size,
                "next_job": self.next_job,
                "records_offset": records_offset,
                "accumulator_file": accumulator_path.name,
                "accumulator_sha256": accumulator_sha256,
            },
        )
        self.accumulator_path = accumulator_path
        if previous_accumulator != accumulator_path and previous_accumulator.exists():
            previous_accumulator.unlink()

    def complete(self) -> dict[str, Any]:
        """Validate, checksum, and seal one worker shard."""
        if self.next_job != self.layout.job_count:
            raise ValueError("Cannot complete a partial worker shard")
        self.checkpoint()
        if not all(
            np.isfinite(value).all()
            for value in (self.object_states, self.role_gradients, self.logit_contrasts)
        ):
            raise FloatingPointError(
                "Completed worker arrays contain non-finite values"
            )
        if len(self.records_path.read_text(encoding="utf-8").splitlines()) != (
            self.layout.job_count * 2
        ):
            raise ValueError("Completed worker record count differs from the layout")
        np.save(self.directory / "gradient_sums.npy", self.gradient_sums)
        np.save(self.directory / "gradient_counts.npy", self.gradient_counts)
        checksums = {
            name: file_sha256(self.directory / name)
            for name in (*SHARD_ARRAY_FILES, "records.jsonl", "resolved_config.json")
        }
        atomic_json(self.directory / "checksums.json", checksums)
        manifest = {
            "schema_version": "1.0.0",
            "status": "complete",
            "rank": self.rank,
            "world_size": self.world_size,
            "run_fingerprint": self.run_fingerprint,
            "sample_count": int(self.sample_indices.size),
            "job_count": self.layout.job_count,
            "record_count": self.layout.job_count * 2,
            "orientations": ["original", "swapped"],
            "object_role_order": ["query", "reference"],
            "model_layer_count": self.layout.model_layer_count,
            "hidden_size": self.layout.hidden_size,
            "gradient_counts": self.gradient_counts.tolist(),
            "checksums_sha256": file_sha256(self.directory / "checksums.json"),
        }
        atomic_json(self.directory / "manifest.json", manifest)
        return manifest


def validate_complete_shard(
    directory: Path, expected_fingerprint: str
) -> dict[str, Any]:
    """Validate one sealed shard and every content checksum before merge."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_manifest_fields = {
        "schema_version",
        "status",
        "rank",
        "world_size",
        "run_fingerprint",
        "sample_count",
        "job_count",
        "record_count",
        "orientations",
        "object_role_order",
        "model_layer_count",
        "hidden_size",
        "gradient_counts",
        "checksums_sha256",
    }
    if (
        set(manifest) != expected_manifest_fields
        or manifest.get("status") != "complete"
    ):
        raise ValueError(f"Worker shard is not complete: {directory}")
    if manifest["schema_version"] != "1.0.0":
        raise ValueError("Worker shard schema version differs")
    if manifest.get("run_fingerprint") != expected_fingerprint:
        raise ValueError("Worker shard run fingerprint mismatch")
    checksums_path = directory / "checksums.json"
    if file_sha256(checksums_path) != manifest["checksums_sha256"]:
        raise ValueError("Worker checksum manifest hash mismatch")
    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    expected = {*SHARD_ARRAY_FILES, "records.jsonl", "resolved_config.json"}
    if set(checksums) != expected:
        raise ValueError("Worker checksum file set differs from the schema")
    for name in sorted(expected):
        if file_sha256(directory / name) != checksums[name]:
            raise ValueError(f"Worker checksum mismatch for {name}")
    if len(
        (directory / "records.jsonl").read_text(encoding="utf-8").splitlines()
    ) != int(manifest["record_count"]):
        raise ValueError("Worker record count differs from its manifest")
    if manifest["record_count"] != manifest["job_count"] * 2:
        raise ValueError("Worker manifest job and record counts differ")
    return manifest
