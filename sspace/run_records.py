"""Structured run records shared by formal command-line workflows."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def canonical_fingerprint(value: dict[str, Any]) -> str:
    """Return SHA-256 of canonical compact JSON for resume compatibility."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def failure_is_current(failure: dict[str, Any]) -> bool:
    """Ignore error markers left by an earlier distributed launch."""
    attempt = os.environ.get("SSPACE_ATTEMPT_ID")
    return attempt is None or failure.get("attempt_id") == attempt


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_sha256(path: Path) -> str:
    """Hash one file or a directory tree with stable relative file names."""
    value = Path(path)
    if value.is_file():
        return file_sha256(value)
    if not value.is_dir():
        raise FileNotFoundError(value)
    files = {
        str(item.relative_to(value)): file_sha256(item)
        for item in sorted(value.rglob("*"))
        if item.is_file()
    }
    if not files:
        raise ValueError(f"Cannot fingerprint empty directory {value}")
    return canonical_fingerprint({"files": files})


def python_source_identity(project_root: Path) -> dict[str, Any]:
    """Return one portable digest for the complete maintained Python package."""
    root = Path(project_root).resolve()
    source_root = root / "sspace"
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    sources = {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(source_root.rglob("*.py"))
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not sources:
        raise ValueError(f"Python source tree is empty: {source_root}")
    return {
        "schema_version": "1.0.0",
        "algorithm": "sha256-canonical-python-tree-v1",
        "file_count": len(sources),
        "sha256": canonical_fingerprint({"files": sources}),
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace one JSON record on the same filesystem."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


@contextmanager
def atomic_text(path: Path, *, newline: str | None = None):
    """Publish a derived text file only after its writer succeeds."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline=newline) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def code_revision(project_root: Path) -> dict[str, Any]:
    """Return optional Git metadata only when this project is the Git root."""
    root = Path(project_root).resolve()
    unavailable = {
        "available": False,
        "commit": None,
        "project_worktree_dirty": None,
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "--show-toplevel", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        lines = revision.stdout.splitlines()
        if (
            revision.returncode != 0
            or len(lines) != 2
            or Path(lines[0]).resolve() != root
        ):
            return unavailable
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--",
                ".",
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return unavailable
    if status.returncode != 0:
        return unavailable
    return {
        "available": True,
        "commit": lines[1],
        "project_worktree_dirty": bool(status.stdout),
    }


def code_provenance(project_root: Path) -> dict[str, Any]:
    """Bind optional Git metadata to the source digest used by every workflow."""
    return {
        "git": code_revision(project_root),
        "python_source": python_source_identity(project_root),
    }


class RunRecorder:
    """Write deterministic config, stage events, final status, and failures."""

    def __init__(
        self, directory: Path, config: dict[str, Any], project_root: Path
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = config
        self.fingerprint = canonical_fingerprint(config)
        resolved = self.directory / "resolved_config.json"
        encoded = json.dumps(config, indent=2, sort_keys=True) + "\n"
        if resolved.exists() and resolved.read_text(encoding="utf-8") != encoded:
            raise ValueError("Run directory contains a different resolved config")
        if not resolved.exists():
            resolved.write_text(encoded, encoding="utf-8")
        manifest_path = self.directory / "run_manifest.json"
        existing = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else None
        )
        if existing is not None:
            if existing.get("run_fingerprint") != self.fingerprint:
                raise ValueError("Run directory contains a different run fingerprint")
            if existing.get("status") == "complete":
                raise FileExistsError("Completed run directories are immutable")
        self.manifest = {
            "status": "running",
            "run_fingerprint": self.fingerprint,
            "created_at": (
                existing["created_at"]
                if existing is not None
                else datetime.now(timezone.utc).isoformat()
            ),
            "resumed_at": (
                datetime.now(timezone.utc).isoformat() if existing is not None else None
            ),
            "python": platform.python_version(),
            "code": code_provenance(project_root),
            "config": config,
        }
        atomic_json(self.directory / "run_manifest.json", self.manifest)

    def event(self, stage: str, status: str, **fields: Any) -> None:
        """Append one fsynced structured stage event."""
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "status": status,
            **fields,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        with (self.directory / "run.log").open("a", encoding="utf-8") as stream:
            details = " ".join(
                f"{name}={json.dumps(value, sort_keys=True)}"
                for name, value in sorted(fields.items())
            )
            stream.write(f"{record['timestamp']} {stage} {status} {details}\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _write_checksums(self) -> None:
        checksums = {}
        for path in sorted(self.directory.rglob("*")):
            if (
                not path.is_file()
                or path.name == "checksums.json"
                or path.suffix == ".tmp"
            ):
                continue
            checksums[str(path.relative_to(self.directory))] = file_sha256(path)
        atomic_json(self.directory / "checksums.json", checksums)

    def complete(self, **fields: Any) -> None:
        """Mark the run complete only after all required stages succeed."""
        self.manifest.update(
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            **fields,
        )
        atomic_json(self.directory / "run_manifest.json", self.manifest)
        (self.directory / "failure.json").unlink(missing_ok=True)
        self._write_checksums()

    def fail(self, error: BaseException, stage: str) -> None:
        """Persist the first failure context; callers must re-raise afterward."""
        failure = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "run_fingerprint": self.fingerprint,
        }
        atomic_json(self.directory / "failure.json", failure)
        self.manifest.update(status="failed", failure=failure)
        atomic_json(self.directory / "run_manifest.json", self.manifest)
        self._write_checksums()
