"""Bind launcher resume decisions to the exact experiment inputs."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path
from typing import Any

from sspace.data.assets import MODEL_ASSETS
from sspace.run_records import (
    canonical_fingerprint,
    file_sha256,
    path_sha256,
    python_source_identity,
)


MODEL_ASSET_MARKER = ".sspace_hub_asset.json"
MODEL_ASSETS_BY_KEY = {asset.key: asset for asset in MODEL_ASSETS}


def runtime_identity(model_adapter: str, project_root: Path) -> dict[str, Any]:
    """Bind non-config workflows to the actual code and inference backend."""
    from sspace.core.models.runtime import RUNTIME_SPECS

    spec = RUNTIME_SPECS[model_adapter]
    return {
        "python_source": python_source_identity(project_root),
        "model_revision": spec.revision,
        "attention_backend": spec.attention_backend,
        "sdp_kernel": spec.sdp_kernel,
        "dtype": str(spec.dtype),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
    }


def _resolve(path: str, project_root: Path) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else project_root / value).resolve()


def _declared_data_identity(values: dict[str, Any]) -> dict[str, Any]:
    """Return config-declared immutable data checksums and fingerprints."""
    return {
        key: value
        for key, value in sorted(values.items())
        if key.endswith("_fingerprint")
        or (key.endswith("_sha256") and not key.startswith("artifact_"))
    }


def _model_identity(values: dict[str, Any], project_root: Path) -> dict[str, Any]:
    """Identify a pinned model without hashing multi-gigabyte weight files."""
    model_adapter = values.get("model_adapter")
    if model_adapter not in MODEL_ASSETS_BY_KEY:
        raise ValueError(f"Unknown model asset for adapter {model_adapter!r}")
    expected = MODEL_ASSETS_BY_KEY[model_adapter]
    model_dir = _resolve(str(values["model_path"]), project_root)
    identity: dict[str, Any] = {
        "path": str(model_dir),
        "present": model_dir.is_dir(),
        "expected": {
            "repo_id": expected.repo_id,
            "repo_type": expected.repo_type,
            "revision": expected.revision,
        },
    }
    if not model_dir.is_dir():
        identity["identity_source"] = None
        identity["asset_marker_sha256"] = None
        return identity

    marker_path = model_dir / MODEL_ASSET_MARKER
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        expected_marker = {
            "schema_version": "1.0.0",
            "repo_id": expected.repo_id,
            "repo_type": expected.repo_type,
            "revision": expected.revision,
        }
        if any(marker.get(key) != value for key, value in expected_marker.items()):
            raise ValueError(f"Model asset marker does not match {model_adapter}")
        identity["identity_source"] = "asset_marker"
        identity["asset_marker_sha256"] = file_sha256(marker_path)
        return identity

    if model_dir.name != expected.revision:
        raise ValueError(
            f"Model directory for {model_adapter} has no asset marker and is not "
            f"the pinned revision directory {expected.revision}"
        )
    identity["identity_source"] = "resolved_revision_directory"
    identity["asset_marker_sha256"] = None
    return identity


def experiment_launch_identity(
    config_path: Path,
    project_root: Path,
    limit: int | None,
) -> dict[str, Any]:
    """Fingerprint one config, its nested prompt configs, artifact, data, and limit."""
    path = config_path.resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    effective_limit = limit if limit is not None else values.get("limit")
    identity: dict[str, Any] = {
        "config_path": str(path),
        "config_sha256": file_sha256(path),
        "limit": effective_limit,
        "declared_data_identity": _declared_data_identity(values),
        "python_source": python_source_identity(project_root),
    }
    model_value = values.get("model_path")
    if model_value is not None:
        identity["model"] = _model_identity(values, project_root)
    artifact_value = values.get("artifact_dir")
    if artifact_value is not None:
        artifact_dir = _resolve(str(artifact_value), project_root)
        identity["artifact_dir"] = str(artifact_dir)
        identity["artifact_sha256"] = (
            path_sha256(artifact_dir) if artifact_dir.exists() else None
        )
    source_values = values.get("source_configs")
    if source_values is not None:
        identity["source_configs"] = [
            experiment_launch_identity(
                _resolve(str(source), project_root), project_root, None
            )
            for source in source_values
        ]
    return identity


def experiment_launch_fingerprint(
    config_path: Path,
    project_root: Path,
    limit: int | None,
) -> str:
    """Return the canonical fingerprint used by runners and suite resume checks."""
    return canonical_fingerprint(
        experiment_launch_identity(config_path, project_root, limit)
    )


def bind_launch_identity(
    resolved: dict[str, Any],
    config_path: Path,
    project_root: Path,
    limit: int | None,
) -> None:
    """Add the shared launch identity and fingerprint to a resolved run config."""
    identity = experiment_launch_identity(config_path, project_root, limit)
    resolved["launch_identity"] = identity
    resolved["launch_fingerprint"] = canonical_fingerprint(identity)
