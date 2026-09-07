"""Versioned S-Space handoff format (ART-01 through ART-03).

This module is the only formal boundary between axis extraction and downstream
evaluation. It stores model-native covectors without
embedding training checkpoints or benchmark-specific state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.numpy import load_file, save_file


SCHEMA_VERSION = "2.0.0"
AXIS_ORDER = ("horizontal", "vertical", "distance")
NEGATIVE_ENDPOINTS = ("left", "above", "far")
POSITIVE_ENDPOINTS = ("right", "below", "close")
CONSTRUCTION_DATASET_ID = "coco_train2017_mask_depth_train6000_v2"
CONSTRUCTION_DATASET_FINGERPRINT = (
    "d826f948bc3db36c0cecd965118a6fa96c53ac29d91e7bf84a3b2abc2099fe6f"
)
VALIDATION_DATASET_ID = "coco_train2017_mask_depth_validation1800_v2"
VALIDATION_DATASET_FINGERPRINT = (
    "125146e175ddb8f688628caca0e4544e3049bc12a80fca2924d75b192e8b7c3d"
)
PROMPT_SET_ID = "coco_spatial_relations_v2"
PROMPT_STYLES = ("baseline", "direct", "natural", "choice_first", "terse")
ROLE_GRADIENT = "(grad_query(z_g)-grad_reference(z_g))/2"
VALIDATION_FIELDS = {
    "dataset_id",
    "dataset_fingerprint",
    "sample_count",
    "train_image_overlap_count",
    "decision_protocol",
    "score",
    "selection_metric",
    "selection_rule",
    "selected_layer",
    "selected_metrics",
    "per_layer",
}
LAYER_METRIC_FIELDS = {
    "layer_id",
    "correct",
    "total",
    "overall_accuracy",
    "axis_accuracy",
    "worst_axis_accuracy",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize artifact metadata as UTF-8 with LF and one final newline."""

    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _load_canonical_json(path: Path) -> Any:
    """Parse JSON only when its bytes use UTF-8, LF, and one final newline."""

    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw:
        raise ValueError(f"{path.name} must use UTF-8 without BOM and LF line endings")
    if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError(f"{path.name} must end with exactly one LF")
    return json.loads(raw.decode("utf-8"))


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_probability(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not np.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return number


def _validate_validation(
    value: Any,
    model: "ModelSpec",
    validation_samples: int,
) -> None:
    if not isinstance(value, dict) or set(value) != VALIDATION_FIELDS:
        raise ValueError("Construction validation fields differ from schema 2")
    expected = {
        "dataset_id": VALIDATION_DATASET_ID,
        "dataset_fingerprint": VALIDATION_DATASET_FINGERPRINT,
        "sample_count": validation_samples,
        "train_image_overlap_count": 0,
        "decision_protocol": "margin_sign_v1",
        "score": "v^T(h_query-h_reference)",
        "selection_metric": "margin_sign_overall_accuracy",
        "selection_rule": [
            "overall_accuracy",
            "worst_axis_accuracy",
            "shallower_layer_id",
        ],
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"Construction validation {name} differs")
    if not all(
        _is_integer(value[name])
        for name in ("sample_count", "train_image_overlap_count", "selected_layer")
    ):
        raise ValueError("Validation counts and selected layer must be integers")
    rows = value.get("per_layer")
    if not isinstance(rows, list) or len(rows) != len(model.layer_ids):
        raise ValueError("Validation must contain one metric row per exported layer")
    seen_layers = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != LAYER_METRIC_FIELDS:
            raise ValueError("Validation layer metric fields differ")
        layer = row["layer_id"]
        correct = row["correct"]
        total = row["total"]
        if not all(_is_integer(number) for number in (layer, correct, total)):
            raise ValueError("Validation layer, correct, and total must be integers")
        if total != validation_samples or not 0 <= correct <= total:
            raise ValueError("Validation metric counts are inconsistent")
        overall = _finite_probability(row["overall_accuracy"], "overall_accuracy")
        if not np.isclose(overall, correct / total, rtol=0.0, atol=1e-15):
            raise ValueError("Validation overall accuracy differs from its counts")
        axis = row["axis_accuracy"]
        if not isinstance(axis, (list, tuple)) or len(axis) != 3:
            raise ValueError("Validation axis accuracy must follow H/V/D order")
        axis_values = tuple(
            _finite_probability(number, "axis_accuracy") for number in axis
        )
        worst = _finite_probability(row["worst_axis_accuracy"], "worst_axis_accuracy")
        if worst != min(axis_values):
            raise ValueError("Validation worst-axis accuracy is inconsistent")
        seen_layers.append(layer)
    if tuple(seen_layers) != model.layer_ids:
        raise ValueError("Validation layer rows differ from artifact layer IDs")
    selected = max(
        rows,
        key=lambda row: (
            float(row["overall_accuracy"]),
            float(row["worst_axis_accuracy"]),
            -int(row["layer_id"]),
        ),
    )
    if value.get("selected_metrics") != selected:
        raise ValueError("Validation selected metrics differ from the frozen rule")
    if value.get("selected_layer") != model.selected_layer:
        raise ValueError("Validation and model selected layers differ")
    if selected["layer_id"] != model.selected_layer:
        raise ValueError("Model selected layer is not the validation winner")


def _validate_construction(value: Any, model: "ModelSpec") -> None:
    if not isinstance(value, dict):
        raise ValueError("Construction metadata must be an object")
    required = {
        "dataset_id",
        "dataset_fingerprint",
        "dataset_manifest_sha256",
        "train_samples",
        "validation_samples",
        "prompt_set",
        "prompt_styles",
        "orientations",
        "object_token_protocol",
        "role_gradient",
        "axis_order",
        "validation",
    }
    if not required.issubset(value):
        raise ValueError(f"Construction metadata lacks {sorted(required - set(value))}")
    expected = {
        "dataset_id": CONSTRUCTION_DATASET_ID,
        "dataset_fingerprint": CONSTRUCTION_DATASET_FINGERPRINT,
        "prompt_set": PROMPT_SET_ID,
        "prompt_styles": list(PROMPT_STYLES),
        "orientations": ["original", "swapped"],
        "object_token_protocol": "last_overlapping_subtoken",
        "role_gradient": ROLE_GRADIENT,
        "axis_order": list(AXIS_ORDER),
    }
    for name, expected_value in expected.items():
        if value.get(name) != expected_value:
            raise ValueError(f"Construction {name} differs from schema 2")
    # The manifest records local paths and software versions; its byte hash is
    # provenance, while the dataset fingerprint above is the portable identity.
    manifest_digest = value["dataset_manifest_sha256"]
    if not isinstance(manifest_digest, str) or re.fullmatch(
        r"[0-9a-f]{64}", manifest_digest
    ) is None:
        raise ValueError("Construction dataset_manifest_sha256 must be a SHA-256 digest")
    train_samples = value["train_samples"]
    validation_samples = value["validation_samples"]
    if not all(
        _is_integer(number) and number > 0
        for number in (train_samples, validation_samples)
    ):
        raise ValueError("Construction sample counts must be positive integers")
    _validate_validation(value["validation"], model, validation_samples)
    migration = value.get("data_provenance_migration")
    if migration is not None:
        migration_fields = {
            "protocol",
            "source_construction_dataset_id",
            "source_construction_dataset_fingerprint",
            "source_validation_dataset_id",
            "source_validation_dataset_fingerprint",
            "construction_rows_sha256",
            "validation_rows_sha256",
        }
        if not isinstance(migration, dict) or set(migration) != migration_fields:
            raise ValueError("Data provenance migration fields differ")
        migration_expected = {
            "protocol": "split_dataset_identity_migration_v1",
            "source_construction_dataset_id": (
                "coco_train2017_mask_depth_train6000_val300_v1"
            ),
            "source_construction_dataset_fingerprint": (
                "487754bff7f986c5d73bc9a159237430b65bdf1caec7a9f8356aa8479bf1b5fd"
            ),
            "source_validation_dataset_id": (
                "coco_train2017_mask_depth_train6000_val1800_v2"
            ),
            "source_validation_dataset_fingerprint": (
                "4e7bae31bcc6b937afffc1497e5355e3dfafd516d4ad02552b276d7910ecd283"
            ),
            "construction_rows_sha256": (
                "b5082e8738b2abefe246961b50ea4662aee8d1e8b4dea273e01916703c49d0a5"
            ),
            "validation_rows_sha256": (
                "dc41e43ee8f36063050305cdc4e40284700c36ff9d748b57db8527e56ef92cb6"
            ),
        }
        if migration != migration_expected:
            raise ValueError("Data provenance migration evidence is invalid")
    statistics = value.get("axis_statistics")
    if statistics is not None:
        fields = {
            "protocol",
            "minimum_mean_gradient_l2",
            "raw_mean_gradient_l2",
        }
        if not isinstance(statistics, dict) or set(statistics) != fields:
            raise ValueError("Axis statistics fields differ")
        threshold = statistics["minimum_mean_gradient_l2"]
        raw = np.asarray(statistics["raw_mean_gradient_l2"])
        if (
            statistics["protocol"] != "mean_gradient_l2_gate_v1"
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not np.isfinite(threshold)
            or threshold <= 0
            or raw.ndim != 2
            or raw.shape[0] != 3
            or raw.shape[1] <= max(model.layer_ids)
            or not np.issubdtype(raw.dtype, np.number)
            or not np.isfinite(raw).all()
            or np.any(raw[:, model.layer_ids] < float(threshold))
        ):
            raise ValueError("Axis raw norms or degeneracy gate are invalid")


def _validate_software(value: Any) -> None:
    required = {"python", "torch", "numpy", "transformers"}
    allowed = required | {"safetensors"}
    if (
        not isinstance(value, dict)
        or not required.issubset(value)
        or not set(value) <= allowed
    ):
        raise ValueError("Software metadata fields differ from schema 2")
    if any(not isinstance(item, str) or not item.strip() for item in value.values()):
        raise ValueError("Software versions must be non-empty strings")


@dataclass(frozen=True)
class ModelSpec:
    """Identify the exact model representation space used by an artifact.

    Axes are model-native covectors. Model and tokenizer revisions, hidden
    size, layer IDs, and layer numbering must therefore match at evaluation.
    """

    model_id: str
    revision: str
    architecture: str
    tokenizer_id: str
    tokenizer_revision: str
    hidden_size: int
    layer_ids: tuple[int, ...]
    selected_layer: int
    layer_numbering: str = "zero_based_post_block"


@dataclass(frozen=True)
class SSpaceManifest:
    """Store immutable provenance and conventions for one S-Space artifact.

    The manifest records everything required to decide whether the tensors can
    be interpreted by a downstream model. Large numeric arrays remain in the
    safetensors file and are covered by separate checksums.
    """

    artifact_id: str
    method: str
    model: ModelSpec
    construction: dict[str, Any]
    software: dict[str, str]
    schema_version: str = SCHEMA_VERSION
    axis_order: tuple[str, ...] = AXIS_ORDER
    negative_endpoints: tuple[str, ...] = NEGATIVE_ENDPOINTS
    positive_endpoints: tuple[str, ...] = POSITIVE_ENDPOINTS
    normalization: str = "l2_unit_per_layer_axis"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest without changing field values.

        Side effects:
            None.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SSpaceManifest":
        """Construct a manifest from an already parsed JSON object.

        Raises:
            KeyError: A required schema field is absent.
            TypeError: A nested record does not match its declared dataclass.
        """
        expected = {
            "artifact_id",
            "method",
            "model",
            "construction",
            "software",
            "schema_version",
            "axis_order",
            "negative_endpoints",
            "positive_endpoints",
            "normalization",
        }
        if set(value) != expected:
            raise ValueError(f"Manifest fields {set(value)} != {expected}")
        model = dict(value["model"])
        model_expected = {
            "model_id",
            "revision",
            "architecture",
            "tokenizer_id",
            "tokenizer_revision",
            "hidden_size",
            "layer_ids",
            "selected_layer",
            "layer_numbering",
        }
        if set(model) != model_expected:
            raise ValueError(f"Model fields {set(model)} != {model_expected}")
        model["layer_ids"] = tuple(model["layer_ids"])
        return cls(
            artifact_id=str(value["artifact_id"]),
            method=str(value["method"]),
            model=ModelSpec(**model),
            construction=dict(value["construction"]),
            software=dict(value["software"]),
            schema_version=str(value["schema_version"]),
            axis_order=tuple(value["axis_order"]),
            negative_endpoints=tuple(value["negative_endpoints"]),
            positive_endpoints=tuple(value["positive_endpoints"]),
            normalization=str(value["normalization"]),
        )


@dataclass(frozen=True)
class SSpaceArtifact:
    """Expose validated per-layer H/V/D axes (ART-01).

    Shapes:
        ``axes_tensor`` is ``[L, 3, D]`` in H/V/D order.
    """

    manifest: SSpaceManifest
    axes_tensor: np.ndarray

    def validate(self) -> None:
        """Check every public artifact invariant in a fixed order (ART-01).

        The checks establish: supported schema and method; fixed H/V/D endpoint
        semantics; unique layer IDs; tensor shape; floating finite values; and
        unit L2 axes. Validation never
        changes, reorders, or renormalizes the supplied tensors.

        Raises:
            ValueError: The first schema, convention, shape, numerical, or
                normalization invariant that fails.

        Side effects:
            None.
        """
        manifest = self.manifest
        if not manifest.artifact_id:
            raise ValueError("Artifact ID must be non-empty")
        if manifest.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version {manifest.schema_version!r}")
        if manifest.method != "final_logit_jacobian":
            raise ValueError(f"Unsupported extraction method {manifest.method!r}")
        if manifest.axis_order != AXIS_ORDER:
            raise ValueError(f"Axis order must be {AXIS_ORDER}")
        if (
            manifest.negative_endpoints != NEGATIVE_ENDPOINTS
            or manifest.positive_endpoints != POSITIVE_ENDPOINTS
        ):
            raise ValueError("Axis endpoints do not match the public convention")
        if manifest.normalization != "l2_unit_per_layer_axis":
            raise ValueError("Unsupported axis normalization")
        model = manifest.model
        required_strings = {
            "model_id": model.model_id,
            "revision": model.revision,
            "architecture": model.architecture,
            "tokenizer_id": model.tokenizer_id,
            "tokenizer_revision": model.tokenizer_revision,
            "layer_numbering": model.layer_numbering,
        }
        empty = [
            name
            for name, value in required_strings.items()
            if not isinstance(value, str) or not value
        ]
        if empty:
            raise ValueError(f"Model identity fields must be non-empty: {empty}")
        if not _is_integer(model.hidden_size) or model.hidden_size <= 0:
            raise ValueError("Model hidden size must be positive")
        layers = manifest.model.layer_ids
        if not layers or any(not _is_integer(layer) for layer in layers):
            raise ValueError("Layer IDs must be non-empty integers")
        if len(set(layers)) != len(layers):
            raise ValueError("Layer IDs must be non-empty and unique")
        if tuple(sorted(layers)) != layers:
            raise ValueError("Layer IDs must be strictly increasing")
        if not _is_integer(model.selected_layer) or model.selected_layer not in layers:
            raise ValueError("Selected layer must be present in layer IDs")
        _validate_construction(manifest.construction, model)
        _validate_software(manifest.software)
        expected_axes = (len(layers), 3, manifest.model.hidden_size)
        if self.axes_tensor.shape != expected_axes:
            raise ValueError(f"axes shape {self.axes_tensor.shape} != {expected_axes}")
        if (
            not np.issubdtype(self.axes_tensor.dtype, np.floating)
            or not np.isfinite(self.axes_tensor).all()
        ):
            raise ValueError("axes must contain finite floating-point values")
        norms = np.linalg.norm(self.axes_tensor.astype(np.float64), axis=-1)
        if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-5):
            raise ValueError("Every layer-axis vector must have unit L2 norm")

    def _layer_index(self, layer: int) -> int:
        try:
            return self.manifest.model.layer_ids.index(int(layer))
        except ValueError as error:
            raise KeyError(f"Layer {layer} is not present in this artifact") from error

    def axes(self, layer: int) -> np.ndarray:
        """Return H/V/D unit axes for one exported layer as ``[3, D]``.

        Raises:
            KeyError: The requested layer was not exported.

        Side effects:
            None.
        """
        return self.axes_tensor[self._layer_index(layer)]

    def project_objects(self, hidden: np.ndarray, layer: int) -> np.ndarray:
        """Project object states onto one layer's axes.

        Computes ``hidden @ axes[layer].T``. Input shape is ``[..., D]`` and
        output shape is ``[..., 3]`` in H/V/D order.

        Raises:
            ValueError: The final hidden dimension is not ``D``.
            KeyError: The requested layer was not exported.

        Side effects:
            None.
        """
        value = np.asarray(hidden)
        if value.ndim == 0 or value.shape[-1] != self.manifest.model.hidden_size:
            raise ValueError("Object hidden state is incompatible with the artifact")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError("Object hidden state must contain finite numeric values")
        return value @ self.axes(layer).T

    def project_pairwise(
        self, query_hidden: np.ndarray, reference_hidden: np.ndarray, layer: int
    ) -> np.ndarray:
        """Project a directed object relation into H/V/D scores (EVAL-01).

        The implemented equation is
        ``(h_query - h_reference) @ axes[layer].T``. Role subtraction removes
        components shared by both objects and preserves the directed relation.
        Inputs must broadcast to ``[..., D]``; the result is ``[..., 3]``.

        Raises:
            ValueError: The states cannot broadcast or the final dimension is
                not the model hidden size.
            KeyError: The requested layer was not exported.

        Side effects:
            None.
        """
        query = np.asarray(query_hidden)
        reference = np.asarray(reference_hidden)
        if query.shape != reference.shape:
            raise ValueError(
                f"Query shape {query.shape} must equal reference shape {reference.shape}"
            )
        return self.project_objects(query - reference, layer)

    def classify_pairwise(self, scores: np.ndarray, layer: int) -> np.ndarray:
        """Map pairwise scores to endpoint signs (EVAL-02).

        Returns ``-1`` for left/above/far, ``0`` for an exact tie, and ``+1``
        for right/below/close. Input shape must end in three scores. The fixed
        decision boundary is zero, and a zero decision cannot match a
        directional ground-truth sign.

        Returns:
            Signed ``int8`` predictions with the same shape as ``scores``.

        Raises:
            ValueError: The score tensor does not end in H/V/D components.
            KeyError: The requested layer was not exported.

        Side effects:
            None.
        """
        self._layer_index(layer)
        value = np.asarray(scores)
        if value.ndim == 0 or value.shape[-1] != 3:
            raise ValueError("Pairwise scores must end in three H/V/D components")
        if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
            raise ValueError("Pairwise scores must contain finite numeric values")
        return np.sign(value).astype(np.int8, copy=False)

    def save(self, directory: Path) -> None:
        """Validate and create a new immutable artifact directory (ART-02).

        The write order is tensors, manifest, then checksums. Safetensors stores
        exactly ``axes``. Both JSON files use UTF-8 bytes,
        LF line endings, and one final newline so checksums are independent of
        the host platform. The target directory must not exist because released
        artifact IDs are immutable.

        Raises:
            FileExistsError: The target directory already exists.
            ValueError: ART-01 validation fails.

        Side effects:
            Creates one artifact directory and three files.
        """
        self.validate()
        directory = Path(directory)
        if directory.exists():
            raise FileExistsError(directory)
        temporary = directory.with_name(f"{directory.name}.incomplete")
        if temporary.exists():
            raise FileExistsError(temporary)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.mkdir(exist_ok=False)
        try:
            tensors_path = temporary / "tensors.safetensors"
            save_file(
                {"axes": self.axes_tensor.astype(np.float32, copy=False)}, tensors_path
            )
            manifest_path = temporary / "manifest.json"
            manifest_path.write_bytes(_canonical_json_bytes(self.manifest.to_dict()))
            checksums = {
                name: _sha256(temporary / name)
                for name in ("manifest.json", "tensors.safetensors")
            }
            (temporary / "checksums.json").write_bytes(_canonical_json_bytes(checksums))
            type(self).load(temporary)
            os.replace(temporary, directory)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(cls, directory: Path) -> "SSpaceArtifact":
        """Load only after file and semantic validation (ART-03).

        The fixed order is: require canonical UTF-8/LF JSON, verify both file
        checksums, parse the manifest, require exactly one tensor field,
        construct the object, then execute all ART-01 checks. No legacy or
        repair path is attempted.

        Raises:
            ValueError: A checksum, field set, schema, convention, shape, or
                numerical invariant is invalid.
            FileNotFoundError: A required artifact file is absent.

        Side effects:
            Reads three files without modifying them.
        """
        directory = Path(directory)
        checksums = _load_canonical_json(directory / "checksums.json")
        expected_checksums = {"manifest.json", "tensors.safetensors"}
        if set(checksums) != expected_checksums:
            raise ValueError(
                f"Checksum fields {set(checksums)} != {expected_checksums}"
            )
        for name in sorted(expected_checksums):
            if checksums.get(name) != _sha256(directory / name):
                raise ValueError(f"Checksum mismatch for {name}")
        manifest = SSpaceManifest.from_dict(
            _load_canonical_json(directory / "manifest.json")
        )
        tensors = load_file(directory / "tensors.safetensors")
        expected = {"axes"}
        if set(tensors) != expected:
            raise ValueError(f"Tensor archive fields {set(tensors)} != {expected}")
        artifact = cls(
            manifest=manifest,
            axes_tensor=tensors["axes"],
        )
        artifact.validate()
        return artifact

    def validate_model(self, model: ModelSpec) -> None:
        """Reject a downstream model that differs from the artifact space.

        Compatibility covers the model and tokenizer identities, revisions,
        architecture, hidden size, exported post-block layer availability, and
        layer numbering. The runtime may expose additional layers that the
        artifact does not export. ``selected_layer`` is an artifact readout
        choice and is not a property of the downstream model, so it is not
        compared.

        Args:
            model: Explicit downstream model metadata.

        Raises:
            ValueError: Any representation-space field differs.

        Side effects:
            None.

        Example:
            ``artifact.validate_model(runtime_model_spec)``
        """
        expected = self.manifest.model
        fields = (
            "model_id",
            "revision",
            "architecture",
            "tokenizer_id",
            "tokenizer_revision",
            "hidden_size",
            "layer_numbering",
        )
        mismatches = [
            name for name in fields if getattr(expected, name) != getattr(model, name)
        ]
        if mismatches:
            raise ValueError(f"Model is incompatible in fields: {mismatches}")
        runtime_layers = tuple(int(layer) for layer in model.layer_ids)
        if not runtime_layers or runtime_layers != tuple(sorted(set(runtime_layers))):
            raise ValueError("Runtime model layer IDs must be unique and increasing")
        missing_layers = sorted(set(expected.layer_ids) - set(runtime_layers))
        if missing_layers:
            raise ValueError(
                f"Runtime model does not expose artifact layers: {missing_layers}"
            )
