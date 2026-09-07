"""Acquire the pinned external assets for COCO construction (PRE-00)."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from sspace.run_records import atomic_json

from .acquisition import _acquire_coco_images
from .annotations import load_coco_instances
from .schema import CocoImage


@dataclass(frozen=True)
class DownloadSpec:
    """Describe one immutable remote file and its local asset-root path."""

    url: str
    relative_path: str
    size: int
    sha256: str


ANNOTATION_ARCHIVE = DownloadSpec(
    "https://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "archives/annotations_trainval2017.zip",
    252_907_541,
    "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
)
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
DEPTH_MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
DEPTH_MODEL_FILES = (
    DownloadSpec(
        f"https://huggingface.co/{DEPTH_MODEL_ID}/resolve/{DEPTH_MODEL_REVISION}/config.json?download=true",
        "models/depth_anything_v2_small_hf/config.json",
        950,
        "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
    ),
    DownloadSpec(
        f"https://huggingface.co/{DEPTH_MODEL_ID}/resolve/{DEPTH_MODEL_REVISION}/preprocessor_config.json?download=true",
        "models/depth_anything_v2_small_hf/preprocessor_config.json",
        775,
        "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
    ),
    DownloadSpec(
        f"https://huggingface.co/{DEPTH_MODEL_ID}/resolve/{DEPTH_MODEL_REVISION}/model.safetensors?download=true",
        "models/depth_anything_v2_small_hf/model.safetensors",
        99_173_660,
        "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70",
    ),
)
COCO_ANNOTATION_MEMBER = "annotations/instances_train2017.json"
COCO_ANNOTATION_SIZE = 469_785_474
COCO_ANNOTATION_SHA256 = (
    "610fce4944abdeb15354cc765333805529359d12d88f2f711393ca586901d01d"
)
MANIFEST_ROOT = Path(__file__).with_name("manifests")
TRAIN_IMAGE_IDS = MANIFEST_ROOT / "coco6000_train_image_ids.txt"
VALIDATION_IMAGE_IDS = MANIFEST_ROOT / "coco1800_validation_image_ids.txt"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file(path: Path, size: int, sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != size:
        raise ValueError(f"Asset size {path.stat().st_size} != {size}: {path}")
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(f"Asset SHA-256 {actual} != {sha256}: {path}")


def _copy_response(response: BinaryIO, path: Path, expected_bytes: int) -> None:
    written = 0
    next_report = 256 * 1024 * 1024
    with path.open("ab") as stream:
        while chunk := response.read(8 * 1024 * 1024):
            stream.write(chunk)
            written += len(chunk)
            if written >= next_report:
                print(
                    f"downloaded {path}: {written}/{expected_bytes} bytes", flush=True
                )
                next_report += 256 * 1024 * 1024
        stream.flush()
        os.fsync(stream.fileno())
    if written != expected_bytes:
        raise ValueError(f"Download wrote {written} bytes, expected {expected_bytes}")


def _download_file(asset_root: Path, spec: DownloadSpec) -> str:
    destination = asset_root / spec.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _validate_file(destination, spec.size, spec.sha256)
        return "existing_verified"

    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > spec.size:
        raise ValueError(f"Partial download exceeds expected size: {partial}")
    request = urllib.request.Request(spec.url)
    if offset:
        request.add_header("Range", f"bytes={offset}-")
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", 200)
        expected_status = 206 if offset else 200
        if status != expected_status:
            raise ValueError(f"Download HTTP status {status} != {expected_status}")
        remaining = spec.size - offset
        content_length = response.headers.get("Content-Length")
        if content_length is None or int(content_length) != remaining:
            raise ValueError(f"Download Content-Length must equal {remaining}")
        _copy_response(response, partial, remaining)
    _validate_file(partial, spec.size, spec.sha256)
    os.replace(partial, destination)
    return "downloaded_verified"


def _extract_annotation(asset_root: Path, archive: Path) -> str:
    destination = asset_root / "datasets/coco2017/annotations/instances_train2017.json"
    if destination.exists():
        _validate_file(destination, COCO_ANNOTATION_SIZE, COCO_ANNOTATION_SHA256)
        return "existing_verified"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if partial.exists():
        raise FileExistsError(f"Remove interrupted annotation extraction: {partial}")
    with (
        zipfile.ZipFile(archive) as source,
        source.open(COCO_ANNOTATION_MEMBER) as member,
        partial.open("xb") as output,
    ):
        while chunk := member.read(8 * 1024 * 1024):
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    _validate_file(partial, COCO_ANNOTATION_SIZE, COCO_ANNOTATION_SHA256)
    os.replace(partial, destination)
    return "extracted_verified"


def _image_ids(path: Path, expected_count: int) -> tuple[int, ...]:
    values = tuple(int(line) for line in path.read_text(encoding="utf-8").splitlines())
    if len(values) != expected_count or len(set(values)) != expected_count:
        raise ValueError(f"Frozen COCO image IDs differ: {path}")
    return values


def _upgrade_coco_urls(manifest: dict[str, object]) -> dict[str, object]:
    """Normalize the official COCO origin from HTTP to HTTPS for safe migration."""
    for section in ("downloads", "images"):
        for record in manifest.get(section, ()):  # type: ignore[union-attr]
            url = record.get("url")
            if isinstance(url, str) and url.startswith(
                "http://images.cocodataset.org/"
            ):
                record["url"] = "https://" + url.removeprefix("http://")
    return manifest


def required_frozen_images(
    annotation_path: Path,
) -> tuple[tuple[str, CocoImage], ...]:
    """Load the exact 6,000 train and 1,800 validation image identities.

    Args:
        annotation_path: Verified official COCO 2017 train instances JSON.

    Returns:
        Ordered train rows followed by validation rows.

    Raises:
        ValueError: A manifest count, identity, overlap, or annotation differs.

    Side effects:
        Reads and parses the annotation file; it does not inspect local images.
    """
    images, _ = load_coco_instances(annotation_path)
    train = _image_ids(TRAIN_IMAGE_IDS, 6000)
    validation = _image_ids(VALIDATION_IMAGE_IDS, 1800)
    overlap = set(train) & set(validation)
    if overlap:
        raise ValueError(f"Frozen COCO train/validation overlap: {len(overlap)}")
    missing = (set(train) | set(validation)) - set(images)
    if missing:
        raise ValueError(f"Frozen COCO IDs absent from annotations: {len(missing)}")
    return tuple(
        [("train", images[image_id]) for image_id in train]
        + [("validation", images[image_id]) for image_id in validation]
    )


def prepare_coco_assets(asset_root: Path, workers: int) -> dict[str, object]:
    """Download and verify every immutable PRE-00 COCO construction input.

    Args:
        asset_root: User-selected persistent directory for datasets and models.
        workers: Positive number of concurrent COCO image downloads.

    Returns:
        JSON-safe source manifest containing revisions, checksums, and image records.

    Raises:
        FileNotFoundError: An expected downloaded or extracted file is absent.
        FileExistsError: An interrupted annotation extraction needs attention.
        ValueError: A response, size, checksum, image, or candidate count differs.

    Side effects:
        Downloads pinned inputs and writes ``manifest.json`` below ``asset_root``.

    Example:
        ``prepare_coco_assets(Path("/data/sspace-assets"), workers=16)``
    """
    if workers <= 0:
        raise ValueError("workers must be positive")
    root = Path(asset_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    expected_images: dict[int, str] = {}
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_images = {
            int(record["image_id"]): str(record["sha256"])
            for record in previous.get("images", ())
        }
    for spec in (ANNOTATION_ARCHIVE, *DEPTH_MODEL_FILES):
        _download_file(root, spec)
    annotation = root / "datasets/coco2017/annotations/instances_train2017.json"
    _extract_annotation(root, root / ANNOTATION_ARCHIVE.relative_path)
    required = required_frozen_images(annotation)
    records = _acquire_coco_images(
        required,
        root / "datasets/coco2017/train2017",
        root / ".tmp/coco_images",
        workers,
        expected_images,
    )
    stable_records = [
        {name: value for name, value in record.items() if name != "status"}
        for record in records
    ]
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "coco_release": "2017",
        "depth_model_id": DEPTH_MODEL_ID,
        "depth_model_revision": DEPTH_MODEL_REVISION,
        "downloads": [
            asdict(spec) for spec in (ANNOTATION_ARCHIVE, *DEPTH_MODEL_FILES)
        ],
        "annotation": {
            "relative_path": "datasets/coco2017/annotations/instances_train2017.json",
            "size": COCO_ANNOTATION_SIZE,
            "sha256": COCO_ANNOTATION_SHA256,
        },
        "train_image_ids_sha256": _sha256(TRAIN_IMAGE_IDS),
        "validation_image_ids_sha256": _sha256(VALIDATION_IMAGE_IDS),
        "train_image_count": 6000,
        "validation_image_count": 1800,
        "images": stable_records,
    }
    encoded = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if _upgrade_coco_urls(previous) != manifest:
            raise ValueError(
                f"Asset manifest differs from pinned PRE-00: {manifest_path}"
            )
        if manifest_path.read_text(encoding="utf-8") != encoded:
            atomic_json(manifest_path, manifest)
    else:
        atomic_json(manifest_path, manifest)
    return manifest
