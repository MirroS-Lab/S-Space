"""Load frozen matched-pair datasets for direct generation."""

from __future__ import annotations

from sspace.run_records import canonical_fingerprint
from sspace.core.data import load_construction_split
from sspace.core.prompts.rendering import render_prompt_pair

from ..pairwise.registry import BenchmarkRegistry
from .config import DirectGenerationConfig
from .schema import DirectGenerationSample


def _prompt(style: str, group: str, query: str, reference: str) -> str:
    """Render only the original orientation of the frozen pairwise prompt."""
    return render_prompt_pair(style, group, query, reference)[0].text


def _coco_samples(
    config: DirectGenerationConfig,
) -> tuple[str, tuple[DirectGenerationSample, ...]]:
    """Load one validated COCO training or validation dataset."""
    manifest, source_samples = load_construction_split(
        config.dataset_source, config.split
    )
    benchmark = (
        "coco_construction_train6000"
        if config.split == "train"
        else "coco_validation1800"
    )
    samples = tuple(
        DirectGenerationSample(
            benchmark=benchmark,
            sample_id=sample.sample_id,
            split=sample.split,
            group=sample.group,
            query=sample.query,
            reference=sample.reference,
            gold_endpoint=sample.label,
            prompt=_prompt(
                config.prompt_style,
                sample.group,
                sample.query,
                sample.reference,
            ),
            image_path=sample.image_path,
            image_sha256=None,
            subset_tags=(),
            metadata={
                "dataset_id": str(manifest["dataset_id"]),
                "dataset_fingerprint": str(manifest["dataset_fingerprint"]),
                "prompt_protocol": "matched_pairwise_baseline",
            },
        )
        for sample in source_samples
    )
    return str(manifest["dataset_fingerprint"]), samples


def _benchmark_samples(
    config: DirectGenerationConfig,
) -> tuple[str, tuple[DirectGenerationSample, ...]]:
    """Convert one strict pairwise adapter to the shared baseline prompt."""
    source_samples = BenchmarkRegistry.load(
        config.dataset_adapter, config.dataset_source
    )
    samples = tuple(
        DirectGenerationSample(
            benchmark=sample.benchmark,
            sample_id=sample.sample_id,
            split=sample.split,
            group=sample.group,
            query=sample.query,
            reference=sample.reference,
            gold_endpoint=sample.gold_endpoint,
            prompt=_prompt(
                config.prompt_style,
                sample.group,
                sample.query,
                sample.reference,
            ),
            image_bytes=sample.image_bytes,
            image_sha256=sample.image_sha256,
            metadata={
                **sample.metadata,
                "source_direct_prompt": sample.direct_prompt,
                "prompt_protocol": "matched_pairwise_baseline",
            },
        )
        for sample in source_samples
    )
    source_identity = canonical_fingerprint(
        {
            "adapter": config.dataset_adapter,
            "sample_identity": [
                [
                    sample.sample_id,
                    sample.group,
                    sample.query,
                    sample.reference,
                    sample.gold_endpoint,
                    sample.image_sha256,
                ]
                for sample in samples
            ],
        }
    )
    return source_identity, samples


def direct_dataset_fingerprint(
    samples: tuple[DirectGenerationSample, ...], source_identity: str
) -> str:
    """Bind ordered identities, prompts, image identities, and source artifact."""
    return canonical_fingerprint(
        {
            "source_identity": source_identity,
            "samples": [
                [
                    sample.benchmark,
                    sample.sample_id,
                    sample.split,
                    list(sample.subset_tags),
                    sample.group,
                    sample.query,
                    sample.reference,
                    sample.gold_endpoint,
                    sample.prompt,
                    sample.image_sha256,
                ]
                for sample in samples
            ],
        }
    )


def load_direct_generation_samples(
    config: DirectGenerationConfig,
) -> tuple[tuple[DirectGenerationSample, ...], str]:
    """Load, validate, and fingerprint one complete EVAL-07 dataset.

    Args:
        config: Validated direct-generation configuration.

    Returns:
        Ordered samples and their task-level SHA-256 fingerprint.

    Raises:
        ValueError: Adapter output, sample count, or frozen fingerprint differs.
        FileNotFoundError: A declared source or image is absent.

    Side effects:
        Reads and validates the complete declared source dataset.
    """
    config.validate()
    if config.dataset_adapter == "coco_construction":
        source_identity, samples = _coco_samples(config)
    else:
        source_identity, samples = _benchmark_samples(config)
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("Direct-generation sample IDs must be unique")
    for sample in samples:
        sample.validate()
    fingerprint = direct_dataset_fingerprint(samples, source_identity)
    if len(samples) != config.expected_sample_count:
        raise ValueError(
            f"Direct dataset has {len(samples)} rows, expected "
            f"{config.expected_sample_count}"
        )
    if fingerprint != config.expected_dataset_fingerprint:
        raise ValueError(
            f"Direct dataset fingerprint {fingerprint} != "
            f"{config.expected_dataset_fingerprint}"
        )
    return samples, fingerprint
