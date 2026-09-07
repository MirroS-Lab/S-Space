"""Register formal pairwise benchmark adapters explicitly."""

from __future__ import annotations

from pathlib import Path

from sspace.experiments.benchmarks.cvbench.pairwise import load_cvbench_pairwise
from sspace.experiments.benchmarks.embspatial.adapter import (
    load_embspatial_balanced1200,
)
from sspace.experiments.benchmarks.spatialtunnel.adapter import (
    load_spatialtunnel,
)


class BenchmarkRegistry:
    """Construct named adapters without implicit aliases or fallback paths."""

    _LOADERS = {
        "embspatial_balanced1200": load_embspatial_balanced1200,
        "spatialtunnel": load_spatialtunnel,
        "cvbench_pairwise": load_cvbench_pairwise,
    }

    @classmethod
    def load(cls, name: str, source: Path):
        """Load one declared benchmark source through its strict adapter.

        Args:
            name: Exact key returned by :meth:`names`.
            source: Benchmark parquet file or snapshot directory.

        Returns:
            An ordered tuple of validated pairwise samples.

        Raises:
            KeyError: ``name`` is not registered.
            FileNotFoundError: ``source`` is absent.
            ValueError: The pinned benchmark schema or row count differs.

        Side effects:
            Reads the benchmark source without changing it.
        """
        if name not in cls._LOADERS:
            raise KeyError(f"Unknown benchmark {name!r}; expected {cls.names()}")
        if not source.exists():
            raise FileNotFoundError(source)
        return cls._LOADERS[name](source)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        """Return registered names in stable public order."""
        return tuple(cls._LOADERS)
