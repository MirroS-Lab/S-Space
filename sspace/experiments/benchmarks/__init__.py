"""Strict benchmark-specific parsers for the formal evaluation API."""

from .cvbench.pairwise import load_cvbench_pairwise
from .embspatial.adapter import load_embspatial_balanced1200
from .spatialtunnel.adapter import load_spatialtunnel

__all__ = [
    "load_cvbench_pairwise",
    "load_embspatial_balanced1200",
    "load_spatialtunnel",
]
