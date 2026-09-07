"""SpinBench fixed-template Evolving-CoT experiments."""

from .template import render_prefilled_cot

STAGE_SCORE_PROTOCOL = "natural_prefilled_cot_three_stage_rotation_v1"

__all__ = ["render_prefilled_cot", "STAGE_SCORE_PROTOCOL"]
