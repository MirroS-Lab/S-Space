"""Render the frozen label-conditioned SpinBench activation intervention."""

from __future__ import annotations

from sspace.experiments.spinbench.perspective_taking.adapter import (
    SpinBenchSample,
)


TEMPLATE_ID = "native_exact_winner"

def _relation_words(sample: SpinBenchSample) -> tuple[str, str, str, str]:
    """Return the ordered objects and source/target relation phrases.

    This intervention intentionally reads the official answer to instantiate
    a coherent reasoning path. It is an activation intervention, not an
    independent accuracy evaluation.
    """
    winner = sample.object_a if sample.answer == "A" else sample.object_b
    loser = sample.object_b if sample.answer == "A" else sample.object_a
    source_relation = {
        ("distance", "back", "closer"): "farther from the front viewer than",
        ("distance", "left", "left"): "farther from the front viewer than",
        ("distance", "right", "left"): "closer to the front viewer than",
        ("horizontal", "back", "left"): "to the right of",
        ("horizontal", "left", "closer"): "to the left of",
        ("horizontal", "right", "closer"): "to the right of",
    }.get((sample.source_group, sample.target_view, sample.target_property))
    if source_relation is None:
        raise ValueError(f"Unsupported SpinBench transform {sample.transform!r}")
    target_relation = (
        "to the left of"
        if sample.target_property == "left"
        else "closer to the viewer than"
    )
    return winner, loser, source_relation, target_relation


def render_prefilled_cot(sample: SpinBenchSample) -> str:
    """Instantiate three paired object mentions across reasoning stages.

    Args:
        sample: One validated SpinBench perspective-taking sample.

    Returns:
        The frozen label-conditioned reasoning prefill. Each object appears
        once in the observation, viewpoint-integration, and completion stage.

    Raises:
        ValueError: The sample declares an unsupported transform.

    Side effects:
        None.
    """
    winner, loser, source_relation, target_relation = _relation_words(sample)
    question = (
        "appear on the left" if sample.target_property == "left"
        else "be closer to the viewer"
    )
    return f"""The user wants me to determine which object would {question} if the scene were viewed from the {sample.target_view}.

1. **Analyze the current view:**
* In the front view, the {winner} is {source_relation} the {loser}.

2. **Imagine the new perspective:**
* I now imagine viewing the {winner} and the {loser} from the {sample.target_view}.

3. **Conclusion:**
* From this perspective, the {winner} is {target_relation} the {loser}."""
