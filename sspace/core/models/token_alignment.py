"""Map declared prompt spans to exact model token positions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..prompts.rendering import RenderedPrompt


def find_unique_subsequence(sequence: Sequence[int], subsequence: Sequence[int]) -> int:
    """Return the only start position of ``subsequence`` inside ``sequence``.

    Raises:
        ValueError: The question token sequence is absent or occurs more than once.

    Side effects:
        None.
    """
    matches = [
        index
        for index in range(len(sequence) - len(subsequence) + 1)
        if list(sequence[index : index + len(subsequence)]) == list(subsequence)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one question-token match, found {len(matches)}")
    return matches[0]


def object_token_positions(
    tokenizer: Any,
    full_ids: Sequence[int],
    prompt: RenderedPrompt,
) -> tuple[dict[str, int], dict[str, str]]:
    """Locate the last subtoken overlapping each declared object span.

    Args:
        tokenizer: Fast tokenizer supporting offset mappings.
        full_ids: Complete multimodal chat-template token IDs.
        prompt: Rendered question with half-open query/reference character spans.

    Returns:
        Exact full-sequence positions and decoded token pieces for query/reference.

    Raises:
        ValueError: The question is not unique or an object span has no token.

    Side effects:
        None.
    """
    return named_object_token_positions(
        tokenizer,
        full_ids,
        prompt.text,
        {"query": prompt.query_span, "reference": prompt.reference_span},
    )


def named_object_token_positions(
    tokenizer: Any,
    full_ids: Sequence[int],
    text: str,
    object_spans: Mapping[str, tuple[int, int]],
) -> tuple[dict[str, int], dict[str, str]]:
    """Locate the last subtoken for every named object span.

    Args:
        tokenizer: Fast tokenizer supporting offset mappings.
        full_ids: Complete multimodal chat-template token IDs.
        text: Exact question text embedded in the chat prompt.
        object_spans: Stable role to half-open character-span mapping.

    Returns:
        Full-sequence token positions and decoded token pieces keyed by role.

    Raises:
        ValueError: The question is not unique, roles are absent, or a span
            does not overlap exactly one or more text subtokens.

    Side effects:
        None.
    """
    if not text.strip() or not object_spans:
        raise ValueError("Named object alignment requires text and named roles")
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    question_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    question_start = find_unique_subsequence(full_ids, question_ids)
    positions: dict[str, int] = {}
    pieces: dict[str, str] = {}
    for role, (start, end) in object_spans.items():
        if not role.strip() or not 0 <= start < end <= len(text):
            raise ValueError(f"Invalid declared object span for {role!r}")
        overlapping = [
            index
            for index, (token_start, token_end) in enumerate(offsets)
            if token_start < end and token_end > start
        ]
        if not overlapping:
            raise ValueError(f"No token overlaps the declared {role} span")
        local_index = overlapping[-1]
        positions[role] = question_start + local_index
        pieces[role] = tokenizer.decode([question_ids[local_index]])
    return positions, pieces
