"""Render frozen spatial prompts and locate exact object character spans."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
from string import Formatter

from .templates import AXIS_ORDER, PROMPT_STYLE_ORDER, PROMPT_TEMPLATES


@dataclass(frozen=True)
class RenderedPrompt:
    """One rendered prompt and its half-open object character spans.

    Attributes:
        text: Complete question sent to the model.
        query_span: Half-open ``[start, end)`` span of the query object.
        reference_span: Half-open span of the reference object.
    """

    text: str
    query_span: tuple[int, int]
    reference_span: tuple[int, int]


@dataclass(frozen=True)
class RenderedObjectPrompt:
    """One prompt with exact half-open spans for every named object.

    ``object_spans`` maps stable semantic roles to character spans. The roles
    are independent of mention order, which lets multi-object evaluations
    align the same entities across reordered prompts.
    """

    text: str
    object_spans: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        """Freeze and validate all declared object spans."""
        if not self.text.strip() or not self.object_spans:
            raise ValueError("Object prompt must contain text and at least one role")
        spans: dict[str, tuple[int, int]] = {}
        for role, span in self.object_spans.items():
            if not role.strip() or len(span) != 2:
                raise ValueError("Object prompt roles and spans must be explicit")
            start, end = (int(value) for value in span)
            if not 0 <= start < end <= len(self.text):
                raise ValueError(f"Object span for {role!r} is outside the prompt")
            spans[role] = (start, end)
        if len(set(spans.values())) != len(spans):
            raise ValueError("Object prompt spans must be distinct")
        object.__setattr__(self, "object_spans", MappingProxyType(spans))


@dataclass(frozen=True)
class RenderedMultiImagePrompt:
    """Interleave images with text and declare object spans in one text segment.

    ``text_segments[i]`` follows image ``i`` in the model-native user content.
    A non-object segment may be empty to place consecutive image inputs before
    the question. The object-bearing segment must remain non-empty.
    ``object_text_index`` identifies the only segment whose spans are read.
    This keeps benchmark language outside model adapters while preserving exact
    token alignment after multi-image chat-template expansion.
    """

    text_segments: tuple[str, ...]
    object_text_index: int
    object_spans: Mapping[str, tuple[int, int]]

    def __post_init__(self) -> None:
        """Freeze and validate image-aligned text and object spans."""
        segments = tuple(str(text) for text in self.text_segments)
        if len(segments) < 2:
            raise ValueError("Multi-image prompt requires at least two text segments")
        if not 0 <= self.object_text_index < len(segments):
            raise ValueError("object_text_index is outside the text segments")
        rendered = RenderedObjectPrompt(
            segments[self.object_text_index], self.object_spans
        )
        object.__setattr__(self, "text_segments", segments)
        object.__setattr__(self, "object_spans", rendered.object_spans)


def _validate_templates() -> None:
    if tuple(PROMPT_TEMPLATES) != PROMPT_STYLE_ORDER:
        raise ValueError("Prompt style order does not match the frozen protocol")
    for style, templates in PROMPT_TEMPLATES.items():
        if tuple(templates) != AXIS_ORDER:
            raise ValueError(f"Axis order differs for prompt style {style!r}")
        for axis, template in templates.items():
            fields = [field for _, field, _, _ in Formatter().parse(template) if field]
            if (
                fields.count("query") != 1
                or fields.count("reference") != 1
                or len(fields) != 2
            ):
                raise ValueError(f"Invalid object fields in {style}/{axis}")


def _render(template: str, query: str, reference: str) -> RenderedPrompt:
    values = {"query": query, "reference": reference}
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    length = 0
    for literal, field, format_spec, conversion in Formatter().parse(template):
        if format_spec or conversion:
            raise ValueError("Prompt object fields cannot use formatting or conversion")
        parts.append(literal)
        length += len(literal)
        if field is not None:
            value = values[field]
            spans[field] = (length, length + len(value))
            parts.append(value)
            length += len(value)
    return RenderedPrompt("".join(parts), spans["query"], spans["reference"])


def render_prompt_pair(
    style: str,
    axis: str,
    query: str,
    reference: str,
) -> tuple[RenderedPrompt, RenderedPrompt]:
    """Render original and object-swapped frozen prompts (TRN-01).

    Args:
        style: One of the five values in ``PROMPT_STYLE_ORDER``.
        axis: ``horizontal``, ``vertical``, or ``distance``.
        query: Name of the original query object.
        reference: Name of the original reference object.

    Returns:
        The original prompt followed by the prompt with object roles swapped.
        Both outputs contain exact half-open character spans for each role.

    Raises:
        KeyError: ``style`` or ``axis`` is not part of the frozen prompt set.
        ValueError: An object name is empty or the two names are identical.

    Side effects:
        None.

    Example:
        ``render_prompt_pair("direct", "horizontal", "apple", "cup")``
    """
    if not query.strip() or not reference.strip():
        raise ValueError("Object names must be non-empty")
    if query == reference:
        raise ValueError("Query and reference names must differ")
    template = PROMPT_TEMPLATES[style][axis]
    return _render(template, query, reference), _render(template, reference, query)


_validate_templates()
