"""Exact generated-token matching for MMSI direction words."""

from __future__ import annotations

from functools import cache
from typing import Any

PROBE_WORDS = (
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "left",
    "right",
    "above",
    "below",
    "top",
    "bottom",
    "upper",
    "lower",
)
PROBE_VARIANTS = {
    "northeast": ("northeast", "north-east", "north east"),
    "northwest": ("northwest", "north-west", "north west"),
    "southeast": ("southeast", "south-east", "south east"),
    "southwest": ("southwest", "south-west", "south west"),
}


def find_direction_occurrences(
    tokenizer: Any,
    generated: list[int],
    *,
    probe_words: tuple[str, ...] = PROBE_WORDS,
) -> list[dict[str, Any]]:
    """Find exact direction-token spans without double-counting compounds.

    Token IDs are matched directly, then checked for whole-word boundaries.
    Inflections such as ``northwestern`` are not standalone direction words.
    Compound spellings such as ``northwest``, ``north-west`` and ``north west`` are
    canonicalized. A compound span suppresses nested base-word matches, so a
    split-token ``north west`` contributes one ``northwest`` occurrence rather
    than three observations.
    """

    found: dict[tuple[int, int, str], dict[str, Any]] = {}

    @cache
    def prefix(end: int) -> str:
        return tokenizer.decode(generated[:end])

    full_text = prefix(len(generated))

    def word_character(char: str) -> bool:
        return char.isalnum() or char == "_"

    for word in probe_words:
        variants: set[tuple[int, ...]] = set()
        spellings = PROBE_VARIANTS.get(word, (word,))
        for spelling in spellings:
            for text in (
                spelling,
                " " + spelling,
                spelling.capitalize(),
                " " + spelling.capitalize(),
            ):
                ids = tuple(
                    int(value)
                    for value in tokenizer(text, add_special_tokens=False)["input_ids"]
                )
                if ids:
                    variants.add(ids)
        for needle in variants:
            width = len(needle)
            for start in range(len(generated) - width + 1):
                if tuple(generated[start : start + width]) != needle:
                    continue
                before = prefix(start)
                through = prefix(start + width)
                matched = through[len(before):]
                char_start = len(before) + len(matched) - len(matched.lstrip())
                char_end = len(through.rstrip())
                if (
                    char_start > 0 and word_character(full_text[char_start - 1])
                    or char_end < len(full_text) and word_character(full_text[char_end])
                ):
                    continue
                key = (start, start + width, word)
                left = max(0, start - 24)
                right = min(len(generated), start + width + 24)
                found[key] = {
                    "word": word,
                    "generated_start": start,
                    "generated_end": start + width,
                    "token_ids": list(needle),
                    "token_pieces": [tokenizer.decode([value]) for value in needle],
                    "context": tokenizer.decode(
                        generated[left:right], skip_special_tokens=True
                    ),
                }
    ordered = sorted(
        found.values(),
        key=lambda item: (
            int(item["generated_start"]),
            -(int(item["generated_end"]) - int(item["generated_start"])),
            -len(str(item["word"])),
        ),
    )
    accepted: list[dict[str, Any]] = []
    for candidate in ordered:
        start = int(candidate["generated_start"])
        end = int(candidate["generated_end"])
        if any(
            int(existing["generated_start"]) <= start
            and end <= int(existing["generated_end"])
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return sorted(
        accepted, key=lambda item: (item["generated_start"], item["generated_end"])
    )
