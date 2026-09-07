"""Build checksum-bound contextual-case JSONL from a compact research spec."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sspace.run_records import file_sha256


def occurrence_span(text: str, value: str, occurrence: int = 0) -> tuple[int, int]:
    """Return one explicit zero-based substring occurrence or fail closed."""
    if not value or occurrence < 0:
        raise ValueError("Object text and occurrence must be explicit")
    start = -1
    cursor = 0
    for _ in range(occurrence + 1):
        start = text.find(value, cursor)
        if start < 0:
            raise ValueError(f"Occurrence {occurrence} of {value!r} is absent")
        cursor = start + len(value)
    return start, start + len(value)


def image_record(image_root: Path, relative: str) -> dict[str, str]:
    """Bind a relative stimulus path to its current SHA-256 bytes."""
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Stimulus paths must stay below image_root")
    root = image_root.resolve()
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Resolved stimulus path must stay below image_root")
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": path.as_posix(), "sha256": file_sha256(resolved)}


def build_paired_prompts(group: dict, image_root: Path) -> list[dict]:
    """Expand a matched prompt sweep over explicitly named image conditions."""
    target = str(group["target"])
    reference = str(group["reference"])
    rows = []
    for prompt_index, prompt in enumerate(group["prompts"], 1):
        spans = {
            "target": occurrence_span(
                prompt, target, int(group.get("target_occurrence", 0))
            ),
            "reference": occurrence_span(
                prompt, reference, int(group.get("reference_occurrence", 0))
            ),
        }
        for condition, image in group["condition_images"].items():
            rows.append(
                {
                    "case_id": f"{group['family']}_{prompt_index:02d}_{condition}",
                    "theme": "beyond_visible",
                    "family": group["family"],
                    "condition": condition,
                    "images": [image_record(image_root, image)],
                    "prompt": prompt,
                    "object_spans": {key: list(value) for key, value in spans.items()},
                    "metadata": {
                        "prompt_key": f"{group['family']}_{prompt_index:02d}",
                        "effect_sign": int(group["effect_sign"]),
                        "effect_axis": str(group.get("effect_axis", "horizontal")),
                        "effect_condition_a": str(
                            group.get("effect_condition_a", "original")
                        ),
                        "effect_condition_b": str(
                            group.get("effect_condition_b", "mirror")
                        ),
                        "effect_conditions_a": list(
                            group.get("effect_conditions_a", [])
                        ),
                        "effect_conditions_b": list(
                            group.get("effect_conditions_b", [])
                        ),
                        "condition_semantics": str(
                            group.get("condition_semantics", {}).get(condition, "")
                        ),
                    },
                }
            )
    return rows


def build_web_showcase(group: dict, image_root: Path) -> list[dict]:
    """Read every highlighted web token in the identical image/prompt context."""
    prompt = group["prompt"]
    return [
        {
            "case_id": f"{group['family']}_token_{index}",
            "theme": "beyond_visible",
            "family": group["family"],
            "condition": "web_showcase",
            "images": [image_record(image_root, group["image"])],
            "prompt": prompt,
            "object_spans": {"target": list(occurrence_span(prompt, token))},
            "metadata": {"token_text": token, "effect_axis": group["axis"]},
        }
        for index, token in enumerate(group["tokens"])
    ]


def build_political_templates(group: dict, image_root: Path) -> list[dict]:
    """Expand matched political words over scenes and counterbalanced templates."""
    rows = []
    reference = str(group["reference"])
    for condition, image in group["condition_images"].items():
        for word in group["target_words"]:
            for template_index, template in enumerate(group["templates"]):
                prompt = template.format(target=word, reference=reference)
                spans = {
                    "target": occurrence_span(prompt, word),
                    "reference": occurrence_span(prompt, reference),
                }
                rows.append(
                    {
                        "case_id": f"political_{word}_{condition}_t{template_index}",
                        "theme": "language_supervision",
                        "family": "political_semantics",
                        "condition": condition,
                        "images": [image_record(image_root, image)],
                        "prompt": prompt,
                        "object_spans": {
                            key: list(value) for key, value in spans.items()
                        },
                        "metadata": {
                            "target_word": word,
                            "template_index": template_index,
                            "scene_condition": condition,
                        },
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument(
        "--family",
        action="append",
        help="Optional exact family filter; repeat to select multiple spec groups.",
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = []
    selected = set(args.family or ())
    available = {
        str(group.get("family", "political_semantics"))
        for group in spec["groups"]
    }
    if selected - available:
        raise ValueError(
            f"Unknown requested families: {sorted(selected - available)}"
        )
    for group in spec["groups"]:
        if selected and group.get("family", "political_semantics") not in selected:
            continue
        if group["mode"] == "paired_prompts":
            rows.extend(build_paired_prompts(group, args.image_root))
        elif group["mode"] == "political_templates":
            rows.extend(build_political_templates(group, args.image_root))
        elif group["mode"] == "web_showcase":
            rows.extend(build_web_showcase(group, args.image_root))
        else:
            raise ValueError(f"Unknown case build mode {group['mode']!r}")
    if not rows or len({row["case_id"] for row in rows}) != len(rows):
        raise ValueError("Built cases must be non-empty and unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
