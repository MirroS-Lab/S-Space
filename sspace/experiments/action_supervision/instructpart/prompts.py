"""Freeze and render the ten InstructPart target-localization templates."""

from __future__ import annotations

from dataclasses import dataclass

from sspace.core.prompts.rendering import RenderedObjectPrompt


PROMPT_PROTOCOL = "instructpart_named_part_prompt_ensemble_10"
TARGET_ROLE = "target"
TARGET_ORDER = ("object", "part")

ACTION_TEMPLATES = {
    "assist": "stand up with assistance from the {object}",
    "close": "close the {object}",
    "contain": "contain items using the {object}",
    "control": "control the {object}",
    "cook": "cook food using the {object}",
    "cover": "cover the {object}",
    "cursor control": "control the cursor using the {object}",
    "cut": "cut something using the {object}",
    "dig": "dig using the {object}",
    "display": "view visual output on the {object}",
    "flush": "flush the {object}",
    "grasp": "grasp the {object}",
    "grip": "use the {object} for gripping",
    "hold": "hold the {object}",
    "light up": "light up the room using the {object}",
    "move": "move the {object}",
    "open": "open the {object}",
    "open/close": "open or close the {object}",
    "pick up": "pick up the {object}",
    "pierce": "pierce food using the {object}",
    "pour": "pour using the {object}",
    "protect": "protect something using the {object}",
    "put": "put something on or in the {object}",
    "select channels": "select channels on the {object}",
    "set": "adjust the settings on the {object}",
    "sit": "sit on the {object}",
    "store": "store items in the {object}",
    "support": "use the {object} for support",
    "swing": "make the {object} swing",
    "turn off": "turn off the {object}",
    "turn on": "turn on the {object}",
    "type": "type using the {object}",
}


@dataclass(frozen=True)
class PromptTemplate:
    """Declare one prompt surface form and its evaluation group."""

    template_id: str
    group: str
    text: str


PROMPT_TEMPLATES = (
    PromptTemplate("T01", "context_free", "Point to {target}."),
    PromptTemplate("T02", "context_free", "Find the {target}."),
    PromptTemplate("T03", "context_free", "Locate the {target}."),
    PromptTemplate(
        "T04",
        "context_free",
        "Look for {target} in the image and show me where they are.",
    ),
    PromptTemplate(
        "T05",
        "context_free",
        "Please find {target} and show me where they are.",
    ),
    PromptTemplate(
        "T06",
        "action_context",
        'Given the instruction "{instruction}", locate the {target}.',
    ),
    PromptTemplate(
        "T07",
        "action_context",
        "The robot needs to {instruction}. Find the {target}.",
    ),
    PromptTemplate(
        "T08",
        "action_context",
        "The robot has been asked to {instruction}. Point to the {target}.",
    ),
    PromptTemplate(
        "T09",
        "action_context",
        "The robot's task is to {instruction}. Help me find the {target}.",
    ),
    PromptTemplate(
        "T10",
        "action_context",
        'To carry out the instruction "{instruction}", find the {target}.',
    ),
)


def render_action(action: str, object_name: str) -> str:
    """Render the historical action phrase from dataset-provided fields."""
    try:
        return ACTION_TEMPLATES[action].format(object=object_name)
    except KeyError as error:
        raise ValueError(f"Unsupported InstructPart action: {action!r}") from error


def render_template_prompt(
    template: PromptTemplate,
    instruction: str,
    target: str,
) -> RenderedObjectPrompt:
    """Render one prompt and declare the exact measured target span."""
    if not target.strip():
        raise ValueError("Prompt target must be non-empty")
    before, after = template.text.split("{target}")
    before = before.format(instruction=instruction)
    after = after.format(instruction=instruction)
    text = before + target + after
    return RenderedObjectPrompt(
        text,
        {TARGET_ROLE: (len(before), len(before) + len(target))},
    )


def render_prompt_ensemble(
    action: str,
    object_name: str,
    part_name: str,
) -> tuple[tuple[RenderedObjectPrompt, RenderedObjectPrompt], ...]:
    """Render separate object/part forwards in fixed T01--T10 order."""
    instruction = render_action(action, object_name)
    return tuple(
        (
            render_template_prompt(template, instruction, object_name),
            render_template_prompt(template, instruction, part_name),
        )
        for template in PROMPT_TEMPLATES
    )


def _validate_templates() -> None:
    expected_ids = tuple(f"T{index:02d}" for index in range(1, 11))
    if tuple(template.template_id for template in PROMPT_TEMPLATES) != expected_ids:
        raise ValueError("Prompt template IDs must be T01--T10")
    if tuple(template.group for template in PROMPT_TEMPLATES) != (
        ("context_free",) * 5 + ("action_context",) * 5
    ):
        raise ValueError("Prompt ensemble must contain five templates per group")
    for template in PROMPT_TEMPLATES:
        if template.text.count("{target}") != 1:
            raise ValueError(f"{template.template_id} must contain one target")
        required_instruction = int(template.group == "action_context")
        if template.text.count("{instruction}") != required_instruction:
            raise ValueError(f"{template.template_id} instruction field differs")


_validate_templates()
