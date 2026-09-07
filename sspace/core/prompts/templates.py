"""Define frozen prompt templates for COCO S-Space axis extraction.

The module contains data only. Prompt rendering and object-span calculation
belong to :mod:`sspace.core.prompts.rendering`.
"""

from __future__ import annotations


PROMPT_SET_ID = "coco_spatial_relations_v2"
AXIS_ORDER = ("horizontal", "vertical", "distance")
PROMPT_STYLE_ORDER = ("baseline", "direct", "natural", "choice_first", "terse")

# TRN-01: These templates were frozen after development on a prompt-dev split
# and one evaluation on a disjoint 120-image COCO validation split. Braced
# object fields occur exactly once so token roles can be aligned unambiguously.
PROMPT_TEMPLATES = {
    "baseline": {
        "horizontal": "Is the {query} to the left or right of the {reference}? Answer with left or right.",
        "vertical": "Is the {query} above or below the {reference}? Answer with above or below.",
        "distance": "Is the {query} farther from or closer to the camera than the {reference}? Answer with far or close.",
    },
    "direct": {
        "horizontal": "Where is the {query} relative to the {reference}: left or right? Answer with left or right.",
        "vertical": "Where is the {query} relative to the {reference}: above or below? Answer with above or below.",
        "distance": "Considering only depth, is the {query} farther from or closer to the camera than the {reference}? Answer far or close.",
    },
    "natural": {
        "horizontal": "In the image, is the {query} to the left or right of the {reference}? Answer with left or right.",
        "vertical": "In the image, is the {query} above or below the {reference}? Answer with above or below.",
        "distance": "In the image, is the {query} farther from or closer to the camera than the {reference}? Answer with far or close.",
    },
    "choice_first": {
        "horizontal": "Which describes the position of the {query} relative to the {reference}: left or right? Answer with left or right.",
        "vertical": "Which describes the position of the {query} relative to the {reference}: above or below? Answer with above or below.",
        "distance": "Choose far or close: is the {query} farther from or closer to the camera than the {reference}? Answer with far or close.",
    },
    "terse": {
        "horizontal": "Relative to the {reference}, is the {query} on the left or right? Answer with left or right.",
        "vertical": "Relative to the {reference}, is the {query} above or below? Answer with above or below.",
        "distance": "Is the {query} farther from the camera than the {reference}, or closer? Answer far or close.",
    },
}
