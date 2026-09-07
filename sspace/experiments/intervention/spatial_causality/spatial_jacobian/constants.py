"""Experiment constants frozen for the 2026-07-21 reproduction."""

from __future__ import annotations

from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[5]
DEFAULT_HF_HOME = PROJECT_DIR / ".cache" / "huggingface"
DEFAULT_MODEL_PATH = PROJECT_DIR / ".cache" / "assets" / "models" / "molmo2_er"
DEFAULT_ANNOTATION_PATH = (
    PROJECT_DIR
    / ".cache"
    / "assets"
    / "datasets"
    / "embspatial_annotations"
    / "data"
    / "test-00000-of-00001.parquet"
)

MODEL_ID = "allenai/Molmo2-ER"
MODEL_REVISION = "dab22564403d2607855bb1fffb0721285b445081"

CATEGORY_ORDER = ["left", "right", "above", "below", "far", "close"]
GROUP_ORDER = ["horizontal", "vertical", "distance"]
GROUP_MAP = {
    "left": "horizontal",
    "right": "horizontal",
    "above": "vertical",
    "below": "vertical",
    "far": "distance",
    "close": "distance",
}
OPPOSITE = {
    "left": "right",
    "right": "left",
    "above": "below",
    "below": "above",
    "far": "close",
    "close": "far",
}
CANONICAL = {"horizontal": "left", "vertical": "above", "distance": "far"}
POSITIVE_ANSWER = {"horizontal": "right", "vertical": "below", "distance": "close"}
NEGATIVE_ANSWER = {"horizontal": "left", "vertical": "above", "distance": "far"}

TEMPLATES = {
    "baseline": {
        "horizontal": "Is the {obj1} to the left or right of the {obj2}? Answer with only one word.",
        "vertical": "Is the {obj1} above or below the {obj2}? Answer with only one word.",
        "distance": "Compared to {obj2}, is {obj1} far or close from you? Answer with only one word.",
    },
    "direct": {
        "horizontal": "Where is the {obj1} relative to the {obj2}: left or right? Reply with one word.",
        "vertical": "Where is the {obj1} relative to the {obj2}: above or below? Reply with one word.",
        "distance": "Is {obj1} far from or close to you compared with {obj2}? Reply with one word.",
    },
    "natural": {
        "horizontal": "Looking at the image, would you say the {obj1} is left or right of the {obj2}? Use one word.",
        "vertical": "Looking at the image, would you say the {obj1} is above or below the {obj2}? Use one word.",
        "distance": "Looking at the image, compared with {obj2}, does {obj1} appear far or close to you? Use one word.",
    },
    "choice_first": {
        "horizontal": "Choose left or right: the {obj1} is on which side of the {obj2}? Answer with the chosen word.",
        "vertical": "Choose above or below: where is the {obj1} relative to the {obj2}? Answer with the chosen word.",
        "distance": "Choose far or close: relative to {obj2}, how far from you is {obj1}? Answer with the chosen word.",
    },
    "terse": {
        "horizontal": "{obj1} relative to {obj2} -- left or right? One word only.",
        "vertical": "{obj1} relative to {obj2} -- above or below? One word only.",
        "distance": "{obj1} versus {obj2} -- far or close from you? One word only.",
    },
}
STYLE_ORDER = list(TEMPLATES)

DATASET_REVISIONS = {
    "embspatial": {
        "repo_id": "ch-min/EmbSpatial-Bench-tsv",
        "revision": "c953faef1693576727fe6af1910e7c92082b246c",
        "filename": "EmbSpatial-Bench.tsv",
        "sha256": "21e87cea7d30dd769674694e14e51a7ca07af1dd0b3da4d5cc57eca08d9a515e",
    },
    "spatialtunnel": {
        "repo_id": "cubec/spatialtunnel",
        "revision": "96f48f3e6ab8738e87a151e7c8fc29954fa10fd6",
        "filename": "contrastive_probing.tsv",
        "sha256": "f08c3b8d126f6d2fcfae751eb682ff56d510091209040e03e78a37edc5eecfcc",
    },
    "embspatial_annotations": {
        "repo_id": "FlagEval/EmbSpatial-Bench",
        "revision": "3c0e6b34b632de666a51091727c0128d07a54a6a",
        "filename": "data/test-00000-of-00001.parquet",
        "sha256": "a439675722fca13073e75965b9045eb949c30712c131832e41e97000caf3f88a",
    },
}

EXPECTED_DATASET_FINGERPRINTS = {
    "embspatial": "99017d3c308520b73d5bf48498e50e6d8f1661c4b2c9c42e8d554b114f8a2846",
    "spatialtunnel": "d832d413c9041c310fbcc0ae0d4de9c935261a494d62d64f11fc35615e6267d1",
}
HISTORICAL_PREDICTION_SHA256 = (
    "9e8fa1171da7790eb273e191170d425ee09ba8b1ceca1cdc3f00649ad938a6da"
)
