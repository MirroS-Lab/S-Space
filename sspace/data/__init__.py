"""Build and validate reproducible S-Space datasets.

GPU preprocessing modules are imported lazily so read-only evaluation does not
require COCO mask-decoding or depth-inference dependencies.
"""

__all__ = [
    "CocoTrainingConfig",
    "CocoValidationConfig",
    "acquire_validation_images",
    "load_coco_training_config",
    "load_coco_validation_config",
    "prepare_coco_assets",
    "prepare_coco_training",
    "prepare_coco_validation",
    "required_validation_images",
    "selected_image_fingerprint",
    "validate_coco_validation_dataset",
    "validate_preprocessed_coco",
]


def __getattr__(name: str):
    """Load only the requested dependency boundary."""
    if name in {
        "CocoTrainingConfig",
        "CocoValidationConfig",
        "load_coco_training_config",
        "load_coco_validation_config",
    }:
        from .coco.config import (
            CocoTrainingConfig,
            CocoValidationConfig,
            load_coco_training_config,
            load_coco_validation_config,
        )

        return {
            "CocoTrainingConfig": CocoTrainingConfig,
            "CocoValidationConfig": CocoValidationConfig,
            "load_coco_training_config": load_coco_training_config,
            "load_coco_validation_config": load_coco_validation_config,
        }[name]
    if name == "prepare_coco_training":
        from .coco.pipeline import prepare_coco_training

        return prepare_coco_training
    if name == "prepare_coco_assets":
        from .coco.assets import prepare_coco_assets

        return prepare_coco_assets
    if name == "validate_preprocessed_coco":
        from .coco.validation import validate_preprocessed_coco

        return validate_preprocessed_coco
    if name == "selected_image_fingerprint":
        from .coco.image_integrity import selected_image_fingerprint

        return selected_image_fingerprint
    if name in {"acquire_validation_images", "required_validation_images"}:
        from .coco.acquisition import (
            acquire_validation_images,
            required_validation_images,
        )

        return {
            "acquire_validation_images": acquire_validation_images,
            "required_validation_images": required_validation_images,
        }[name]
    if name in {"prepare_coco_validation", "validate_coco_validation_dataset"}:
        from .coco.validation_dataset import (
            prepare_coco_validation,
            validate_coco_validation_dataset,
        )

        return {
            "prepare_coco_validation": prepare_coco_validation,
            "validate_coco_validation_dataset": validate_coco_validation_dataset,
        }[name]
    raise AttributeError(name)
