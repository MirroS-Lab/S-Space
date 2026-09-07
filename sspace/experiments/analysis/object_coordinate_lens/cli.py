"""Run Object Coordinate Lens once without the removed HTTP/Web UI layer."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from PIL import Image

from .inference import OnlineInferencePipeline, normalise_upload_image, worker_python_path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
WORKER_ENVIRONMENTS = {
    "molmo2_er": "spatialmqa",
    "molmoact2": "molmoact",
    "molmoact2_pretrain": "molmoact",
    "qwen35_4b": "qwen36",
    "qwen36_27b": "qwen36",
}


def _worker_python(model_id: str, override: Path | None) -> Path:
    if override is not None:
        path = worker_python_path(override)
    else:
        if model_id not in WORKER_ENVIRONMENTS:
            raise ValueError(f"Unknown Object Lens model {model_id!r}")
        path = PROJECT_ROOT / ".envs" / WORKER_ENVIRONMENTS[model_id] / "bin/python"
    if not path.is_file():
        raise FileNotFoundError(f"Object Lens worker Python is missing: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="molmo2_er")
    parser.add_argument("--layer", type=int)
    parser.add_argument(
        "--mention",
        action="append",
        dest="mentions",
        help="Exact prompt substring; repeat for 2-8 manually selected objects.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--worker-python", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    with Image.open(args.image) as opened:
        image = normalise_upload_image(opened)
    pipeline = OnlineInferencePipeline(
        cache_dir=args.cache_dir,
        device=args.device,
        worker_python=_worker_python(args.model, args.worker_python),
        runtime_root=args.runtime_root,
    )
    try:
        run_id = uuid.uuid4().hex
        result = pipeline.infer(
            image,
            args.prompt,
            run_id,
            ".",
            manual_mentions=args.mentions,
            model_id=args.model,
            layer=args.layer,
        )
    finally:
        pipeline.close()
    mask_dir = args.output_dir / "masks"
    mask_dir.mkdir(parents=True)
    image.save(args.output_dir / "image.png", format="PNG")
    for object_id, mask in result.masks.items():
        mask.save(mask_dir / f"{object_id}.png", format="PNG")
    (args.output_dir / "result.json").write_text(
        json.dumps(result.payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
