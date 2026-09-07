"""Teacher-force a fixed SpinBench CoT and capture object-token projections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from sspace.determinism import configure_deterministic_cuda
from sspace.core.artifacts import SSpaceArtifact
from sspace.experiments.spinbench.evolving_cot.template import (
    render_prefilled_cot,
)
from sspace.experiments.spinbench.perspective_taking.adapter import (
    load_spinbench,
)
from sspace.experiments.spinbench.evolving_cot.template import TEMPLATE_ID
from sspace.run_records import atomic_json
from sspace.core.extraction.execution import move_model_inputs
from sspace.experiments.spinbench.evolving_cot.runtime import (
    _append_generated,
    _mentions,
    _load_runtime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[5]
IMAGE_ROOT = PROJECT_ROOT / ".cache/assets/datasets/spinbench"
ANNOTATION_PATH = IMAGE_ROOT / "test.jsonl"
ARTIFACT_DIR = PROJECT_ROOT / (
    "sspace/core/artifacts/pretrained/qwen36_27b_coco6000_final_logit_axes_v2"
)
OUTPUT_ROOT = PROJECT_ROOT / "outputs/reproduction/cot_evolution/activations"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _StateCapture:
    """Capture selected positions from one post-block layer."""

    def __init__(self, block: torch.nn.Module, positions: Sequence[int]):
        self.block = block
        self.positions = tuple(int(value) for value in positions)
        self.handle: Any = None
        self.value: torch.Tensor | None = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self.value = hidden[0, self.positions, :].detach().float().cpu()

    def __enter__(self) -> "_StateCapture":
        self.handle = self.block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.handle.remove()

    def result(self) -> torch.Tensor:
        if self.value is None or not torch.isfinite(self.value).all():
            raise RuntimeError("Selected-layer object states were not captured")
        return self.value


def _inputs(processor: Any, image: Image.Image, prompt: str) -> dict[str, torch.Tensor]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_tensors="pt",
        return_dict=True,
    )


def _atomic_tensor(path: Path, values: dict[str, torch.Tensor]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(values, temporary)
    os.replace(temporary, path)


def main() -> None:
    configure_deterministic_cuda()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

    samples = load_spinbench(ANNOTATION_PATH, IMAGE_ROOT, "without")
    selected = samples[:6] if args.mode == "smoke" else samples
    artifact = SSpaceArtifact.load(ARTIFACT_DIR)
    layer = artifact.manifest.model.selected_layer
    if layer != 43:
        raise ValueError(f"Expected Qwen3.6 COCO-1800 readout L43, found L{layer}")
    axes = torch.from_numpy(artifact.axes(layer)).float()
    output_dir = OUTPUT_ROOT / args.mode
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "tensors").mkdir()
    runtime = _load_runtime("sdpa")
    records_path = output_dir / "records.jsonl"
    with records_path.open("x", encoding="utf-8") as stream:
        for index, sample in enumerate(selected):
            rationale = render_prefilled_cot(sample)
            rationale_ids = runtime.processor.tokenizer(
                rationale, add_special_tokens=False
            )["input_ids"]
            closing_ids = runtime.processor.tokenizer(
                "\n</think>\n", add_special_tokens=False
            )["input_ids"]
            mentions = _mentions(
                rationale_ids,
                runtime.processor.tokenizer,
                {"object_a": sample.object_a, "object_b": sample.object_b},
            )
            if {value["role"] for value in mentions} != {"object_a", "object_b"}:
                raise ValueError(
                    f"Both objects are not token-aligned in {sample.sample_id}"
                )
            role_counts = {
                role: sum(value["role"] == role for value in mentions)
                for role in ("object_a", "object_b")
            }
            if role_counts != {"object_a": 3, "object_b": 3}:
                raise ValueError(
                    f"Expected three paired mentions per object in {sample.sample_id}: "
                    f"{role_counts}"
                )
            if any(
                line and not line[0].isdigit() and not line.startswith("*")
                for line in rationale.splitlines()[1:]
                if line.strip()
            ):
                raise ValueError("Every non-heading CoT step must start with '*'")
            last_positions = sorted(
                {value["generated_token_end"] - 1 for value in mentions}
            )
            position_to_column = {
                position: column for column, position in enumerate(last_positions)
            }
            image = Image.open(sample.image_path)
            image.load()
            inputs = _inputs(
                runtime.processor, image.convert("RGB"), sample.direct_prompt
            )
            prompt_length = int(inputs["input_ids"].shape[1])
            _append_generated(inputs, [*rationale_ids, *closing_ids])
            full_positions = [prompt_length + value for value in last_positions]
            with _StateCapture(runtime.blocks[layer], full_positions) as capture:
                with torch.inference_mode():
                    runtime.model(
                        **move_model_inputs(inputs, runtime.model),
                        use_cache=False,
                        return_dict=True,
                        logits_to_keep=1,
                    )
            states = capture.result()
            coordinates = states @ axes.T
            model_inputs = move_model_inputs(inputs, runtime.model)
            input_length = int(model_inputs["input_ids"].shape[1])
            with torch.inference_mode():
                generated = runtime.model.generate(
                    **model_inputs,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=1,
                )
            answer_ids = [int(value) for value in generated[0, input_length:].tolist()]
            if len(answer_ids) != 1:
                raise ValueError("Answer mapping must generate exactly one token")
            answer_text = (
                runtime.processor.tokenizer.decode(answer_ids, skip_special_tokens=True)
                .strip()
                .upper()
            )
            predicted_answer = answer_text if answer_text in {"A", "B"} else None
            for mention in mentions:
                position = mention["generated_token_end"] - 1
                mention["last_subtoken_position"] = position
                mention["activation_column"] = position_to_column[position]
                mention["projection_hvd"] = coordinates[
                    position_to_column[position]
                ].tolist()
            tensor_file = f"tensors/{sample.sample_id}.safetensors"
            _atomic_tensor(
                output_dir / tensor_file,
                {
                    "post_block_states": states,
                    "projection_hvd": coordinates,
                    "rationale_last_subtoken_positions": torch.tensor(last_positions),
                    "full_sequence_positions": torch.tensor(full_positions),
                },
            )
            stream.write(
                json.dumps(
                    {
                        "sample_index": index,
                        "sample_id": sample.sample_id,
                        "object_a": sample.object_a,
                        "object_b": sample.object_b,
                        "gold_answer": sample.answer,
                        "target_view": sample.target_view,
                        "target_property": sample.target_property,
                        "selected_layer": layer,
                        "decision_protocol": "label_conditioned_prefill_activation_intervention_v1",
                        "system_prompt": None,
                        "generation_performed": True,
                        "answer_token_id": answer_ids[0],
                        "answer_token_text": answer_text,
                        "predicted_answer": predicted_answer,
                        "answer_mapping_correct": predicted_answer == sample.answer,
                        "rationale": rationale,
                        "mentions": mentions,
                        "tensor_file": tensor_file,
                        "tensor_sha256": _sha256(output_dir / tensor_file),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            print(f"prefilled CoT activations {index + 1}/{len(selected)}", flush=True)
    records = [json.loads(line) for line in records_path.read_text().splitlines()]
    mapping_correct = sum(row["answer_mapping_correct"] for row in records)
    parse_failures = sum(row["predicted_answer"] is None for row in records)
    summary = {
        "protocol": "qwen36_spinbench_natural_prefilled_cot_v1",
        "template_id": TEMPLATE_ID,
        "sample_count": len(selected),
        "selected_layer": layer,
        "system_prompt": None,
        "generation_performed": True,
        "generation_protocol": "greedy_one_answer_token_v1",
        "forward_semantics": "causal teacher forcing over a code-instantiated assistant CoT followed by one generated token",
        "template_uses_official_answer": True,
        "valid_for_benchmark_accuracy_evaluation": False,
        "answer_mapping": {
            "correct": mapping_correct,
            "total": len(records),
            "accuracy": mapping_correct / len(records),
            "parse_failures": parse_failures,
        },
        "object_state": "last subtoken post-block state for every exact object mention",
        "mention_stages": ["observation", "viewpoint_integration", "completion"],
        "mentions_per_object": 3,
        "bullet_prefix": "*",
        "records_sha256": _sha256(records_path),
    }
    atomic_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
