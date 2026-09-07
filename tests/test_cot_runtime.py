"""Contracts for shared fixed-template CoT token handling."""

import unittest

import torch

from sspace.experiments.spinbench.evolving_cot.runtime import (
    _append_generated,
    _mentions,
)


class Tokenizer:
    def __call__(self, text, **kwargs):
        return {"input_ids": [10000] if text == "</think>" else list(map(ord, text))}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


class CotRuntimeTests(unittest.TestCase):
    def test_mentions_stop_at_reasoning_boundary_and_keep_aliases(self):
        tokenizer = Tokenizer()
        text = "red cube and rubik's cube"
        ids = list(map(ord, text)) + [10000] + list(map(ord, " red cube"))
        matches = _mentions(ids, tokenizer, {"a": "red cube", "b": "rubiks cube"})
        self.assertEqual({match["role"] for match in matches}, {"a", "b"})
        self.assertTrue(all(match["generated_token_end"] <= len(text) for match in matches))
        for match in matches:
            start, end = match["generated_token_start"], match["generated_token_end"]
            self.assertEqual(ids[start:end], match["token_ids"])

    def test_prefill_extends_only_text_tokens_and_keeps_image_inputs(self):
        pixels = torch.tensor([3.])
        inputs = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]]),
            "mm_token_type_ids": torch.tensor([[1, 0]]),
            "pixel_values": pixels,
        }
        _append_generated(inputs, [5, 6])
        torch.testing.assert_close(inputs["input_ids"], torch.tensor([[1, 2, 5, 6]]))
        torch.testing.assert_close(inputs["attention_mask"], torch.ones((1, 4), dtype=torch.long))
        torch.testing.assert_close(inputs["mm_token_type_ids"], torch.tensor([[1, 0, 0, 0]]))
        self.assertIs(inputs["pixel_values"], pixels)


if __name__ == "__main__":
    unittest.main()
