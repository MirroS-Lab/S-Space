"""Eleven CPU contracts for the retained numerical implementation."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from sspace.experiments.intervention.spatial_causality.spatial_jacobian.spatial_causality import (
    coordinate_right_inverse,
)
from sspace.experiments.validation.coco1800.selection.layers import (
    LayerMetrics,
    build_validation_evidence,
)
from sspace.smoke import (
    fingerprint,
    held_out_selection,
    hstar_contract,
    orientation_aggregate,
    pairwise_contract,
    projection_contract,
    right_inverse_contract,
    run_once,
    spinbench_contract,
    toy_extraction,
)


class CoreContractTests(unittest.TestCase):
    def test_final_logit_positive_minus_negative(self) -> None:
        contrasts, _, _ = toy_extraction()
        torch.testing.assert_close(contrasts, torch.tensor(((30.0, 60.0, -30.0),)))

    def test_query_reference_half_gradient(self) -> None:
        _, gradients, _ = toy_extraction()
        direction = torch.tensor((1.0, -2.0))
        torch.testing.assert_close(
            gradients, torch.stack((3.0 * direction, direction))[None]
        )

    def test_original_swapped_mean_and_unit_axes(self) -> None:
        result = orientation_aggregate()
        self.assertEqual(result["counts"], [2, 2, 2])
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(result["axes"]), axis=-1), 1.0
        )

    def test_query_minus_reference_projection(self) -> None:
        self.assertEqual(projection_contract(), [[[2.0, 3.0, 2.5]]])

    def test_held_out_layer_selection_and_overlap_gate(self) -> None:
        result = held_out_selection()
        self.assertEqual(result["selected_layer"], 3)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_validation_evidence(
                "toy",
                "a" * 64,
                (LayerMetrics(0, 1, 1, 1.0, (1.0, 1.0, 1.0), 1.0),),
                (1,),
                (1,),
            )

    def test_pairwise_sign_and_tie(self) -> None:
        self.assertEqual(pairwise_contract()["tie"], "tie")

    def test_spinbench_proper_rotation_and_sign(self) -> None:
        result = spinbench_contract()
        self.assertEqual(result["left_view"], [2.0, 0.0, 0.0])

    def test_hstar_ba_minus_ab(self) -> None:
        result = hstar_contract()
        self.assertEqual(result["accuracy"], 1.0)

    def test_spatial_causality_right_inverse(self) -> None:
        self.assertLessEqual(right_inverse_contract()["max_error"], 1e-12)
        with self.assertRaisesRegex(ValueError, "rank three"):
            coordinate_right_inverse(torch.ones((3, 4), dtype=torch.float64))

    def test_same_seed_fingerprint(self) -> None:
        first = fingerprint(run_once(7))
        second = fingerprint(run_once(7))
        self.assertEqual(first, second)
        self.assertNotEqual(first, fingerprint(run_once(8)))


if __name__ == "__main__":
    unittest.main()
