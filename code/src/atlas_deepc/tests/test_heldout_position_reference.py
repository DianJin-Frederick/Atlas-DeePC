"""Paired position-reference ablation keeps all other baseline settings frozen."""

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import cvxpy as cp
import numpy as np

from atlas_deepc.ablations import heldout_position_reference as study


class PositionReferenceTest(unittest.TestCase):
    def test_cost_is_exactly_two_dimensional_reference_tracking(self):
        rng = np.random.default_rng(17)
        p = 8
        Q = np.diag([0., 0., 0., 0., 0., 0., 1e5, 1e5])
        R = 1e-4*np.eye(2)
        controller = study.baseline.experiment.routed.pipe.DeePCController(
            Q, R, 1e5, 1., "CLARABEL", dimensions=(2, p, 2, 10))
        y, ref = rng.normal(size=(80, 1)), rng.normal(size=(80, 1))
        u, sigma = rng.normal(size=(20, 1)), rng.normal(size=(16, 1))
        value = controller.tracking_cost(cp.Constant(u), cp.Constant(y), cp.Constant(sigma), ref).value
        expected = (1e5*np.sum((y.reshape(10, p)[:, 6:8]-ref.reshape(10, p)[:, 6:8])**2)
                    + 1e-4*np.sum(u**2) + 1e5*np.sum(sigma**2))
        self.assertAlmostEqual(np.asarray(value/expected).item(), 1.)
        changed = ref.reshape(10, p).copy()
        changed[:, :6] += 10000.
        changed_value = controller.tracking_cost(
            cp.Constant(u), cp.Constant(y), cp.Constant(sigma), changed.reshape(80, 1)).value
        np.testing.assert_array_equal(value, changed_value)

    def test_only_nonposition_weights_change(self):
        for p, mode, _ in study.CASES:
            old = asdict(study._config_for(p, mode))
            new = asdict(study.config_for(p, mode))
            self.assertEqual({k for k in old if old[k] != new[k]}, {"q_data_goal_joint", "q_dq"})
            self.assertEqual((new["q_data_goal_joint"], new["q_dq"]), (0., 0.))
            self.assertEqual(new["q_xy"], 100000.)

    def test_context_restores_heldout_protocol(self):
        original_seeds = study.baseline.SEEDS
        original_config = study.baseline.config_for
        with study.ablation_context():
            self.assertEqual(study.baseline.SEEDS, tuple(range(10001, 10101)))
            self.assertEqual(study.baseline.config_for(8, "terminal").q_dq, 0.)
        self.assertIs(study.baseline.config_for, original_config)
        self.assertIs(study.baseline.SEEDS, original_seeds)

    def test_manifest_freezes_only_requested_case_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = (study.CASES[0],)
            first = study.prepare(root, cases)
            second = study.prepare(root, cases)
            self.assertEqual(first, second)
            self.assertEqual(first["cases"], [[8, "terminal", 53]])
            self.assertEqual(len(first["seeds"]), 100)
            self.assertEqual(first["case_signatures"]["8_terminal"]["tracked_output_rows"], [6, 7])
            self.assertFalse(list(root.rglob("seed*_control_results.npz")))


if __name__ == "__main__":
    unittest.main()
