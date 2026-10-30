"""Guardrails for weight-only development tuning."""
import unittest
from atlas_deepc.benchmarks import tune_mvc_bdpc_terminal8 as tune


class TuningTests(unittest.TestCase):
    def test_only_control_weights_change(self):
        base = tune.m.Config()
        c = tune.weight_config(base, {"q_xy": 40000., "lambda_epsilon": 20.})
        self.assertEqual(c.q_xy, 40000.)
        self.assertEqual(c.svd_rtol, base.svd_rtol)
        self.assertEqual(c.past, base.past)
        self.assertEqual(c.lambda_view1, base.lambda_view1)
        for key in ("svd_rtol", "clusters", "past", "solver_tol"):
            with self.assertRaises(ValueError):
                tune.weight_config(base, {key: 1})

    def test_invalid_weights_rejected(self):
        for value in (0., -1., float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tune.weight_config(tune.m.Config(), {"r": value})

    def test_no_test_seeds_or_too_short_runs(self):
        for seeds, steps in [([10001], 300), ([211, 211], 300), ([], 300), ([211], 99), ([211], 301)]:
            with self.assertRaises(ValueError):
                tune.run("unused", "unused", seeds, steps)
