import unittest
import numpy as np
from atlas_deepc.benchmarks import mvc_bdpc_rank_experiment as experiment


class FixedRankTests(unittest.TestCase):
    def test_exact_rank_53_and_original_width(self):
        matrix = np.random.default_rng(4).normal(size=(70, 194))
        basis, reduced, spectrum = experiment.fixed_subspace(matrix, 53)
        self.assertEqual(basis.shape, (70, 53))
        self.assertEqual(reduced.shape, matrix.shape)
        self.assertEqual(np.linalg.matrix_rank(reduced), 53)
        np.testing.assert_allclose(basis.T @ basis, np.eye(53), atol=1e-12)
        self.assertAlmostEqual(np.linalg.norm(matrix-reduced)**2, np.sum(spectrum[53:]**2), places=8)

    def test_no_null_direction_padding(self):
        with self.assertRaises(ValueError):
            experiment.fixed_subspace(np.ones((70, 194)), 53)

    def test_invalid_ranks_and_test_seeds(self):
        for rank in [0, 71, 2.5, True]:
            with self.assertRaises(ValueError):
                experiment.fixed_subspace(np.ones((70, 194)), rank)
        with self.assertRaises(ValueError):
            experiment.run("unused", 53, [10001])
