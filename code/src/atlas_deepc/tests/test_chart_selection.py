import unittest

import numpy as np

from atlas_deepc.atlas.chart_selection import FullWindowProjectionSelector
from atlas_deepc.ablations import chart_selection_experiment as study


class FullWindowProjectionTests(unittest.TestCase):
    def test_full_hankel_rows_and_least_squares_residual(self):
        rng = np.random.default_rng(71)
        # L=3, m=1, p=2: each full chart has (m+p)L=9 rows.
        cache = {(k,): ({"basis": rng.normal(size=(9, 3)), "offset": np.zeros((9, 1))}, [])
                 for k in range(3)}
        selector = FullWindowProjectionSelector(cache, 1, 2, 3)
        z = selector.pack_history([[1], [2], [3]], [[4, 5], [6, 7], [8, 9]])
        np.testing.assert_array_equal(z, np.arange(1, 10))
        expected = []
        for matrix, _ in cache.values():
            basis = matrix["basis"]
            expected.append(np.linalg.norm(z - basis @ np.linalg.lstsq(basis, z, rcond=None)[0]))
        np.testing.assert_allclose(selector.residuals(z), expected, atol=1e-12)
        self.assertEqual(selector.select(z), np.argmin(expected))
        self.assertEqual(study.rank_audit(selector)["window_entries"], 9)
        with self.assertRaisesRegex(ValueError, "Expected L observed"):
            selector.pack_history([1, 2], [3, 4, 5, 6, 7, 8])

    def test_reacher_dimensions_and_deterministic_ties(self):
        rng = np.random.default_rng(12)
        basis, _ = np.linalg.qr(rng.normal(size=(120, 60)))
        cache = {(k,): ({"basis": basis, "offset": np.zeros((120, 1))}, [])
                 for k in (2, 7)}
        selector = FullWindowProjectionSelector(cache, 2, 8, 12)
        self.assertEqual(selector.input_rows, 24)
        self.assertEqual(selector.output_rows, 96)
        self.assertEqual(selector.window_dimension, 120)
        self.assertEqual(selector.ranks.tolist(), [60, 60])
        z = selector.pack_history(np.zeros((12, 2)), np.ones((12, 8)))
        self.assertEqual(selector.select(z), 2)
        self.assertEqual(selector.tie_history, [2])

    def test_invalid_chart_geometry_and_query(self):
        cache = {(0,): ({"basis": np.eye(9), "offset": np.ones((9, 1))}, [])}
        with self.assertRaisesRegex(ValueError, "requires linear charts"):
            FullWindowProjectionSelector(cache, 1, 2, 3)
        cache[(0,)][0]["offset"][:] = 0
        selector = FullWindowProjectionSelector(cache, 1, 2, 3)
        with self.assertRaisesRegex(ValueError, "finite full-window"):
            selector.select(np.ones(8))


if __name__ == "__main__":
    unittest.main()
