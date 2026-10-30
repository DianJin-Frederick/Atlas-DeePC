
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import small_atlas_ablations as experiment
class SumConstraintTests(unittest.TestCase):
    def test_cache_copy_adds_only_sum_constraint(self):
        basis = np.eye(3)
        original = {(0,): ({"offset": np.zeros((3, 1)), "basis": basis}, np.arange(3))}
        copied = experiment.sum_one_cache(original)
        self.assertNotIn("g_sum", original[(0,)][0])
        self.assertEqual(copied[(0,)][0]["g_sum"], 1.0)
        self.assertIs(copied[(0,)][0]["basis"], basis)
        self.assertEqual(set(copied[(0,)][0])-set(original[(0,)][0]), {"g_sum"})

    def test_actual_qp_coefficient_sum(self):
        with threadpool_limits(limits=1), tempfile.TemporaryDirectory() as folder:
            experiment.initialize()
            experiment._controller.config = replace(experiment._controller.config, steps=4)
            row = experiment.run_seed(211, folder)
            self.assertEqual(row["unsolved_steps"], 0)
            with np.load(Path(folder) / "seed211_control_results.npz", allow_pickle=True) as trace:
                np.testing.assert_allclose(trace["coefficient_sum"], 1.0, atol=1e-6)
            self.assertLess(row["max_g_sum_residual"], 1e-6)


if __name__ == "__main__":
    unittest.main()
