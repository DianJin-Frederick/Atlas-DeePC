"""Integration checks for the actual baseline QP and resumable rank runner."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from concurrent.futures import Future

import numpy as np
from atlas_deepc.ablations import linear_subspace_rank_ablation as experiment
class RankAblationTests(unittest.TestCase):
    def test_low_default_high_rank_qp_and_checkpoint(self):
        experiment.initialize_worker()
        config_factory = experiment.rank_config
        solve = experiment.pipe.DeePCController.solve
        chart_members = {}
        with tempfile.TemporaryDirectory() as directory:
            for rank in (20, 60, 80):
                controller = experiment.controller_for_rank(rank)
                config = controller.config
                expected = asdict(experiment.baseline.default_config())
                expected["svd_rank"] = rank
                self.assertEqual(asdict(config), expected)
                for key, (matrix, members) in controller.matrix_cache.items():
                    if key in chart_members:
                        np.testing.assert_array_equal(members, chart_members[key])
                    chart_members[key] = members.copy()
                    self.assertEqual(matrix["basis"].shape, (144, rank))
                    coordinates = matrix["basis"].T @ experiment._H[:, members]
                    mean = coordinates.mean(axis=1, keepdims=True)
                    scale = np.maximum(coordinates.std(axis=1, keepdims=True), 1e-8)
                    np.testing.assert_allclose(matrix["g_min"], mean - 3*scale)
                    np.testing.assert_allclose(matrix["g_max"], mean + 3*scale)
                seen = []

                def checked_solve(qp, matrix, u_ini, y_ini, y_ref):
                    np.testing.assert_array_equal(qp.Q, np.diag([5000]*4 + [20]*2 + [100000]*2 + [5000]*2))
                    np.testing.assert_array_equal(qp.R, .0001*np.eye(2))
                    self.assertFalse(hasattr(qp, "Q_terminal"))
                    self.assertEqual((qp.lam_ini, qp.u_limit), (100000, 1))
                    self.assertEqual((u_ini.shape, y_ini.shape, y_ref.shape), ((4, 1), (20, 1), (100, 1)))
                    self.assertFalse(np.any(matrix["offset"]))
                    self.assertNotIn("g_sum", matrix)
                    result = solve(qp, matrix, u_ini, y_ini, y_ref)
                    self.assertIn(result[3], ("optimal", "optimal_inaccurate"))
                    self.assertEqual((result[0].shape, result[1].shape, result[2].shape), ((20, 1), (100, 1), (rank, 1)))
                    seen.append(rank)
                    return result

                controller.config = replace(config, steps=8)
                source = Path(directory) / f"full{rank}.npz"
                destination = Path(directory) / f"compact{rank}.npz"
                try:
                    with patch.object(experiment.pipe.DeePCController, "solve", checked_solve):
                        controller.run(211, source)
                    self.assertEqual(len(seen), 8)
                    with np.load(source, allow_pickle=True) as trace:
                        np.testing.assert_array_equal(trace["matrix_columns"],
                            [len(chart_members[(int(chart),)]) for chart in trace["chart"]])
                        self.assertEqual(trace["ref"].shape, (8, 10))
                    with patch.object(experiment, "rank_config", lambda r: replace(config_factory(r), steps=8)):
                        row = experiment.compact_trace(source, destination, rank, 211)
                        self.assertEqual(experiment.load_checkpoint(destination, rank, 211), row)
                    timing = experiment.summarize_timing([destination])
                    self.assertEqual(timing["timed_steps"], 3)
                    modules = timing["modules"]
                    self.assertAlmostEqual(sum(r["mean_ms"] for r in modules[:-1]), modules[-1]["mean_ms"])
                finally:
                    controller.config = config

    def test_failed_future_stops_submission(self):
        class FailedPool:
            def __init__(self, **kwargs):
                self.submitted = 0
                self.shutdown_args = None

            def submit(self, *args):
                self.submitted += 1
                future = Future()
                future.set_exception(RuntimeError("deliberate worker failure"))
                return future

            def shutdown(self, **kwargs):
                self.shutdown_args = kwargs

        pool = FailedPool()
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(experiment, "RESULT_ROOT", Path(directory)), \
                patch.object(experiment, "prepare", return_value={}), \
                patch.object(experiment, "ProcessPoolExecutor", return_value=pool):
            with self.assertRaisesRegex(RuntimeError, "deliberate worker failure"):
                experiment.run_all(workers=3)
        self.assertEqual(pool.submitted, 3)
        self.assertEqual(pool.shutdown_args, {"wait": False, "cancel_futures": True})


if __name__ == "__main__":
    unittest.main()
