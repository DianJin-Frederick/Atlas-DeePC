from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import cvxpy as cp

import baseline_clarabel_waypoint_comparison as experiment


class ClarabelWaypointTest(unittest.TestCase):
    def test_invalid_requests(self):
        with self.assertRaises(ValueError):
            experiment.run_panel([])
        with self.assertRaises(ValueError):
            experiment.run_panel([211], modes=("invalid",))
        with self.assertRaises(ValueError):
            experiment.run_panel([211], config=replace(experiment.default_config(), lam_g=0))

    def test_paired_rollouts_resume_and_solver_scope(self):
        original_solve = cp.Problem.solve
        config = replace(experiment.default_config(), steps=12)
        with tempfile.TemporaryDirectory() as directory:
            first = experiment.run_panel([237], directory, ("terminal", "graph"), config)
            self.assertIs(cp.Problem.solve, original_solve)
            self.assertEqual(first["session"]["new_rollouts"], 2)
            self.assertEqual(first["signature"]["solver"], "CLARABEL")
            for mode in ("terminal", "graph"):
                self.assertEqual(first["panels"][mode]["timing"]["timed_steps"], 7)
                self.assertEqual(experiment.load_rows(directory, mode, [237])[0]["lam_g"], .05)
            paths = list(Path(directory).glob("*/seed*.npz"))
            before = {p: p.stat().st_mtime_ns for p in paths}
            second = experiment.run_panel([237], directory, ("terminal", "graph"), config)
            self.assertEqual(second["session"]["reused_rollouts"], 2)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in paths})
            with self.assertRaises(ValueError):
                experiment.run_panel([237], directory, config=replace(config, steps=13))


if __name__ == "__main__":
    unittest.main()
