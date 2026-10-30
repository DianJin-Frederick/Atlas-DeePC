"""Protocol isolation and non-invasive full-data logging checks."""

import tempfile
import unittest

import numpy as np

from atlas_deepc.benchmarks import atlas_reacher_comparison as study


class ComparisonTest(unittest.TestCase):
    def test_prepare_uses_100_development_seeds_and_preserves_other_studies(self):
        runner = study.runner
        before = (runner.SEEDS, runner.CASES, runner.CONTROL_STEPS)
        with tempfile.TemporaryDirectory() as root:
            protocol = study.prepare(root)
            self.assertEqual(protocol["seeds"], list(range(211, 311)))
            self.assertEqual(protocol["cases"], [[8, "terminal", 53]])
            self.assertEqual(protocol["charts"], {"8_terminal": 64})
            self.assertEqual(protocol["control_steps"], 300)
            config = protocol["case_signatures"]["8_terminal"]["config"]
            self.assertEqual((config["lam_g"], config["lam_du"], config["lam_ini"]), (0, 0, 1e5))
            self.assertEqual(study.prepare(root), protocol)
            self.assertEqual(study.report(root)["completed"], 0)
        self.assertEqual((runner.SEEDS, runner.CASES, runner.CONTROL_STEPS), before)

    def test_logging_does_not_change_solver_return(self):
        u, y, g = np.zeros((20, 1)), np.zeros((80, 1)), np.zeros((53, 1))
        answer = (u, y, g, "optimal", 4.)

        class FakeController:
            Q = np.eye(8)
            R = np.eye(2)
            lam_ini = 10.

            def solve(self, *args):
                return answer

        records = []
        controller = study.logging_controller(FakeController, records)()
        matrix = {"basis": np.zeros((120, 53)), "offset": np.zeros((120, 1))}
        got = controller.solve(matrix, np.zeros((4, 1)), np.ones((16, 1)), np.zeros((80, 1)))
        self.assertIs(got, answer)
        self.assertEqual(len(records), 1)
        np.testing.assert_array_equal(records[0]["slack"], -np.ones(16))
        self.assertEqual(records[0]["objective"], 160.)
        self.assertEqual(records[0]["qp_equality_residual"], 0.)

    def test_logging_keeps_failed_solve_as_failure(self):
        answer = (None, None, None, "infeasible", np.inf)

        class FailedController:
            def solve(self, *args):
                return answer

        records = []
        controller = study.logging_controller(FailedController, records)()
        self.assertIs(controller.solve({}, np.zeros((4, 1)), np.zeros((16, 1)), None), answer)
        self.assertTrue(np.isnan(records[0]["planned_y"]).all())
        self.assertTrue(np.isnan(records[0]["g_coefficients"]).all())


if __name__ == "__main__":
    unittest.main()
