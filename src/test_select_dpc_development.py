"""Targeted checks for the bounded 200-column development study."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

import faithful_dpc_controllers as native
import select_dpc_development as study


class StudyTests(unittest.TestCase):
    def test_fixed_budget_and_no_other_methods(self):
        p = study.default_plan()
        study.validate_plan(p)
        self.assertEqual(p["seeds"], list(range(211, 311)))
        self.assertEqual(len(p["candidates"]), 6)
        self.assertEqual(6 * len(p["screening_seeds"]) + 3 * (100-len(p["screening_seeds"])), 330)
        self.assertTrue(all(c["columns"] == 200 for c in p["candidates"].values()))
        p["candidates"]["paper_weights"]["columns"] = 100
        with self.assertRaises(ValueError):
            study.validate_plan(p)

    def test_finalists_retain_paper_and_rank_alternatives(self):
        p = study.default_plan()
        names = list(p["candidates"])
        rows = {n: [{"success": False, "common_task_cost": 20.}] for n in names}
        rows[names[2]] = [{"success": True, "common_task_cost": 100.}]
        rows[names[3]] = [{"success": True, "common_task_cost": 10.}]
        self.assertEqual(study.choose_finalists(rows, p), ["paper_weights", names[3], names[2]])

    def test_strict_success_and_failures(self):
        p = study.default_plan()
        trace = {"y": np.zeros((320, 10)), "u": np.zeros((320, 2)), "target": np.zeros(2),
                 "control_compute_time_s": np.full(320, .02), "statuses": np.full((320, 5), "optimal")}
        row = study.summarize(trace, p)
        self.assertTrue(row["success"])
        self.assertEqual(row["subproblems"], 1600)
        trace["y"][-1, 6] = .021
        self.assertFalse(study.summarize(trace, p)["success"])

    def test_partial_reports_never_claim_100_seed_winner(self):
        p = study.default_plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study.write_json(root / "development/paper_weights/seed211.json",
                             {"seed": 211, "success": True, "common_task_cost": 1.,
                              "unsolved_subproblems": 0, "subproblems": 1600})
            report = study.report(root, p)
            self.assertEqual(report["completed_rollouts"], 1)
            self.assertFalse(report["complete"])
            self.assertIsNone(report["best_completed_100_seed_candidate"])
            self.assertFalse((root / "best_parameters.json").exists())

    def test_full_report_and_resume_guard(self):
        p = study.default_plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study.write_json(root / "finalists.json", {"candidates": ["paper_weights"]})
            for seed in p["seeds"]:
                study.write_json(root / f"development/paper_weights/seed{seed}.json",
                                 {"seed": seed, "success": seed % 2 == 0, "common_task_cost": 1.,
                                  "unsolved_subproblems": 0, "subproblems": 1600})
            self.assertTrue(study.report(root, p)["complete"])
            winner = json.loads((root / "best_parameters.json").read_text())
            self.assertEqual(winner["successes"], 50)
            self.assertEqual(winner["trials"], 100)

    def test_raw_qp_projection_and_objective(self):
        rng = np.random.default_rng(19)
        c = native.SelectConfig(horizon=3, columns=200, q_xy=10, q_velocity=2, r=1,
                                lambda_g1=.1, lambda_projection=3, lambda_slack=50)
        Hu, Hy = rng.normal(size=(10, 200)), rng.normal(size=(40, 200))
        with threadpool_limits(limits=1):
            model = native.SelectSubproblem(Hu, Hy, c)
            _, status = model.solve(np.zeros(4), np.zeros(16), native.native_reference([.1, -.1]))
        self.assertIn(status, (cp.OPTIMAL, cp.OPTIMAL_INACCURATE))
        self.assertTrue(model.problem.is_qp())
        self.assertEqual(model.g.size, 200)
        Z = np.vstack([Hu, Hy[:16], np.ones((1, 200))])
        with threadpool_limits(limits=1):
            np.testing.assert_allclose(model.projection, np.linalg.pinv(Z) @ Z, atol=1e-12)
        np.testing.assert_allclose(model.projection @ model.projection, model.projection, atol=1e-12)
        g, u, y, slack = model.g.value, model.u.value, model.y.value, model.slack.value
        self.assertAlmostEqual(g.sum(), 1., places=5)
        np.testing.assert_allclose(Hu[:4] @ g, np.zeros(4), atol=1e-5)
        np.testing.assert_allclose(Hy[:16] @ g, slack, atol=1e-5)
        q = np.tile(np.diag(model.Q), c.horizon)
        objective = np.sum(q * (y - model.ref.value)**2) + c.r * np.sum(u**2)
        objective += c.lambda_slack * np.abs(slack).sum() + c.lambda_g1 * np.abs(g).sum()
        objective += c.lambda_projection * np.sum(((np.eye(200) - model.projection) @ g)**2)
        self.assertAlmostEqual(objective, model.problem.value, places=5)


if __name__ == "__main__":
    unittest.main()
