"""Protocol and formulation checks, including an independent authors-code oracle."""
import ast
import json
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
import warnings
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from threadpoolctl import threadpool_limits

import faithful_dpc_controllers as native
import three_method_benchmark as bench


class ProtocolTests(unittest.TestCase):
    def test_test_leakage_and_architecture_are_rejected(self):
        plan = bench.default_plan()
        bench.validate_plan(plan)
        plan["test_seeds"][0] = plan["development_seeds"][0]
        with self.assertRaisesRegex(ValueError, "overlap"):
            bench.validate_plan(plan)
        plan = bench.default_plan()
        plan["methods"]["atlas_deepc"]["base"]["svd_rank"] = 40
        with self.assertRaisesRegex(ValueError, "architecture"):
            bench.validate_plan(plan)

    def test_common_cost_does_not_depend_on_tuned_controller_weights(self):
        plan = bench.default_plan()
        trace = {"y": np.zeros((100, 10)), "u": np.zeros((100, 2)), "target": np.array([.1, 0.]),
                 "control_compute_time_s": np.full(100, .01), "statuses": np.full((100, 1), "optimal")}
        first = bench.summarize_trace(trace, plan)
        plan["methods"]["atlas_deepc"]["base"]["q_xy"] = 1e-9
        second = bench.summarize_trace(trace, plan)
        self.assertEqual(first["common_task_cost"], second["common_task_cost"])
        self.assertAlmostEqual(first["common_task_cost"], 100000.)
        self.assertFalse(first["success"])

    def test_frozen_manifest_rejects_changed_weights(self):
        plan = bench.default_plan()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bench.check_manifest(root, plan)
            plan["evaluation_cost"]["q_xy"] *= 2
            with self.assertRaisesRegex(ValueError, "changed"):
                bench.check_manifest(root, plan)

    def test_episode_reconstruction_preserves_windows(self):
        rng = np.random.default_rng(2)
        inputs = [rng.normal(size=(8, 2)) for _ in range(2)]
        outputs = [rng.normal(size=(8, 10)) for _ in range(2)]
        H = np.vstack([np.hstack([native.block_hankel(u, 3) for u in inputs]),
                       np.hstack([native.block_hankel(y, 3) for y in outputs])])
        actual_u, actual_y = native.reconstruct_offline_episodes(H, length=3, episode_steps=8)
        np.testing.assert_array_equal(actual_u, inputs)
        np.testing.assert_array_equal(actual_y, outputs)
        H[0, 1] += 1
        with self.assertRaisesRegex(ValueError, "inconsistently"):
            native.reconstruct_offline_episodes(H, length=3, episode_steps=8)

    def test_tune_freeze_test_resume_and_paired_report(self):
        plan = bench.default_plan()
        plan.update(development_seeds=[91001, 91002], test_seeds=[100001, 100002],
                    screening_seeds=1, finalists=2, steps=100)
        for method in bench.METHODS:
            key = "lam_g" if method == "atlas_deepc" else (
                "lambda_g1" if method == "select_dpc" else "lambda_g_bar")
            base = plan["methods"][method]["base"][key]
            plan["methods"][method]["stages"] = [{key: [base, base * 2]}]
        calls = []

        class FakeRunner:
            def __init__(self):
                self.plan = plan

            def run(self, method, config, seed, path, steps=None):
                calls.append((method, seed))
                trace = {"y": np.zeros((100, 10)), "u": np.zeros((100, 2)),
                         "target": np.zeros(2), "control_compute_time_s": np.full(100, .01),
                         "statuses": np.full((100, 1), "optimal")}
                np.savez_compressed(path, **trace)
                return {"method": method, "seed": seed, "parameters": config,
                        **bench.summarize_trace(trace, plan)}

        with tempfile.TemporaryDirectory() as temp, patch("builtins.print"):
            root, runner = Path(temp), FakeRunner()
            bench.check_manifest(root, plan)
            with self.assertRaises(FileNotFoundError):
                bench.evaluate(runner, root)
            bench.tune(runner, root)
            self.assertTrue(all(seed in plan["development_seeds"] for _, seed in calls))
            bench.freeze(root, plan)
            with self.assertRaisesRegex(ValueError, "frozen"):
                bench.tune(runner, root)
            calls.clear()
            bench.evaluate(runner, root)
            self.assertEqual(len(calls), 6)
            self.assertTrue(all(seed in plan["test_seeds"] for _, seed in calls))
            bench.evaluate(runner, root)
            self.assertEqual(len(calls), 6)  # Completed rollouts are not rerun.
            payload = bench.report(root, plan)
            self.assertTrue(payload["complete"])
            self.assertEqual(len(payload["methods"]), 3)
            self.assertTrue((root / "three_method_comparison.png").exists())
            self.assertTrue(all(row["p99_compute_ms"] == 10. for row in payload["methods"]))
            frozen = json.loads((root / "frozen.json").read_text())
            missing = root / "test" / "select_dpc" / bench.digest(frozen["selected"]["select_dpc"])[:16] / "seed100002.json"
            missing.unlink()
            payload = bench.report(root, plan)
            self.assertFalse(payload["complete"])
            self.assertEqual(payload["paired_seeds"], [100001])


def load_reference_oracle(Hu, Hy):
    """Load unedited class bodies without unrelated torch/manifold imports.

    This is test-only. The two class bodies execute verbatim; only the data
    loader and timing accumulator are replaced with simple test fixtures.
    """
    source = Path("/Users/kumiko/Projects/choose-wisely-paper-main/select_deepc/deepc_controller.py")
    if not source.exists():
        raise unittest.SkipTest("Authors' supplied source tree is not installed")
    class Generator:
        def __init__(self, *unused):
            pass
        def generate_hankel_matrices(self, unused):
            return Hu, Hy
    class Averager:
        def update(self, value):
            pass
    names = {"_as_flat_array", "DataDrivenPredictiveController", "SelectDeePC"}
    tree = ast.parse(source.read_text())
    bodies = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    env = {"np": np, "cp": cp, "NDArray": np.ndarray, "BaseController": object,
           "DeePCDims": object, "DeePCCost": object, "DeePCConstraints": object,
           "DeePCControllerArgs": object, "HankelMatrixGenerator": Generator,
           "RecursiveAverager": Averager, "perf_counter": time.perf_counter, "warnings": warnings}
    exec(compile(ast.Module(body=bodies, type_ignores=[]), str(source), "exec"), env)
    return env["SelectDeePC"]


class FormulationTests(unittest.TestCase):
    def test_select_matches_original_multi_refinement_actions_and_queries(self):
        rng = np.random.default_rng(7)
        c = native.SelectConfig(horizon=3, columns=24, iterations=3, q_xy=10, r=1,
                               lambda_g1=.01, lambda_projection=2, lambda_slack=100)
        Hu, Hy = rng.normal(size=(10, 60)), rng.normal(size=(40, 60))
        oracle = load_reference_oracle(Hu, Hy)
        costs = SimpleNamespace(Q=np.diag([0., 0., 0., 0., 10., 10., 10., 10.]), R=np.eye(2),
                                slack_cost=100, regularizer_cost_g_1=.01, regularizer_cost_g_pi=2)
        constraints = SimpleNamespace(A_u=np.array([[1, 0], [-1, 0], [0, 1], [0, -1]]),
                                      b_u=np.ones(4), A_y=None, b_y=None)
        args = SimpleNamespace(deepc_dims=SimpleNamespace(T_past=2, T_fut=3, m=2, p=8),
                               trajectory_data=None, controller_costs=costs, controller_constraints=constraints,
                               input_reference=np.zeros(2), verbose=False)
        selections = []
        def selector(u, y, Hu, Hy, ref):
            distances = np.linalg.norm(Hu-u, ord=1, axis=0) + np.linalg.norm(Hy-y, ord=1, axis=0)
            order = np.argsort(distances)
            selections.append(order[:24])
            return order, distances
        original = oracle(args, selector, num_hankel_cols=24, n_iter=3)
        port = native.SelectDPC(Hu, Hy, c)
        with threadpool_limits(limits=1):
            for state in rng.normal(size=(3, 8)):
                reference = native.native_reference([.1, -.1])
                expected = original.compute_action(state, reference)
                action, _, info = port.step(state, reference)
                np.testing.assert_allclose(action, expected, atol=2e-4, rtol=2e-4)
                np.testing.assert_array_equal(info["selected"], selections[-1])
                self.assertEqual(len(info["statuses"]), 3)
                np.testing.assert_allclose(port.subproblem.planned()[0],
                                           original._deepc.get_planned_input_trajectory().reshape(-1), atol=2e-4)

    def test_contextual_raw_constraints_and_paper_penalties(self):
        rng = np.random.default_rng(9)
        c = native.ContextualConfig(horizon=3, columns=20, lambda_g_bar=.03, lambda_slack=2.)
        u_samples, y_samples = rng.normal(size=(100, 2)), rng.normal(size=(100, 8))
        Hu, Hy = rng.normal(size=(10, 40)), rng.normal(size=(40, 40))
        model = native.ContextualDPC(Hu, Hy, u_samples, y_samples, c)
        upast, ypast, ref = np.zeros((2, 2)), np.zeros((2, 8)), native.native_reference([.05, .02])
        with threadpool_limits(limits=1):
            action, prediction, info = model.step(ypast[-1], ref, upast=upast, ypast=ypast)
        self.assertEqual(info["statuses"], ["optimal"])
        self.assertEqual(model.g.size, 20)
        self.assertFalse(model.problem.is_qp())  # Unsquared L2(g) was not silently squared.
        self.assertAlmostEqual(model.g.value.sum(), 1., places=6)
        selected = info["selected"]
        np.testing.assert_array_equal(model.Yf.value, model.Hy[16:, selected])
        np.testing.assert_allclose(model.Up.value @ model.g.value, model.up.value, atol=1e-6)
        np.testing.assert_allclose(model.Yf.value @ model.g.value, model.y.value, atol=1e-6)
        self.assertTrue(np.all(np.abs(action) <= 1.0001))
        q = np.tile(model.q, c.horizon)
        manual = (np.sum(q*(model.y.value-model.ref.value)**2)
                  + c.r*(np.sum(model.u.value**2)+np.sum((model.u.value[2:]-model.u.value[:-2])**2))
                  + c.columns*c.lambda_g_bar*np.linalg.norm(model.g.value)
                  + c.lambda_slack*np.sum(np.abs(model.slack.value)))
        self.assertAlmostEqual(manual, model.problem.value, places=5)


if __name__ == "__main__":
    unittest.main()
