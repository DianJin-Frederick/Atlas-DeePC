"""Check every executable controller copy without launching a simulator."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import cvxpy as cp
import numpy as np


ROOT = project_root()


def unpack(matrix):
    return matrix["offset"], matrix["basis"], matrix.get("g_min"), matrix.get("g_max")


def load_controller(path, name="DeePCController"):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = dict(cp=cp, np=np, M=1, P=1, T_INI=1, T_F=1,
                     unpack_basis=unpack, pipe=SimpleNamespace(unpack_basis=unpack),
                     TaskSpec=object, PilotConfig=object,
                     controller_q_diagonal=lambda spec, config: np.ones(spec.output_dim))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def matrix_with_impossible_box():
    # Exact past input is zero, making g[0] >= 100 impossible if enforced.
    return {"basis": np.eye(4), "offset": np.zeros((4, 1)),
            "g_min": np.full((4, 1), 100.0), "g_max": np.full((4, 1), 101.0)}


class CoefficientBoundsDisabledTest(unittest.TestCase):
    def reacher_controllers(self):
        paths = [ROOT / "src/atlas_deepc/control/expert_pipeline_search.py"]
        paths += sorted((ROOT / "experiments").glob("**/frozen_protocol/src/expert_pipeline_search.py"))
        paths += sorted((ROOT / "experiments").glob("**/frozen_protocol/src/pusher_routed_deepc_experiment.py"))
        self.assertEqual(len(paths), 7)
        for path in paths:
            if path.name.startswith("pusher"):
                controller = load_controller(path, "DimensionedDeePCController")(
                    np.eye(1), np.eye(1), .05, 100., 0., 1., cp.CLARABEL)
            elif path == ROOT / "src/atlas_deepc/control/expert_pipeline_search.py":
                controller = load_controller(path)(np.eye(1), np.eye(1), 100., 1., cp.CLARABEL)
            else:
                controller = load_controller(path)(
                    np.eye(1), np.eye(1), np.zeros((1, 1)), .05, 100., 0., 1., cp.CLARABEL)
            yield path, controller

    def test_empirical_boxes_ignored_in_all_copies(self):
        for path, controller in self.reacher_controllers():
            with self.subTest(path=str(path)):
                result = controller.solve(matrix_with_impossible_box(), np.zeros((1, 1)),
                                          np.zeros((1, 1)), np.ones((1, 1)))
                self.assertIn(result[3], ("optimal", "optimal_inaccurate"))
                self.assertLess(abs(result[2][0, 0]), 1e-7)

    def test_affine_constraint_preserved_without_empirical_box(self):
        for path, controller in self.reacher_controllers():
            with self.subTest(path=str(path)):
                matrix = {**matrix_with_impossible_box(), "g_sum": 1.0}
                result = controller.solve(matrix, np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
                self.assertIn(result[3], ("optimal", "optimal_inaccurate"))
                self.assertAlmostEqual(float(result[2].sum()), 1.0, places=6)

    def test_raw_column_floor_preserved(self):
        for path, controller in self.reacher_controllers():
            with self.subTest(path=str(path)):
                matrix = matrix_with_impossible_box()
                matrix.pop("g_max")
                result = controller.solve(matrix, np.zeros((1, 1)), np.zeros((1, 1)), np.ones((1, 1)))
                self.assertIsNone(result[0])
                self.assertIn("infeasible", result[3])

    def test_input_limits_preserved(self):
        for path, controller in self.reacher_controllers():
            with self.subTest(path=str(path)):
                matrix = {"offset": np.zeros((4, 1)),
                          "basis": np.array([[1., 0., 0.], [0., 1., 0.],
                                             [0., 0., 1.], [0., 1., 0.]])}
                result = controller.solve(matrix, np.zeros((1, 1)), np.zeros((1, 1)), np.full((1, 1), 100.))
                self.assertIn(result[3], ("optimal", "optimal_inaccurate"))
                self.assertAlmostEqual(float(result[0][0, 0]), 1., places=5)

    def test_explicit_nonnegativity_preserved(self):
        for path, controller in self.reacher_controllers():
            if path.name.startswith("pusher"):
                continue
            with self.subTest(path=str(path)):
                matrix = {**matrix_with_impossible_box(), "g_nonnegative": True}
                result = controller.solve(matrix, np.full((1, 1), -.5), np.zeros((1, 1)), np.ones((1, 1)))
                self.assertIsNone(result[0])
                self.assertIn("infeasible", result[3])


if __name__ == "__main__":
    unittest.main()
