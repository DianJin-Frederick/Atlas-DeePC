import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cvxpy as cp
import numpy as np

import baseline_osqp_comparison as experiment
import expert_pipeline_search as pipe


class OSQPBaselineTests(unittest.TestCase):
    def test_scoped_solver_override_restores_original(self):
        original = pipe.DeePCController
        original_solve = cp.Problem.solve
        with experiment.use_osqp([]):
            controller = pipe.DeePCController(np.eye(10), np.eye(2), np.zeros((10, 10)),
                                             0, 1e5, 5, 1, cp.CLARABEL)
            self.assertEqual(controller.solver, cp.OSQP)
            self.assertEqual(controller.lam_g, 0)
        self.assertIs(pipe.DeePCController, original)
        self.assertIs(cp.Problem.solve, original_solve)

    def test_actual_short_paired_rollout_and_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment.prepare(directory)
            experiment.initialize_worker(directory)
            experiment._controller.config = replace(experiment._controller.config, steps=3)
            rows = experiment.run_pair(237)
            self.assertEqual({r["reference_mode"] for r in rows}, {"terminal", "graph"})
            for row in rows:
                self.assertEqual(row["solver"], "OSQP")
                self.assertEqual(row["lam_g"], 0)
                with np.load(row["result_path"], allow_pickle=True) as trace:
                    diagnostics = json.loads(trace["solver_diagnostics"].item())
                    self.assertEqual(len(diagnostics), 3)
                    self.assertTrue(all(d["solver"] == "OSQP" for d in diagnostics))
            self.assertEqual(rows, experiment.run_pair(237))
            manifest = Path(directory) / "manifest.json"
            signature = json.loads(manifest.read_text())
            signature["solver"] = "CLARABEL"
            manifest.write_text(json.dumps(signature))
            with self.assertRaises(ValueError):
                experiment.prepare(directory)


if __name__ == "__main__":
    unittest.main()
