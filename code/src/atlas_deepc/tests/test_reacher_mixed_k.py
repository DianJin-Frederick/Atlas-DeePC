"""Mixed-K protocol and 300-step trace checks, without running experiments."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import reacher_heldout as runner
from atlas_deepc.benchmarks import reacher_mixed_k as study


class MixedKTest(unittest.TestCase):
    def test_notebook_refreshes_stale_runner_before_prepare(self):
        notebook = study.ROOT / "notebooks/reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb"
        cells = json.loads(notebook.read_text())["cells"]
        setup = next("".join(c["source"]) for c in cells
                     if c["cell_type"] == "code" and "saved_manifest =" in "".join(c["source"]))
        imports = setup.split("RUN_EXPERIMENTS =", 1)[0]
        old_signature = runner.case_signature
        old_counts = runner.CHART_COUNTS
        try:
            del runner.CHART_COUNTS
            runner.case_signature = lambda *args: self.fail("Stale signature function was retained")
            scope = {}
            exec(compile(imports, str(notebook), "exec"), scope)
            self.assertTrue(hasattr(runner, "CHART_COUNTS"))
            self.assertIs(study._case_signature, runner.case_signature)
            with tempfile.TemporaryDirectory() as tmp:
                protocol = scope["study"].prepare(tmp)
                self.assertEqual(protocol["control_steps"], 300)
                self.assertEqual(protocol["charts"]["8_graph"], 96)
        finally:
            if not hasattr(runner, "CHART_COUNTS"):
                runner.CHART_COUNTS = old_counts
            if runner.case_signature.__name__ == "<lambda>":
                runner.case_signature = old_signature

    def test_selected_protocol_and_legacy_isolation(self):
        original = (runner.CASES, runner.CONTROL_STEPS, runner.PER_CASE_ATLASES)
        expected = [(8, "terminal", 64, 53), (8, "graph", 96, 57),
                    (10, "terminal", 80, 64), (10, "graph", 64, 62)]
        for p, mode, charts, rank in expected:
            config = study.config_for(p, mode)
            self.assertEqual((config.clusters, config.svd_rank, config.steps), (charts, rank, 300))
            self.assertEqual((config.lam_g, config.lam_du, config.lam_ini), (0, 0, 1e5))
        self.assertEqual((runner.CASES, runner.CONTROL_STEPS, runner.PER_CASE_ATLASES), original)
        self.assertEqual(runner.config_for(8, "terminal").clusters, 48)
        self.assertEqual(study.SEEDS, tuple(range(10001, 10501)))

    def test_prepare_keeps_four_atlases_and_prevents_cache_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = study.prepare(root)
            paths = [study.atlas_path(root, p, mode) for p, mode, _ in study.CASES]
            self.assertEqual(len(set(paths)), 4)
            for (p, mode, rank), path in zip(study.CASES, paths):
                signature = protocol["case_signatures"][f"{p}_{mode}"]
                self.assertEqual(signature["config"]["clusters"], study.CHART_COUNTS[p, mode])
                self.assertEqual(signature["config"]["steps"], 300)
                self.assertEqual(signature["config"]["svd_rank"], rank)
                self.assertEqual(signature["atlas_sha256"], runner.file_digest(path))
                with np.load(path) as atlas:
                    self.assertEqual(atlas["medoid_states"].shape, (study.CHART_COUNTS[p, mode], p))
            self.assertEqual(study.prepare(root), protocol)
            self.assertFalse(list(root.rglob("seed*_control_results.npz")))
            self.assertFalse(study.report(root)["complete"])
            with patch.object(study, "CONTROL_STEPS", 320):
                with self.assertRaisesRegex(ValueError, "protocol/source changed"):
                    study.prepare(root)

    def test_trace_validation_uses_case_chart_count_and_300_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed10001_control_results.npz"
            for p, mode, rank in study.CASES:
                row = {"evaluation_signature_sha256": "test", "seed": 10001,
                       "svd_rank": rank, "lam_g": 0, "lam_du": 0, "lam_ini": 1e5,
                       "unsolved_steps": 0}
                arrays = {"summary": np.array(json.dumps(row)), "y": np.zeros((300, p)),
                          "distance": np.zeros(300), "final_goal": np.zeros(p),
                          "selection_windows": np.zeros((300, (2+p)*12)),
                          "selection_residuals": np.zeros((300, study.CHART_COUNTS[p, mode])),
                          "chart": np.zeros(300, dtype=int), "ref": np.zeros((300, p)),
                          "goal_index": np.zeros(300), "status": np.full(300, "optimal")}
                np.savez(path, **arrays)
                with study.protocol_context():
                    runner.validate_trace(path, p, mode, "test")
                    np.savez(path, **dict(arrays, selection_residuals=np.zeros((300, 48))))
                    with self.assertRaises(AssertionError):
                        runner.validate_trace(path, p, mode, "test")
                    np.savez(path, **dict(arrays, y=np.zeros((320, p))))
                    with self.assertRaises(AssertionError):
                        runner.validate_trace(path, p, mode, "test")

    def test_mixed_report_has_chart_counts_and_correct_horizon(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(runner, "SEEDS", (10001,)), \
                patch.object(runner, "verify_paired_resets"):
            root = Path(tmp)
            for p, mode, rank in study.CASES:
                directory = root / f"output{p}/{mode}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text("{}")
                (directory / "summary.json").write_text("{}")
                (directory / "seed10001_control_results.npz").touch()
                row = {"rank": rank, "successes": 441, "success_rate": 88.2,
                       "median_error_mm": 2., "p90_error_mm": 3., "p95_error_mm": 4.,
                       "p99_error_mm": 5., "median_rollout_rmse_mm": 60., "median_time_ms": 30.,
                       "p95_time_ms": 40., "p99_time_ms": 50.,
                       "signature_sha256": runner.file_digest(directory / "manifest.json"),
                       "summary_sha256": runner.file_digest(directory / "summary.json")}
                runner.save_json(directory / "metrics.json", row)
            self.assertTrue(study.report(root)["complete"])
            table = (root / "table_heldout500.tex").read_text()
            self.assertIn("Charts $K$ & 64 & 96 & 80 & 64", table)
            self.assertIn("Selected rank & 53 & 57 & 64 & 62", table)
            self.assertIn("all 300 controlled steps", table)
            self.assertIn(r"88.2\%", table)
            self.assertIn("not a new untouched test set", table)


if __name__ == "__main__":
    unittest.main()
