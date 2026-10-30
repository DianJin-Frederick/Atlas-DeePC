"""Held-out protocol, resumability, and reporting checks without test rollouts."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_deepc.benchmarks import reacher_heldout as study


class HeldoutTest(unittest.TestCase):
    def test_frozen_seeds_and_ranks(self):
        study.validate_seeds(study.SEEDS)
        self.assertEqual(len(study.SEEDS), 500)
        self.assertFalse(set(study.SEEDS) & set(range(2026, 2226)))
        for wrong in (range(211, 711), study.SEEDS[:-1], tuple(reversed(study.SEEDS))):
            with self.assertRaises(ValueError):
                study.validate_seeds(wrong)
        self.assertEqual([r for p, m, r in study.CASES], [53, 53, 57, 60])
        for p, mode, rank in study.CASES:
            config = study.config_for(p, mode)
            self.assertEqual(config.clusters, 48)
            self.assertEqual(config.svd_rank, rank)
            self.assertEqual((config.lam_g, config.lam_du, config.lam_ini), (0, 0, 1e5))

    def test_protocol_rejects_development_config_drift(self):
        original = study.experiment.baseline.default_config
        with patch.object(study.experiment.baseline, "default_config",
                          side_effect=lambda: study.replace(original(), q_xy=123)):
            with self.assertRaises(AssertionError):
                study.config_for(8, "terminal")

    def test_partial_report_does_not_generate_final_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = study.report(tmp)
            self.assertFalse(result["complete"])
            self.assertEqual(sum(r["completed"] for r in result["progress"]), 0)
            self.assertFalse((Path(tmp) / "table_heldout500.tex").exists())

    def test_full_report_keeps_fractional_percentages(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "SEEDS", (10001,)), \
                patch.object(study, "verify_paired_resets") as paired:
            root = Path(tmp)
            for p, mode, rank in study.CASES:
                directory = root / f"output{p}/{mode}"
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text("{}")
                (directory / "summary.json").write_text("{}")
                (directory / "seed10001_control_results.npz").touch()
                record = {"rank": rank, "successes": 441, "success_rate": 88.2,
                    "median_error_mm": 2., "p90_error_mm": 3., "p95_error_mm": 4., "p99_error_mm": 5.,
                    "median_rollout_rmse_mm": 60., "median_time_ms": 30., "p95_time_ms": 40., "p99_time_ms": 50.,
                    "signature_sha256": study.file_digest(directory / "manifest.json"),
                    "summary_sha256": study.file_digest(directory / "summary.json")}
                study.save_json(directory / "metrics.json", record)
            result = study.report(root)
            self.assertTrue(result["complete"])
            self.assertIn(r"88.2\%", (root / "table_heldout500.tex").read_text())
            paired.assert_called_once_with(root)

    def test_run_case_resumes_complete_panel_without_controller(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(study, "SEEDS", (10001,)):
            root = Path(tmp)
            directory = root / "output8/terminal"
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{}")
            (directory / "seed10001_control_results.npz").touch()
            with patch.object(study, "controller_context") as controller, \
                    patch.object(study, "summarize_case", return_value={"cached": True}):
                self.assertEqual(study.run_case(root, 8, "terminal"), {"cached": True})
                controller.assert_not_called()


if __name__ == "__main__":
    unittest.main()
