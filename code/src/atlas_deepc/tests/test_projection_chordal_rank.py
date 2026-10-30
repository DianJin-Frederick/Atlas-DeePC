"""Rank-selection order and current-baseline protocol isolation."""

import copy
import unittest

from atlas_deepc.ablations import projection_chordal_rank as study
from atlas_deepc.reporting.four_case_tracking_table import format_table


class RankStudyTests(unittest.TestCase):
    def test_success_precedes_rmse_and_rank(self):
        records = [{"rank": 55, "successes": 80, "median_rollout_rmse_mm": 30.},
                   {"rank": 60, "successes": 81, "median_rollout_rmse_mm": 80.},
                   {"rank": 61, "successes": 81, "median_rollout_rmse_mm": 70.},
                   {"rank": 59, "successes": 81, "median_rollout_rmse_mm": 70.}]
        self.assertEqual(study.best_record(records)["rank"], 59)

    def test_rank_only_signature(self):
        signature = copy.deepcopy(study.control_signature(10, "graph"))
        signature["config"]["svd_rank"] = 59
        signature["record_coefficients"] = False
        signature["timing_workers"] = 4
        study.check_signature(signature, 10, "graph", 59, 4)
        signature["config"]["lam_du"] = 5.
        with self.assertRaises(AssertionError):
            study.check_signature(signature, 10, "graph", 59, 4)

    def test_table_adds_rank_row_and_uses_median_rmse(self):
        metrics = {"success_rate": 80., "median_error_mm": 2., "p90_error_mm": 10.,
                   "p95_error_mm": 20., "p99_error_mm": 30., "median_rollout_rmse_mm": 40.,
                   "median_time_ms": 30., "p95_time_ms": 50., "p99_time_ms": 60.}
        columns = [{**metrics, "rank": r} for r in (55, 56, 57, 58)]
        table = format_table(columns, "Example", "tab:test", include_rank=True)
        self.assertIn("Selected rank & 55 & 56 & 57 & 58", table)
        self.assertIn("Median rollout RMSE", table)
        self.assertIn(r"\label{tab:test}", table)
        self.assertNotIn("Full-rollout RMSE", table)


if __name__ == "__main__":
    unittest.main()
