"""Checkpoint validation regression tests; no environments or QP solves."""
from dataclasses import asdict, replace
import unittest

import baseline_control_parameter_ablation as ablation


class CheckpointValidationTest(unittest.TestCase):
    def row(self):
        return {
            **asdict(replace(ablation.default_config(), lam_g=4.941713361323833, seed=245)),
            "parameter_ablation_version": ablation.VERSION,
            "matrix_geometry": "linear", "candidate_count": 10,
            "dynamic_terminal_reference": True, "graph_waypoints_enabled": True,
            "reference_mode": "graph", "solver": "CLARABEL",
            "terminal_reference_policy": "iterative_minimum_route_cost",
            "statuses": {"optimal": 320}, "minimum_selected_goal_source_separation": .07,
            "exact_target_xy": [.1, .1],
            # Actual statistics from the failed seed-245 rollout.
            "mean_last_30": 0.015132354004415904,
            "max_last_30": 0.015132354004415899,
            "mean_last_100": 0.01513235400441726,
            "max_last_100": 0.015132354004430987,
            "success_30": False, "success_100": False,
        }

    def validate(self, row):
        return ablation.validate_row(row, row["lam_g"], row["seed"])

    def test_real_roundoff_preserves_metrics_and_failure(self):
        row = self.row()
        before = dict(row)
        self.assertIs(self.validate(row), row)
        self.assertEqual(row, before)
        self.assertFalse(row["success_30"])

    def test_mean_materially_above_max_is_rejected(self):
        row = self.row()
        row["mean_last_30"] = row["max_last_30"] + 1e-10
        with self.assertRaisesRegex(AssertionError, "Invalid final-30"):
            self.validate(row)

    def test_invalid_numbers_are_rejected(self):
        for value in (-1e-20, float("nan"), float("inf")):
            for key in ("mean_last_30", "max_last_30"):
                with self.subTest(value=value, key=key):
                    row = self.row()
                    row[key] = value
                    with self.assertRaises(AssertionError):
                        self.validate(row)

    def test_success_threshold_remains_strict(self):
        row = self.row()
        row.update(mean_last_30=.01, max_last_30=.01, success_30=False)
        self.validate(row)
        row["success_30"] = True
        with self.assertRaises(AssertionError):
            self.validate(row)

    def test_zero_error_is_valid(self):
        row = self.row()
        for window in (30, 100):
            row[f"mean_last_{window}"] = row[f"max_last_{window}"] = 0.0
            row[f"success_{window}"] = True
        self.validate(row)


if __name__ == "__main__":
    unittest.main()
