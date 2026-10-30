
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from dataclasses import asdict
import unittest

from atlas_deepc.baselines import baseline_atlas_deepc as baseline
from atlas_deepc.reporting.baseline_success_rate import success_statistics, validate_panel


class SuccessRateTests(unittest.TestCase):
    def payload(self):
        config = baseline.default_config()
        row = dict(seed=211, baseline_version=baseline.BASELINE_VERSION,
                   matrix_geometry="linear", candidate_count=10,
                   dynamic_terminal_reference=True, steps=config.steps,
                   minimum_selected_goal_source_separation=0.07,
                   mean_last_30=0.005, max_last_30=0.009, success_30=True,
                   mean_last_100=0.006, max_last_100=0.015, success_100=True)
        payload = dict(method=baseline.BASELINE_VERSION, results=[row], signature=dict(
            matrix_geometry="linear", candidate_count=10,
            config={k: v for k, v in asdict(config).items() if k != "seed"}))
        return config, payload

    def test_incomplete_panel_is_not_a_full_result(self):
        config, payload = self.payload()
        rows, missing = validate_panel(payload, range(211, 1211), config)
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(missing), 999)

    def test_rejects_old_geometry_duplicate_seeds_and_wrong_labels(self):
        config, payload = self.payload()
        payload["signature"]["matrix_geometry"] = "affine"
        with self.assertRaises(ValueError):
            validate_panel(payload, [211], config)
        config, payload = self.payload()
        payload["results"] *= 2
        with self.assertRaises(ValueError):
            validate_panel(payload, [211], config)
        config, payload = self.payload()
        payload["results"][0]["max_last_100"] = 0.025
        with self.assertRaises(ValueError):
            validate_panel(payload, [211], config)

    def test_wilson_and_sustained_success(self):
        rows = [dict(success_30=True, success_100=True)] * 900
        rows += [dict(success_30=True, success_100=False)] * 100
        result = success_statistics(rows)
        self.assertEqual(result["successes"], 900)
        self.assertEqual(result["failures"], 100)
        self.assertAlmostEqual(result["wilson_low"], 0.879848, places=5)
        self.assertAlmostEqual(result["wilson_high"], 0.91709, places=5)

    def test_distance_validation_allows_only_roundoff(self):
        config, payload = self.payload()
        row = payload["results"][0]
        row["mean_last_30"] = 0.0033495108298085504
        row["max_last_30"] = 0.0033495108298085496
        rows, missing = validate_panel(payload, [211], config)
        self.assertEqual(len(rows), 1)
        self.assertFalse(missing)
        row["mean_last_30"] = 0.004
        with self.assertRaises(ValueError):
            validate_panel(payload, [211], config)


if __name__ == "__main__":
    unittest.main()
