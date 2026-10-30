"""The smoothing ablation must change only one current-baseline setting."""

import json
import unittest
from unittest.mock import patch

from atlas_deepc.ablations import four_baseline_input_smoothing as study


class InputSmoothingTests(unittest.TestCase):
    def test_rejects_additional_changes(self):
        previous = {"config": {"lam_du": 5., "lam_g": 0., "svd_rank": 60}}
        current = {"config": {"lam_du": 0., "lam_g": 0., "svd_rank": 60}}
        study.assert_only_smoothing_changed(current, previous)
        current["config"]["svd_rank"] = 59
        with self.assertRaises(AssertionError):
            study.assert_only_smoothing_changed(current, previous)

    def test_scoped_override_restores_baseline_defaults(self):
        original = study.four.experiment.default_config()
        path = study.baseline.RESULT_ROOT / "output8/terminal/manifest.json"
        signature = json.loads(path.read_text())
        signature["config"]["lam_du"] = 0.

        def fake_run(*args, **kwargs):
            self.assertEqual(study.four.experiment.default_config().lam_du, 0.)
            self.assertEqual(kwargs["chart_selection_method"], "full_window_projection")
            self.assertEqual(kwargs["graph_cost"], "chordal")
            self.assertEqual(kwargs["warmup_steps"], 12)
            return {"signature": signature}

        with patch.object(study.four, "run_case", side_effect=fake_run):
            study.run_case(8, "terminal")
        self.assertEqual(study.four.experiment.default_config(), original)


if __name__ == "__main__":
    unittest.main()
