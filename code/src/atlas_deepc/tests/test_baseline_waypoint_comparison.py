
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import baseline_waypoint_comparison as experiment
class BaselineWaypointTests(unittest.TestCase):
    def test_only_nominal_lambda_default_changes(self):
        old = asdict(experiment.routed.RoutedDeePCConfig())
        new = asdict(experiment.baseline.default_config())
        self.assertEqual({key for key in old if old[key] != new[key]}, {"lam_g"})
        self.assertEqual(new["lam_g"], 0)
        self.assertEqual(old["lam_g"], .05)

    def test_paired_references_and_actual_qp(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment.initialize_worker(directory)
            controller = experiment._controller
            controller.config = replace(controller.config, steps=12)
            seen = []
            constructor = experiment.pipe.DeePCController

            def checked_constructor(*args, **kwargs):
                result = constructor(*args, **kwargs)
                self.assertEqual(result.lam_g, 0)
                np.testing.assert_array_equal(np.diag(result.Q), [5000]*4+[20]*2+[100000]*2+[5000]*2)
                seen.append(result)
                return result

            with patch.object(experiment.pipe, "DeePCController", checked_constructor):
                rows = experiment.run_pair(237)
            self.assertEqual(len(seen), 2)
            self.assertEqual({row["reference_mode"] for row in rows}, {"terminal", "graph"})
            self.assertTrue(all(row["lam_g"] == 0 and row["matrix_geometry"] == "linear" for row in rows))
            with np.load(Path(directory)/"graph/seed237_control_results.npz") as graph:
                with np.load(Path(directory)/"terminal/seed237_control_results.npz") as terminal:
                    self.assertFalse(np.allclose(graph["ref"], terminal["ref"]))
            self.assertEqual(len(experiment.run_pair(237)), 2)
            with self.assertRaises(ValueError):
                controller.run(211, reference_mode="invalid")


if __name__ == "__main__":
    unittest.main()
