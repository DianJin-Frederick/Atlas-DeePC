
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

from dataclasses import asdict, replace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.ablations import position_only_q_ablation as experiment
class PositionOnlyQTests(unittest.TestCase):
    def test_only_two_weights_change(self):
        before = asdict(experiment.noise.experiment_config())
        after = asdict(experiment.experiment_config())
        self.assertEqual({k for k in before if before[k] != after[k]},
                         {"q_data_goal_joint", "q_dq"})
        self.assertEqual(after["lam_g"], 0)

    def test_actual_controller_weights_and_truth_logging(self):
        for ratio in (0., .05):
            with self.subTest(ratio=ratio), tempfile.TemporaryDirectory() as folder, threadpool_limits(limits=1):
                experiment.initialize_worker(ratio)
                experiment.noise._controller.config = replace(experiment.noise._controller.config, steps=8)
                original = experiment.noise.pipe.DeePCController
                seen = []

                def observe(*args, **kwargs):
                    controller = original(*args, **kwargs)
                    seen.append(controller)
                    return controller

                with patch.object(experiment.noise.pipe, "DeePCController", observe):
                    row = experiment.run_seed(212, folder)
                self.assertEqual(len(seen), 1)
                np.testing.assert_array_equal(seen[0].Q, np.diag([0]*6 + [100000]*2 + [0]*2))
                self.assertFalse(hasattr(seen[0], "Q_terminal"))
                self.assertEqual(seen[0].lam_ini, 100000)
                self.assertEqual(row["augmented_output_dimension"], 10)
                with np.load(row["result_path"], allow_pickle=True) as trace:
                    self.assertEqual(trace["y"].shape, (8, 10))
                    np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-row["exact_target_xy"], axis=1))
                    np.testing.assert_array_equal(trace["sensor_true"][-8:], trace["y"])


if __name__ == "__main__":
    unittest.main()
