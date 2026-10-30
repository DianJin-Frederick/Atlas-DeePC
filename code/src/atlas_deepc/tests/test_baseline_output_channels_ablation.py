
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import baseline_output_channels_ablation as experiment
class OutputChannelAblationTests(unittest.TestCase):
    def tearDown(self):
        experiment.pipe.P = 10

    def test_hankel_removal_keeps_every_other_row(self):
        H = np.arange(144 * 3).reshape(144, 3)
        reduced = experiment.remove_joint_sum_channels(H)
        np.testing.assert_array_equal(reduced[:24], H[:24])
        np.testing.assert_array_equal(reduced[24:].reshape(12, 8, 3), H[24:].reshape(12, 10, 3)[:, :8])

    def test_eight_outputs_do_not_depend_on_derived_channels(self):
        experiment.pipe.P = 8
        H = np.random.default_rng(1).normal(size=(144, 20))
        changed = H.copy()
        changed[24:].reshape(12, 10, 20)[:, 8:] += 1000
        np.testing.assert_array_equal(experiment.remove_joint_sum_channels(H),
                                      experiment.remove_joint_sum_channels(changed))

    def test_actual_qp_and_online_dimensions(self):
        constructor = experiment.pipe.DeePCController
        # Test fresh artifacts rather than the version-locked research results.
        with tempfile.TemporaryDirectory() as directory, patch.object(
                experiment, "RESULT_ROOT", Path(directory) / "artifacts"):
            experiment.prepare()
            experiment.initialize_worker()
            for dimension in (10, 8, 10):
                experiment.pipe.P = dimension
                controller = experiment._controllers[dimension]
                original_config = controller.config
                controller.config = replace(original_config, steps=8)
                seen = []

                def checked_constructor(*args, **kwargs):
                    result = constructor(*args, **kwargs)
                    expected = [5000]*4 + [20]*2 + [100000]*2
                    if dimension == 10:
                        expected += [5000]*2
                    np.testing.assert_array_equal(result.Q, np.diag(expected))
                    seen.append(result)
                    return result

                try:
                    path = Path(directory) / f"output{dimension}.npz"
                    with patch.object(experiment.pipe, "DeePCController", checked_constructor):
                        row = controller.run(211, path)
                    row = experiment.finalize_trace(path, dimension, 211, steps=8)
                    assert seen
                    self.assertEqual(len(row["exact_target_xy"]), 2)
                    self.assertEqual(row["lam_g"], original_config.lam_g)
                    self.assertEqual(row["candidate_count"], 10)
                    with np.load(path, allow_pickle=True) as trace:
                        self.assertEqual(trace["y"].shape, (8, dimension))
                        self.assertEqual(trace["ref"].shape, (8, dimension))
                        self.assertEqual(json.loads(trace["summary"].item())["matrix_geometry"], "linear")
                finally:
                    controller.config = original_config


if __name__ == "__main__":
    unittest.main()
