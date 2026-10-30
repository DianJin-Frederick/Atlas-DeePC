"""Regression checks for helpers consolidated during the package reorganization."""

import hashlib
import pickle
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np

from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.common.paths import ROOT, source_path
from atlas_deepc.common.provenance import validate_report_files, validate_report_signature
from atlas_deepc.reporting.noise_videos import NoiseVideoRenderer


class SharedHelpersTests(unittest.TestCase):
    def test_module_paths_and_streaming_hash(self):
        self.assertEqual(source_path("baseline_atlas_deepc.py"),
                         ROOT / "src/atlas_deepc/baselines/baseline_atlas_deepc.py")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "binary.dat"
            data = b"test data" * 200000
            path.write_bytes(data)
            self.assertEqual(file_digest(path), hashlib.sha256(data).hexdigest())

    def test_shared_success_metric(self):
        self.assertTrue(strict_success(true_error_metrics(np.full(320, .005))))
        self.assertFalse(strict_success(true_error_metrics(np.full(320, .01))))
        error = np.full(320, .001)
        error[-1] = .02
        self.assertFalse(strict_success(true_error_metrics(error)))

    def test_dimension_settings_are_separate_but_implementation_shared(self):
        self.assertNotEqual(experiment.result_root(8), experiment.result_root(10))
        with self.assertRaises(ValueError):
            experiment.result_root(9)
        previous = experiment.routed.pipe.P
        with experiment.output_context(8):
            with experiment.output_context(10):
                self.assertEqual(experiment.routed.pipe.P, 10)
            self.assertEqual(experiment.routed.pipe.P, 8)
        self.assertEqual(experiment.routed.pipe.P, previous)

    def test_reports_disclose_code_drift_but_reject_changed_data(self):
        old = {"config": {"rank": 60}, "data_sha256": "a", "source_sha256": {"x": "old"}}
        new = {**old, "source_sha256": {"x": "new"}}
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            validate_report_signature(old, new)
        self.assertEqual(len(captured), 1)
        with self.assertRaises(ValueError):
            validate_report_signature(old, {**new, "data_sha256": "different"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npy"
            path.write_bytes(b"data")
            with self.assertRaises(ValueError):
                validate_report_files({path: "wrong"})

    def test_noise_renderer_is_spawn_picklable(self):
        renderer = NoiseVideoRenderer("atlas_deepc.noise.measurement_norm_noise_rank59",
                                      ((212, .01, .05),), "test")
        restored = pickle.loads(pickle.dumps(renderer.render_one))
        self.assertEqual(restored.__self__, renderer)


if __name__ == "__main__":
    unittest.main()
