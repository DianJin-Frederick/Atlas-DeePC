"""Batch/checkpoint tests with synthetic counts; no simulation or atlas fitting."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import low_k_completion as study


class LowKBatchesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = dict(chart_counts=[1], cases=[[8, "terminal"]],
                             ranks=[28, 29], seeds=[211, 212, 213])
        study.save_json(self.root / "manifest.json", self.manifest)

    def save_panel(self, rank, done, successes):
        path = study.case_path(self.root, 1, 8, "terminal") / f"rank{rank:03d}.json"
        study.save_json(path, dict(signature=dict(
            manifest_sha256=study.runner.file_digest(self.root / "manifest.json"),
            clusters=1, rank=rank, output_dimension=8, reference_mode="terminal"),
            processed_seeds=done, successes=successes, complete=done == 3))
        return path

    def fake_run(self, root, k, rank, p, mode, count):
        panel = study.checkpoint(root, self.manifest, k, rank, p, mode)
        self.save_panel(rank, panel["processed_seeds"] + count, panel["successes"] + count)

    def test_partial_seed_budget(self):
        self.save_panel(28, 2, 1)
        self.assertEqual(study.batch_plan(self.root, 2),
                         [(1, 28, 8, "terminal", 1), (1, 29, 8, "terminal", 1)])
        self.assertEqual(study.batch_plan(self.root, 100),
                         [(1, 28, 8, "terminal", 1), (1, 29, 8, "terminal", 3)])

    def test_select_prune_and_resume(self):
        self.save_panel(28, 3, 2)
        self.save_panel(29, 2, 2)
        self.assertFalse(study.finalize_completed_cases(self.root)["complete"])
        self.save_panel(29, 3, 2)
        result = study.finalize_completed_cases(self.root)
        self.assertTrue(result["complete"])
        self.assertEqual(result["completed_rollouts"], 6)
        self.assertEqual(result["selected"][0]["rank"], 28)
        self.assertFalse((study.case_path(self.root, 1, 8, "terminal") / "rank029.json").exists())
        self.assertEqual(study.batch_plan(self.root, 100), [])
        self.assertEqual(study.finalize_completed_cases(self.root), result)

    def test_corrupt_checkpoint_rejected(self):
        path = self.save_panel(28, 1, 1)
        data = json.loads(path.read_text())
        data["signature"]["manifest_sha256"] = "wrong"
        study.save_json(path, data)
        with self.assertRaises(ValueError):
            study.batch_plan(self.root, 1)

    def test_bounded_batches_resume_without_repeating_seeds(self):
        with patch.object(study, "prepare"), patch.object(study, "run_panel", side_effect=self.fake_run):
            first = study.run_batch(self.root, batch_size=2)
            self.assertEqual(first["batch_completed_rollouts"], 2)
            self.assertFalse(first["complete"])
            second = study.run_batch(self.root, batch_size=10)
            self.assertTrue(second["complete"])
            self.assertEqual(second["batch_completed_rollouts"], 4)
            third = study.run_batch(self.root, batch_size=10)
            self.assertEqual(third["batch_completed_rollouts"], 0)

    def test_interruption_records_stopped_status(self):
        def interrupted(*args):
            self.save_panel(28, 1, 1)
            raise KeyboardInterrupt()
        with patch.object(study, "prepare"), patch.object(study, "run_panel", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                study.run_batch(self.root, batch_size=3)
        status = json.loads((self.root / "run_status.json").read_text())
        self.assertEqual(status["status"], "stopped")
        self.assertEqual(study.batch_plan(self.root, 2), [(1, 28, 8, "terminal", 2)])

    def test_positive_integer_settings_required(self):
        for value in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                study.run_batch(self.root, batch_size=value)
            with self.assertRaises(ValueError):
                study.run_batch(self.root, workers=value)

    def test_worker_count_is_not_part_of_frozen_protocol(self):
        root = self.root / "prepared"
        atlas = dict(labels=np.zeros(37800, dtype=int),
                     feature_mean=np.zeros(120), feature_scale=np.ones(120))
        with patch.object(study, "code_hashes", return_value={}), \
                patch.object(study.runner, "file_digest", return_value="mock-digest"), \
                patch.object(study.runner.experiment, "prepare_atlas", return_value=(root / "atlas.npz", {})), \
                patch.object(study.runner.experiment.routed, "load_atlas", return_value=atlas):
            first = study.prepare(root, cases=((8, "terminal"),), counts=(1,), workers=1)
            second = study.prepare(root, cases=((8, "terminal"),), counts=(1,), workers=4)
        self.assertEqual(first, second)
        self.assertNotIn("workers", first)

    def test_duplicate_run_is_locked_out(self):
        with study.exclusive_lock(self.root / ".run.lock"), patch.object(study, "prepare") as prepare:
            with self.assertRaisesRegex(RuntimeError, "already running"):
                study.run_batch(self.root, batch_size=1)
            prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
