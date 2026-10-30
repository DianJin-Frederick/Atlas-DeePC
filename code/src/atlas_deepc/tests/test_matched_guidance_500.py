"""Matched guidance protocol/statistics checks without simulations."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from atlas_deepc.ablations import matched_guidance_500 as study
from atlas_deepc.notebook_tools.build_effect_of_graph_guidance_notebook import matched_guidance_500_cells, build


class MatchedGuidance500Test(unittest.TestCase):
    def test_fixed_configuration_and_context_restoration(self):
        previous = study.runner.SEEDS
        config_for = study.runner.config_for
        figure_seeds = study.matched.SEEDS
        with study.protocol_context():
            value = study.runner.config_for(8, "graph")
            self.assertEqual((value.clusters, value.svd_rank, value.steps), (64, 53, 300))
            self.assertEqual(study.runner.SEEDS, tuple(range(10001, 10501)))
            self.assertEqual(study.runner.atlas_path(None, 8, "graph"),
                             study.baseline.atlas_path(study.baseline.RESULT_ROOT, 8, "terminal"))
        self.assertIs(study.runner.config_for, config_for)
        self.assertIs(study.runner.SEEDS, previous)
        self.assertIs(study.matched.SEEDS, figure_seeds)

    def test_paired_counts_difference_and_exact_test(self):
        outcomes = [(True, True)] * 2 + [(False, True)] * 6 + [(True, False)] + [(False, False)]
        rows = [dict(seed=i, unguided_success=a, guided_success=b) for i, (a, b) in enumerate(outcomes)]
        stats = study.statistics(rows, tuple(range(10)))
        self.assertEqual(stats["paired_counts"], dict(both_success=2, repaired=6, regressed=1, both_failed=1))
        self.assertEqual(stats["unguided_success_rate"], 30)
        self.assertEqual(stats["guided_success_rate"], 80)
        self.assertEqual(stats["improvement_percentage_points"], 50)
        self.assertEqual(stats["exact_mcnemar_pvalue"], .125)
        rows[-1]["guided_success"] = None
        partial = study.statistics(rows, tuple(range(10)))
        self.assertFalse(partial["complete"])
        self.assertIsNone(partial["exact_mcnemar_pvalue"])

    def test_no_discordance_and_duplicate_seeds(self):
        rows = [dict(seed=1, unguided_success=True, guided_success=True)]
        self.assertEqual(study.statistics(rows, (1,))["exact_mcnemar_pvalue"], 1.)
        with self.assertRaises(ValueError):
            study.statistics(rows * 2, (1, 2))

    def test_manifest_is_repeatable_and_freezes_full_panel(self):
        with TemporaryDirectory() as directory:
            path = study.prepare(directory)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["evaluation_seeds"], list(study.SEEDS))
            self.assertEqual(manifest["config"]["clusters"], 64)
            self.assertEqual(manifest["config"]["svd_rank"], 53)
            self.assertEqual(manifest["config"]["steps"], 300)
            self.assertEqual(manifest["candidate_count"], 10)
            self.assertEqual(manifest["timing_workers"], 1)
            self.assertEqual(study.prepare(directory), path)
            manifest["config"]["svd_rank"] = 54
            study.runner.save_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                study.prepare(directory)

    def test_pairing_rejects_different_initial_state(self):
        with TemporaryDirectory() as directory, patch.object(study, "SEEDS", (10001,)):
            root = Path(directory)
            (root / "output8/graph").mkdir(parents=True)
            (root / "output8/graph/seed10001_control_results.npz").touch()
            study.runner.save_json(root / "output8/graph/manifest.json", dict(
                paired_manifest_sha256=study.runner.file_digest(study.PAIRED_DIRECTORY / "manifest.json"),
                evaluation_seeds=[10001]))
            outcome = dict(success=True, final100_mm=1., rollout_rmse_mm=2.)
            with patch.object(study, "read_outcome", side_effect=[
                    (outcome, np.zeros(120), np.zeros(2)),
                    (outcome, np.ones(120), np.zeros(2))]):
                with self.assertRaises(AssertionError):
                    study.paired_rows(root)

    def test_notebook_is_opt_in(self):
        cells = matched_guidance_500_cells()
        self.assertIn("RUN_MATCHED_GUIDANCE_500 = False", cells[1].source)
        for cell in cells:
            if cell.cell_type == "code":
                compile(cell.source, "guidance500_cell", "exec")

    def test_standalone_notebook_has_setup_and_no_baseline_launch(self):
        import nbformat
        with TemporaryDirectory() as directory:
            path = build(Path(directory) / "guidance.ipynb")
            notebook = nbformat.read(path, as_version=4)
        self.assertEqual(len(notebook.cells), 4)
        self.assertIn("import os, sys", notebook.cells[1].source)
        self.assertIn("import pandas as pd", notebook.cells[1].source)
        self.assertIn("RUN_MATCHED_GUIDANCE_500 = False", notebook.cells[2].source)
        self.assertNotIn("reacher_mixed_k", "\n".join(c.source for c in notebook.cells))
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")


if __name__ == "__main__":
    unittest.main()
