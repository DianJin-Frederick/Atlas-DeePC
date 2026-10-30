"""Read-only protocol checks; no simulator or controller rollouts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from atlas_deepc.ablations import figure_guidance_matched as study
from atlas_deepc.notebook_tools.build_heldout_reacher_notebook import matched_guidance_cells


class MatchedGuidanceTests(unittest.TestCase):
    def test_protocol_and_restoration(self):
        previous = study.runner.SEEDS
        config_for = study.runner.config_for
        with study.protocol_context():
            value = study.runner.config_for(8, "graph")
            self.assertEqual((value.clusters, value.svd_rank, value.steps), (64, 53, 300))
            self.assertEqual(study.runner.SEEDS, (10206, 10349, 10144, 10445, 10330, 10255, 10449))
            self.assertEqual(study.runner.atlas_path(None, 8, "graph"),
                study.baseline.atlas_path(study.baseline.RESULT_ROOT, 8, "terminal"))
        self.assertIs(study.runner.config_for, config_for)
        self.assertIs(study.runner.SEEDS, previous)

    def test_manifest_is_repeatable_and_rejects_changes(self):
        with TemporaryDirectory() as directory:
            path = study.prepare(directory)
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["evaluation_seeds"], list(study.SEEDS))
            self.assertEqual(manifest["graph_cost_method"], "chordal")
            self.assertEqual(manifest["candidate_count"], 10)
            self.assertEqual(manifest["warmup_steps"], 12)
            self.assertEqual(study.prepare(directory), path)
            manifest["config"]["svd_rank"] = 54
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "protocol changed"):
                study.prepare(directory)

    def test_notebook_opt_in_and_pending_table(self):
        import pandas as pd
        cells = matched_guidance_cells()
        self.assertIn("RUN_MATCHED_GUIDANCE = False", cells[1].source)
        for cell in cells:
            if cell.cell_type == "code":
                compile(cell.source, "matched_guidance_cell", "exec")
        with TemporaryDirectory() as directory:
            rows = study.paired_rows(Path(directory))
        self.assertEqual(len(rows), 7)
        self.assertTrue(all(row["guided_success"] is None for row in rows))
        shown = []
        from unittest.mock import Mock
        exec(cells[2].source, {"matched_guidance": Mock(paired_rows=lambda: rows,
            RESULT_ROOT=Path("unused")), "pd": pd, "display": shown.append})
        self.assertEqual(shown[0].shape, (7, 5))


if __name__ == "__main__":
    unittest.main()
