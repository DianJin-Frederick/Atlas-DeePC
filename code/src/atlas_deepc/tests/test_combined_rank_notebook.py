"""The merged notebook is a saved-results report, never a rollout launcher."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import nbformat

from atlas_deepc.notebook_tools import build_combined_rank_notebook as combined


class CombinedRankNotebookTest(unittest.TestCase):
    def test_build_contains_all_counts_and_no_experiment_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = combined.build(Path(directory) / "rank.ipynb")
            notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        markdown = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        for k in (48, 64, 80, 96):
            self.assertIn(f"## K = {k}", markdown)
        code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
        self.assertIn("RUN_EXPERIMENTS = False", code)
        for forbidden in ("run_all(", "run_batch(", "Popen(", "savefig(", "to_json(", "write_text("):
            self.assertNotIn(forbidden, code)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")

    def test_legacy_entry_points_delegate_to_combined_report(self):
        from atlas_deepc.notebook_tools import build_k64_rank_notebook as k64
        from atlas_deepc.notebook_tools import build_projection_chordal_rank_notebook as k48
        with patch.object(combined, "build", return_value=combined.NOTEBOOK) as build:
            self.assertEqual(k64.build(), combined.NOTEBOOK)
            self.assertEqual(k48.build(), combined.NOTEBOOK)
            self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
