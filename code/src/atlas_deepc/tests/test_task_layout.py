"""Task folders preserve historical paths without duplicating experiment data."""

import ast
import json
from pathlib import Path
import unittest

import nbformat
from IPython.core.inputtransformer2 import TransformerManager


ROOT = Path(__file__).resolve().parents[3]
MOVES = ROOT / "docs/task_organization_2026-10-03/moves.json"


class TaskLayoutTest(unittest.TestCase):
    def test_legacy_asset_paths_resolve_to_canonical_task_files(self):
        moves = json.loads(MOVES.read_text())
        for old, new in moves["compatibility_links"].items():
            if new.startswith("experiments/pendulum/"):
                continue
            with self.subTest(path=old):
                legacy, canonical = ROOT / old, ROOT / new
                self.assertTrue(legacy.is_symlink())
                self.assertTrue(canonical.exists())
                self.assertFalse(canonical.is_symlink())
                self.assertTrue(legacy.samefile(canonical))

    def test_moved_notebooks_are_valid_and_find_the_project_root(self):
        moves = json.loads(MOVES.read_text())
        transformer = TransformerManager()
        for old, new in moves["notebooks"].items():
            if new.startswith("notebooks/pendulum/"):
                continue
            with self.subTest(notebook=new):
                path = ROOT / new
                self.assertFalse((ROOT / old).exists())
                self.assertTrue(path.is_file())
                discovered = next(p for p in (path.parent, *path.parent.parents)
                                  if (p / "src/atlas_deepc").is_dir())
                self.assertEqual(discovered, ROOT)
                notebook = nbformat.read(path, as_version=4)
                nbformat.validate(notebook)
                for cell in notebook.cells:
                    if cell.cell_type == "code":
                        ast.parse(transformer.transform_cell(cell.source))

    def test_primary_tasks_are_separate(self):
        for task in ("reacher", "f1tenth"):
            self.assertTrue((ROOT / "notebooks" / task / "README.md").is_file())
            self.assertTrue((ROOT / "experiments" / task).is_dir())
        for task in ("reacher", "f1tenth"):
            self.assertTrue((ROOT / "data" / task).is_dir())
        for category in ("baselines", "ablations", "comparisons"):
            self.assertFalse((ROOT / "notebooks" / category).exists())


if __name__ == "__main__":
    unittest.main()
