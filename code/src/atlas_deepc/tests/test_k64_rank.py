"""K=64 sweep protocol checks without launching rollouts."""

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from atlas_deepc.ablations import k64_rank as study
from atlas_deepc.notebook_tools.build_k64_rank_notebook import full_table


class K64RankTest(unittest.TestCase):
    def test_shared_atlases_are_published_before_workers(self):
        spec = importlib.util.spec_from_file_location(
            "k64_supervisor", study.ROOT / "scripts/run_reacher_k64_rank_sweep.py")
        supervisor = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(supervisor)
        with tempfile.TemporaryDirectory() as directory, patch.object(study, "RANKS", [60, 62]):
            root = Path(directory)
            inventory = {}
            for p in (8, 10):
                source = study.atlas_path(root, p)
                source.parent.mkdir(parents=True)
                source.write_bytes(f"test atlas {p}".encode())
                inventory[str(p)] = {"sha256": study.four.file_digest(source)}
            (root / "atlas_inventory.json").write_text(json.dumps(inventory))
            original_copy = supervisor.shutil.copy2

            def checked_copy(source, temporary):
                self.assertFalse((Path(temporary).parent / "behavioral_atlas.npz").exists())
                return original_copy(source, temporary)

            with patch.object(supervisor.shutil, "copy2", side_effect=checked_copy) as copy:
                supervisor.stage_panel_atlases(root)
                self.assertEqual(copy.call_count, 4)
                supervisor.stage_panel_atlases(root)
                self.assertEqual(copy.call_count, 4)
            for rank in study.RANKS:
                for p, mode in study.four.CASES:
                    atlas = study.panel_directory(root, rank, p, mode).parent / "behavioral_atlas.npz"
                    self.assertEqual(study.four.file_digest(atlas), inventory[str(p)]["sha256"])
            self.assertFalse(list(root.rglob("*.pending.npz")))
            atlas.write_bytes(b"invalid cache")
            with self.assertRaisesRegex(ValueError, "Existing panel atlas checksum mismatch"):
                supervisor.stage_panel_atlases(root)
            self.assertEqual(atlas.read_bytes(), b"invalid cache")

    def test_only_count_and_rank_change(self):
        for rank in (28, 62, 70):
            config = {k: v for k, v in asdict(study.config_for(rank)).items() if k != "seed"}
            for p, mode in study.four.CASES:
                study.validate_config(config, p, mode, rank)
        config["lam_ini"] = 0.
        with self.assertRaises(AssertionError):
            study.validate_config(config, 10, "graph", 70)

    def test_all_172_panels_are_separate(self):
        paths = {study.panel_directory(study.RESULT_ROOT, r, p, mode)
                 for r in study.RANKS for p, mode in study.four.CASES}
        self.assertEqual(len(paths), 172)
        self.assertTrue(all("k64" in str(p) for p in paths))
        self.assertNotIn(study.previous.RESULT_ROOT, [study.RESULT_ROOT])

    def test_partial_latex_does_not_invent_results(self):
        table = full_table([])
        self.assertIn("Partial", table)
        self.assertEqual(table.count("-- & -- & -- & --"), 43)
        self.assertNotIn(r"\textbf", table)


if __name__ == "__main__":
    unittest.main()
