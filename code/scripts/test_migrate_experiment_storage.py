"""Exercise migration safety using temporary directories, never real results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from migrate_experiment_storage import relocate, digest


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source, self.target, self.journal = root / "source.npz", root / "ssd/data.npz", root / "journal"
        self.source.write_bytes(b"saved experimental results")

    def move(self):
        relocate(self.source, self.target, str(self.target), self.journal, "experiments/test.npz")

    def test_verified_move_and_resume(self):
        expected = digest(self.source)
        self.move()
        self.move()
        self.assertTrue(self.source.is_symlink())
        self.assertEqual(digest(self.source), expected)
        self.assertEqual(json.loads(self.journal.read_text())["sha256"], expected)

    def test_conflict_retains_original(self):
        self.target.parent.mkdir()
        self.target.write_bytes(b"different data")
        with self.assertRaisesRegex(RuntimeError, "conflict"):
            self.move()
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(self.target.read_bytes(), b"different data")

    def test_journal_failure_retains_original(self):
        with patch("migrate_experiment_storage.append_record", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.move()
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(digest(self.source), digest(self.target))
        self.move()
        self.assertTrue(self.source.is_symlink())

    def test_source_change_retains_original(self):
        import shutil
        original = shutil.copyfileobj
        def change(source, target, size):
            original(source, target, size)
            self.source.write_bytes(b"new results")
        with patch("migrate_experiment_storage.shutil.copyfileobj", side_effect=change):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                self.move()
        self.assertFalse(self.source.is_symlink())
        self.assertEqual(self.source.read_bytes(), b"new results")


if __name__ == "__main__":
    unittest.main()
