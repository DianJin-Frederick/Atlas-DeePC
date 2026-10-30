"""Dataset-adapter and protocol tests; no controller solves or simulations."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from atlas_deepc.benchmarks import select_dpc_our_data as run

sys.path.insert(0, str(run.base.UPSTREAM))
from select_deepc.hankel_generation import HankelMatrixGenerator


class OurDataTests(unittest.TestCase):
    def source(self):
        rng = np.random.default_rng(42)
        u = rng.uniform(-1, 1, (2, 20, 2))
        y = rng.normal(size=(2, 20, 10))
        source = np.vstack([run.recovery.windows(u, 12), run.recovery.windows(y, 12)])
        return source, u, y

    def test_reconstruction_order_and_episode_boundaries(self):
        source, u, y = self.source()
        data = run.dataset_from_hankel(source, episode_length=20)
        self.assertEqual(len(data.dataset), 2)
        for i, episode in enumerate(data.dataset):
            np.testing.assert_array_equal(episode.input_trajectory, u[i])
            np.testing.assert_array_equal(episode.state_trajectory, y[i][:, run.base.TO_BASELINE])
        hu, hy = HankelMatrixGenerator(2, 15).generate_hankel_matrices(data)
        self.assertEqual(hu.shape, (34, 8))  # Two episodes, four valid windows each.
        self.assertEqual(hy.shape, (136, 8))
        np.testing.assert_array_equal(hu, run.recovery.windows(u, 17))
        np.testing.assert_array_equal(hy, run.recovery.windows(y[:, :, run.base.TO_BASELINE], 17))
        np.testing.assert_array_equal(hu[:, 3], u[0, 3:20].ravel())
        np.testing.assert_array_equal(hu[:, 4], u[1, :17].ravel())

    def test_corrupt_overlap_and_nonfinite_data_rejected(self):
        source, _, _ = self.source()
        for value in (1234., float("nan")):
            corrupt = source.copy()
            corrupt[2, 0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                run.dataset_from_hankel(corrupt, episode_length=20)

    def test_panel_and_batch_validation_before_result_writes(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            for extra in (["--steps", "99"], ["--seed-start", "10001"],
                          ["--block-stop", "311"], ["--max-new-seeds", "0"]):
                cli = run.parser().parse_args(["--output-dir", str(output), *extra])
                with self.subTest(extra=extra), self.assertRaises(ValueError):
                    run.main(cli)
                self.assertFalse(output.exists())

    def test_preflight_does_not_write_or_run(self):
        from types import SimpleNamespace
        args = SimpleNamespace(trajectory_data=SimpleNamespace(dataset=[None] * 200))
        audit = {"max_absolute_difference": 0.}
        signature = {"library_columns": 36800, "data_sha256": "synthetic"}
        with TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            cli = run.parser().parse_args(["--preflight", "--steps", "100", "--output-dir", str(output)])
            with patch.object(run, "prepare", return_value=(None, args, None, audit)), \
                 patch.object(run, "signature_for", return_value=signature), \
                 patch.object(run.base, "run_block") as block, redirect_stdout(io.StringIO()) as log:
                run.main(cli)
            block.assert_not_called()
            self.assertFalse(output.exists())
            self.assertEqual(json.loads(log.getvalue())["rollouts_started"], 0)
            self.assertEqual(json.loads(log.getvalue())["steps"], 100)

    def test_run_delegates_replaced_data_and_guards_resume(self):
        import fcntl
        dc, args, scheduler = object(), object(), object()
        signature = {"data_sha256": "our-data", "steps": 100}
        with TemporaryDirectory() as temp:
            output = Path(temp) / "results"
            cli = run.parser().parse_args(["--steps", "100", "--output-dir", str(output), "--max-new-seeds", "25"])
            with patch.object(run, "prepare", return_value=(dc, args, scheduler, {})), \
                 patch.object(run, "signature_for", return_value=signature), \
                 patch.object(run.base, "run_block") as block, redirect_stdout(io.StringIO()):
                run.main(cli)
                block.assert_called_once_with(dc, args, scheduler, cli, output.resolve(), signature)
                run.main(cli)
                self.assertEqual(block.call_count, 2)
                with (output / "run.lock").open("r") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with self.assertRaises(BlockingIOError):
                        run.main(cli)
                signature["data_sha256"] = "changed-data"
                with self.assertRaisesRegex(ValueError, "Incompatible run manifest"):
                    run.main(cli)
                self.assertEqual(block.call_count, 2)


if __name__ == "__main__":
    unittest.main()
