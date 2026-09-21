import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import baseline_osqp_benchmark as benchmark


class FreshBenchmarkTests(unittest.TestCase):
    def test_fresh_paths_and_single_worker_protocol(self):
        initializer = benchmark.experiment.initialize_worker

        def run(seeds, output_dir, workers):
            self.assertEqual(seeds, [237])
            self.assertEqual(workers, 1)
            self.assertFalse(list(Path(output_dir).glob("*/seed*.npz")))
            self.assertIs(benchmark.experiment.initialize_worker, benchmark.initialize_benchmark_worker)
            return {"workers": workers}

        with tempfile.TemporaryDirectory() as directory, patch.object(benchmark.experiment, "run_panel", side_effect=run):
            first = benchmark.run_fresh_panel([237], directory)
            second = benchmark.run_fresh_panel([237], directory)
            self.assertNotEqual(first, second)
            for path in (first, second):
                result = json.loads((path / "summary.json").read_text())
                protocol = result["benchmark_protocol"]
                self.assertEqual(protocol["reused_rollouts"], 0)
                self.assertEqual(protocol["workers"], 1)
                self.assertEqual(protocol["numerical_threads"], 1)
                self.assertEqual(protocol["warmup_steps"], 10)
                self.assertIn("completed_utc", protocol)
        self.assertIs(benchmark.experiment.initialize_worker, initializer)

    def test_parallel_latency_and_invalid_seeds_rejected(self):
        with self.assertRaises(ValueError):
            benchmark.run_fresh_panel([237], workers=4)
        for seeds in ([], [237,237]):
            with self.assertRaises(ValueError):
                benchmark.run_fresh_panel(seeds)


if __name__ == "__main__":
    unittest.main()
