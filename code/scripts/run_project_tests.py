"""Run the retained unit/integration tests and save a compact machine-readable report."""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "src"), pattern="test_*.py")
    with args.output.with_suffix(".log").open("w") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    summary = {"passed": result.wasSuccessful(), "tests": result.testsRun,
               "failures": [(str(test), text) for test, text in result.failures],
               "errors": [(str(test), text) for test, text in result.errors],
               "skipped": [(str(test), reason) for test, reason in result.skipped],
               "elapsed_seconds": time.monotonic() - started}
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"failures", "errors"}}))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
