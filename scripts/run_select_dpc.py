"""Launch the optional comparison in its separate, pinned environment."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from comparison_methods.select_dpc.dependency import require_installation


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--our-data", action="store_true")
    args, remaining = parser.parse_known_args()
    python = require_installation()
    module = "comparison_methods.select_dpc.our_data" if args.our_data else "comparison_methods.select_dpc.upstream"
    return subprocess.call([str(python), "-u", "-m", module, *remaining], cwd=ROOT,
                           env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONNOUSERSITE="1"))


if __name__ == "__main__":
    raise SystemExit(main())
