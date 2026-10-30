"""Canonical entry point for the expanded physical-noise rank report."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from atlas_deepc.notebook_tools.build_sensor_noise_rank_sweep_notebook import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    main(parser.parse_args().execute)
