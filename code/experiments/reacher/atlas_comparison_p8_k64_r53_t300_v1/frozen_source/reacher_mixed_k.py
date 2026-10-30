"""Development-selected chart counts and ranks, evaluated for 300 control steps."""

import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import shutil
from unittest.mock import patch

from atlas_deepc.benchmarks import reacher_heldout as runner

ROOT = runner.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/four_baselines_500_mixed_k_t300_v1"
VERSION = "four_baselines_500_mixed_k_t300_v1"
SEEDS = runner.SEEDS
CONTROL_STEPS = 300
CASES = ((8, "terminal", 53), (8, "graph", 57), (10, "terminal", 64), (10, "graph", 62))
CHART_COUNTS = {(8, "terminal"): 64, (8, "graph"): 96,
                (10, "terminal"): 80, (10, "graph"): 64}
DEVELOPMENT_ROOTS = {
    (8, "terminal"): ROOT / "experiments/reacher/four_baselines_k64_rank_projection_chordal_v1",
    (8, "graph"): ROOT / "experiments/reacher/large_k_rank_compact_v1/k096",
    (10, "terminal"): ROOT / "experiments/reacher/large_k_rank_compact_v1/k080",
    (10, "graph"): ROOT / "experiments/reacher/four_baselines_k64_rank_projection_chordal_v1",
}
EVALUATION_NOTE = (
    "Chart counts were chosen by the user and ranks selected on development seeds 211--310 "
    "at 320 steps; evaluation uses 300 steps. These 500 seeds were previously evaluated "
    "with K=48, so this is a repeated evaluation, not a new untouched test set")
_case_signature = runner.case_signature


def case_signature(root, p, mode):
    signature = _case_signature(root, p, mode)
    signature["mixed_k_protocol_sha256"] = runner.file_digest(__file__)
    return signature


@contextmanager
def protocol_context():
    # Share the serial runner while leaving historical callers' defaults intact.
    with ExitStack() as stack:
        for name, value in {
            "CASES": CASES, "CHART_COUNTS": CHART_COUNTS,
            "DEVELOPMENT_ROOTS": DEVELOPMENT_ROOTS, "CONTROL_STEPS": CONTROL_STEPS,
            "VERSION": VERSION, "PER_CASE_ATLASES": True,
            "EVALUATION_NOTE": EVALUATION_NOTE, "case_signature": case_signature,
        }.items():
            stack.enter_context(patch.object(runner, name, value))
        yield


def config_for(p, mode):
    with protocol_context():
        return runner.config_for(p, mode)


def atlas_path(root, p, mode):
    with protocol_context():
        return runner.atlas_path(root, p, mode)


def prepare(root=RESULT_ROOT):
    with protocol_context():
        protocol = runner.prepare(root)
    shutil.copy2(__file__, Path(root) / "frozen_source" / Path(__file__).name)
    return protocol


def report(root=RESULT_ROOT):
    with protocol_context():
        return runner.report(root)


def run_all(root=RESULT_ROOT):
    prepare(root)
    with protocol_context():
        return runner.run_all(root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare()
        print("Protocol prepared; no rollouts launched.")
    else:
        run_all()
