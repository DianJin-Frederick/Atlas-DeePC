"""Select-DPC development variant: our offline data, seeds 211--310.

The upstream algorithm and existing rollout/batch implementation are reused.
Only the offline trajectories are replaced. Rollout length defaults to 300.
--preflight never starts a rollout.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from comparison_methods.select_dpc import upstream as base
from comparison_methods.common import reacher as recovery

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = base.ROOT
DATA = ROOT / "data/reacher/select_iid_offline_hankel.npy"
OUTPUT = ROOT / "experiments/reacher/select_dpc_our_data_clarabel_211_310_t300_v1"
SEEDS = list(range(211, 311))
STEPS = 300


def dataset_from_hankel(hankel, episode_length=200):
    """Recover episodes, remove the two derived features, reorder physical outputs.

    The source has post-action y paired with u, already in physical fingertip
    coordinates. Do not apply the upstream raw-observation fix_dataset again.
    The upstream controller constructs L=17 windows separately per episode.
    """
    from select_deepc.deepc_dataclasses import TrajectoryData, TrajectoryDataSet

    inputs, outputs = recovery.recover_episodes(hankel, episode_length)
    return TrajectoryDataSet([
        TrajectoryData(state_trajectory=y[:, base.TO_BASELINE].copy(),
                       input_trajectory=u.copy())
        for u, y in zip(inputs, outputs)
    ], dataset_name="our_reacher_iid_physical8")


def prepare():
    dc, args, scheduler = base.load_upstream()
    source = np.load(DATA, mmap_mode="r", allow_pickle=False)
    if source.shape != (144, 37800):
        raise ValueError(f"Expected 200 episodes of 200 samples, got {source.shape}")
    dataset = dataset_from_hankel(source)
    audit = base.audit_baseline_data(dataset)
    if not audit["matches_baseline"]:
        raise ValueError(f"Our offline data round-trip failed: {audit}")
    audit["note"] = "All L=12 source windows round-trip; Select-DPC rebuilds L=17 within each episode."
    if "CLARABEL" not in base.cp.installed_solvers():
        raise RuntimeError("Clarabel is unavailable in this interpreter")
    return dc, replace(args, trajectory_data=dataset), scheduler, audit


def signature_for(args, audit, cli):
    costs = args.controller_costs
    constraints = args.controller_constraints
    return {
        "version": "select_dpc_our_data_development_v1",
        "trace_version": base.VERSION,
        "runner_sha256": base.digest(__file__),
        "dependencies_sha256": {str(p.relative_to(ROOT)): base.digest(p) for p in (
            Path(base.__file__), Path(recovery.__file__),
            ROOT / "src/atlas_deepc/common/metrics.py",
            ROOT / "src/atlas_deepc/control/expert_pipeline_search.py")},
        "upstream_commit": base.COMMIT,
        "upstream_files": {str(p.relative_to(base.UPSTREAM)): base.digest(p) for p in (
            *sorted((base.UPSTREAM / "select_deepc").glob("*.py")),
            base.UPSTREAM / "reacher_simulation.ipynb")},
        "data_source": str(DATA.relative_to(ROOT)),
        "data_sha256": base.digest(DATA), "data_audit": audit,
        "offline_episodes": 200, "offline_episode_samples": 200,
        "offline_input_range": [-1., 1.], "offline_alignment": "post-action y paired with u",
        "offline_conversion": "Recover L=12 episodes; retain physical 8 outputs; reorder to native; rebuild L=17 per episode",
        "library_columns": 36800, "selected_columns": 200, "refinement_iterations": 5,
        "p": 8, "m": 2, "T_ini": 2, "T_f": 15, "svd": False,
        "seed_start": cli.seed_start, "seed_stop": cli.seed_stop, "steps": cli.steps,
        "seed_role": "development", "environment": "Reacher-v4", "warmup_steps": 0,
        "reference": "upstream DefaultReacherSetpoint; final Cartesian target, no Atlas references",
        "solver": "CLARABEL", "clarabel_options": base.CLARABEL_OPTIONS,
        "max_iter": 6000, "warm_start": False, "solver_tolerances": "Clarabel defaults",
        "workers": 1, "numerical_threads": 1,
        "Q_native": costs.Q.tolist(), "R": costs.R.tolist(),
        "lambda_g_l1": costs.regularizer_cost_g_1,
        "lambda_projection": costs.regularizer_cost_g_pi, "lambda_slack_l1": costs.slack_cost,
        "constraints": {name: getattr(constraints, name).tolist() for name in ("A_u", "b_u", "A_y", "b_y")},
        "saved_output_order": ["cos_q1", "cos_q2", "sin_q1", "sin_q2", "dq1", "dq2", "fingertip_x", "fingertip_y"],
        "comparison_caveats": [
            "Development variant using our +/-1 exploration dataset, not authors' +/-0.1 dataset",
            "Original Reacher-v4, no physical warmup, horizon 15, original cost and constraints retained",
            "Development seeds 211--310, not held-out paper evaluation seeds",
            "Compare unweighted metrics or rescore methods with common weights"],
        "packages": {p: importlib.metadata.version(p) for p in (
            "numpy", "scipy", "cvxpy", "clarabel", "gymnasium", "mujoco", "torch")},
    }


def main(cli):
    if (cli.seed_start, cli.seed_stop) != (211, 310):
        raise ValueError("This development panel is fixed at seeds 211--310; use block options for subsets")
    if cli.steps < 100:
        raise ValueError("Use at least 100 steps so the last-100 success window remains valid")
    block, limit = base.requested_block(cli)
    with threadpool_limits(limits=1):
        if cli.preflight:
            _, args, _, audit = prepare()
            signature = signature_for(args, audit, cli)
            print(json.dumps({"preflight": "passed", "rollouts_started": 0,
                "episodes": len(args.trajectory_data.dataset),
                "library_columns": signature["library_columns"], "p": 8, "T_ini": 2, "T_f": 15,
                "seed_start": 211, "seed_stop": 310, "steps": cli.steps,
                "data_sha256": signature["data_sha256"],
                "source_round_trip_max_error": audit["max_absolute_difference"]}, indent=2))
            return
        output = (cli.output_dir or OUTPUT.with_name(
            f"select_dpc_our_data_clarabel_211_310_t{cli.steps}_v1")).resolve()
        output.mkdir(parents=True, exist_ok=True)
        with (output / "run.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            dc, args, scheduler, audit = prepare()
            signature = signature_for(args, audit, cli)
            manifest = output / "manifest.json"
            if manifest.exists():
                if json.loads(manifest.read_text()) != signature:
                    raise ValueError("Incompatible run manifest; use a fresh output directory")
            else:
                if list(output.glob("seed*_control_results.npz")):
                    raise ValueError("Unversioned rollout cache")
                base.write_json(manifest, signature)
            base.write_json(output / "session.json", {"pid": os.getpid(), "started_unix": time.time(),
                "python": sys.executable, "platform": platform.platform(),
                "block_seeds": block, "max_new_seeds": limit})
            print("Our offline data: 200 episodes, 36,800 L=17 columns; unchanged 200-column/five-refinement Select-DPC.", flush=True)
            base.run_block(dc, args, scheduler, cli, output, signature)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--seed-start", type=int, default=211)
    result.add_argument("--seed-stop", type=int, default=310)
    result.add_argument("--steps", type=int, default=STEPS)
    result.add_argument("--block-start", type=int)
    result.add_argument("--block-stop", type=int)
    result.add_argument("--max-new-seeds", type=int)
    result.add_argument("--output-dir", type=Path, help="Defaults to a separate directory for the chosen --steps")
    result.add_argument("--preflight", action="store_true", help="Validate without rollouts or result writes")
    return result


if __name__ == "__main__":
    main(parser().parse_args())
