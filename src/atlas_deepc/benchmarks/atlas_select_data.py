"""Atlas-DeePC on the authors' released Select-DPC IID Reacher episodes.

Only the data source changes relative to the p=8, K=64/rank-53, unguided
Atlas comparison. No upstream controller or notebook experiment is executed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
from threadpoolctl import threadpool_limits

from comparison_methods.select_dpc import dependency
from atlas_deepc.atlas import candidate_goal_routing_experiment as goals
from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment
from atlas_deepc.benchmarks import atlas_reacher_comparison as comparison
from atlas_deepc.benchmarks.reacher_heldout import LoggedSelector, save_json
from atlas_deepc.common.files import file_digest
from atlas_deepc.common.metrics import strict_success, true_error_metrics
from atlas_deepc.common.timing import summarize_timing

ROOT = experiment.baseline.ROOT
DATA_ROOT = Path(os.environ.get("ATLAS_SELECT_DATA_ROOT", Path.home() / "Projects/atlas_select_dpc_data"))
OUTPUT = DATA_ROOT / "experiments/reacher/atlas_select_data_p8_k64_r53_dev100_t300_v1"
SEEDS = tuple(range(211, 311))
STEPS = 300
LENGTH = 12
VERSION = "atlas_select_data_v1"
CHANNELS = ["cos_q1", "cos_q2", "sin_q1", "sin_q2", "dq1", "dq2", "fingertip_x", "fingertip_y"]


def config(steps=STEPS):
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 6:
        raise ValueError("Use at least 6 steps; paper/development runs use 300")
    return replace(experiment.default_config(), clusters=64, svd_rank=53, steps=steps)


@contextmanager
def run_lock(output):
    output = Path(output)
    # Never silently create a missing external-drive mount.
    if str(output).startswith("/Volumes/") and not Path(*output.parts[:3]).is_mount():
        raise RuntimeError(f"External volume is not mounted: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another worker is using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def episode_hankel(inputs, observations):
    """Match upstream fix_dataset, then reorder into Atlas' output convention.

    Rows are post-action observations, aligned with the input that produced them.
    The two sum-angle channels are storage compatibility only; p=8 drops them.
    Windows never cross an episode boundary. No simulated or rescaled samples.
    """
    u = np.asarray(inputs, dtype=float)
    raw = np.asarray(observations, dtype=float)
    if u.ndim != 2 or u.shape[1] != 2 or raw.shape != (len(u), 11) or len(u) < LENGTH:
        raise ValueError("Expected aligned (N,2) inputs and (N,11) Reacher observations")
    if not np.isfinite(u).all() or not np.isfinite(raw).all():
        raise ValueError("Nonfinite offline samples")
    if np.max(np.abs(u)) > .1 + 1e-12:
        raise ValueError("Expected original IID inputs in [-0.1,0.1]")
    target = raw[0, 4:6].copy()
    np.testing.assert_allclose(raw[:, 4:6], np.broadcast_to(target, (len(raw), 2)), atol=1e-12, rtol=0)
    y = raw[:, [0, 1, 2, 3, 6, 7, 4, 5]].copy()
    y[:, 6:8] += raw[:, 8:10]  # goal + fingertip-minus-goal = absolute fingertip.
    y10 = np.column_stack([y, y[:, 0]*y[:, 1]-y[:, 2]*y[:, 3],
                          y[:, 2]*y[:, 1]+y[:, 0]*y[:, 3]])
    hu = np.stack([u[t:t+LENGTH].ravel() for t in range(len(u)-LENGTH+1)], axis=1)
    hy = np.stack([y10[t:t+LENGTH].ravel() for t in range(len(u)-LENGTH+1)], axis=1)
    return np.vstack([hu, hy]), target


def prepare_data(output, source=None):
    """Read the pinned upstream CSVs without importing its optional dependencies."""
    source = dependency.source_dir() if source is None else Path(source)
    revision = dependency.verify_source(source)
    folder = source / "data/reacher/dataset_iid"
    states = sorted(folder.glob("*_random_data_state.csv"))  # upstream lexicographic order
    ids = [int(p.name.split("_")[0]) for p in states]
    if sorted(ids) != list(range(200)):
        raise ValueError("Expected exactly the authors' 200 IID episodes, IDs 0--199")
    pairs = [(p.with_name(f"{i}_random_data_input.csv"), p) for i, p in zip(ids, states)]
    hashes = {p.name: file_digest(p) for pair in pairs for p in pair}
    signature = dict(upstream_commit=revision, upstream_url=dependency.URL,
                     files_sha256=hashes, episode_ids=ids, episodes=200,
                     samples_per_episode=200, windows_per_episode=189, length=LENGTH,
                     storage_shape=[144, 37800], controller_shape=[120, 37800],
                     output_channels=CHANNELS, converter_sha256=file_digest(__file__),
                     alignment="input[t], post-action output[t]; no cross-episode windows",
                     source_target_policy="recorded raw observation columns 4:6, NOT Atlas reset seeds")
    output = Path(output)
    manifest_path = output / "dataset_manifest.json"
    data_path = output / "select_original_hankel_l12.npy"
    metadata_path = output / "source_episodes.npz"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved["signature"] != signature or any(not p.is_file() for p in (data_path, metadata_path)):
            raise ValueError("Dataset provenance changed; use a new output directory")
        if saved["hankel_sha256"] != file_digest(data_path) or saved["metadata_sha256"] != file_digest(metadata_path):
            raise ValueError("Derived dataset checksum mismatch")
        return data_path, saved
    if data_path.exists() or metadata_path.exists():
        raise ValueError("Unversioned derived data; use a fresh output directory")
    blocks, targets = [], []
    for inputs, states_path in pairs:
        u = np.loadtxt(inputs, delimiter=",")
        raw = np.loadtxt(states_path, delimiter=",")
        if u.shape != (200, 2) or raw.shape != (200, 11):
            raise ValueError(f"Unexpected episode shape: {states_path}")
        block, target = episode_hankel(u, raw)
        blocks.append(block)
        targets.append(target)
    output.mkdir(parents=True, exist_ok=True)
    pending = data_path.with_suffix(".pending.npy")
    np.save(pending, np.concatenate(blocks, axis=1))
    pending.replace(data_path)
    pending = metadata_path.with_suffix(".pending.npz")
    np.savez_compressed(pending, episode_ids=ids, targets=targets)
    pending.replace(metadata_path)
    saved = dict(signature=signature, hankel_sha256=file_digest(data_path),
                 metadata_sha256=file_digest(metadata_path))
    save_json(manifest_path, saved)
    return data_path, saved


class OriginalDataGoal(goals.FixedNearestGoalPolicy):
    """Atlas' fixed-nearest policy, with the actual upstream episode provenance."""
    def __init__(self, H, atlas, cfg, source_targets, episode_ids):
        self.atlas, self.config, self.candidate_count = atlas, cfg, 1
        self.outputs = H[24:].reshape(LENGTH, 8, H.shape[1])[1].T
        self.distances = self.predecessors = None
        self.source_targets = np.asarray(source_targets)
        self.episode_ids = np.asarray(episode_ids)
        if self.source_targets.shape != (len(self.episode_ids), 2) or H.shape[1] != len(self.episode_ids)*189:
            raise ValueError("Episode provenance does not match the Hankel columns")

    def prepare(self, target):
        super().prepare(target)
        reference, metadata, chart, cost = self.fixed_selection
        episode = int(self.indices[0]) // 189
        metadata.update(data_goal_source_seed=int(self.episode_ids[episode]),
                        data_goal_source_episode_id=int(self.episode_ids[episode]),
                        data_goal_source_dataset="Select-DPC original IID")
        self.fixed_selection = reference, metadata, chart, cost


def prepare(output, steps=STEPS):
    """Called under run_lock. Fit a NEW atlas; never reuse our-data charts."""
    output = Path(output)
    cfg = config(steps)
    data_path, dataset = prepare_data(output)
    atlas_path, atlas_manifest = experiment.prepare_atlas(cfg, output / "atlas", data_path,
                                                        output_dimension=8)
    signature = experiment.execution_signature(cfg, data_path, atlas_path, 8, "terminal",
                                               chart_selection_method="full_window_projection", warmup_steps=12)
    signature.update(adapter_sha256=file_digest(__file__),
                     logging_adapter_sha256=file_digest(comparison.__file__),
                     goal_policy_sha256=file_digest(goals.__file__),
                     dataset_manifest_sha256=file_digest(output / "dataset_manifest.json"),
                     packages={name: importlib.metadata.version(name) for name in
                               ("numpy", "scipy", "scikit-learn", "cvxpy", "clarabel", "gymnasium", "mujoco")})
    protocol = dict(version=VERSION, seeds=list(SEEDS), steps=steps, signature=signature,
                    note="Development data-source ablation; K/rank inherited, not tuned on this dataset. "
                         "Atlas Reacher-v5/warmup/weights/constraints are retained; this does not reproduce Select-DPC.",
                    reference="p=8 fixed nearest eligible posture from original IID data, velocity=0, XY=seeded target",
                    timing="Serial, one numerical thread. Online localization + QP; excludes offline SVD, environment/I/O. "
                           "First five control steps discarded per rollout; concurrent jobs affect timing.")
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Protocol/source/configuration changed; use a new output directory")
    else:
        if any(output.glob("seed*_control_results.npz")):
            raise ValueError("Unversioned results; refusing to mix experiments")
        save_json(path, protocol)
    return cfg, data_path, atlas_path


def validate_trace(path, protocol_hash, steps, *, pending=False):
    with np.load(path, allow_pickle=True) as a:
        row = json.loads(a["summary"].item())
        if row["protocol_sha256"] != protocol_hash or row["steps"] != steps:
            raise ValueError(f"Incompatible trace: {path}")
        suffix = ".pending.npz" if pending else ".npz"
        if row["seed"] not in SEEDS or Path(path).name != f"seed{row['seed']}_control_results{suffix}":
            raise ValueError("Unexpected seed")
        for key, shape in {"y": (steps,8), "u": (steps,2), "distance": (steps,),
                           "planned_y": (steps,10,8), "planned_u": (steps,10,2),
                           "g_coefficients": (steps,53), "slack": (steps,16),
                           "selection_windows": (steps,120), "selection_residuals": (steps,64)}.items():
            if a[key].shape != shape:
                raise ValueError(f"Invalid {key} shape in {path}")
        if not all(np.isfinite(a[k]).all() for k in ("y", "u", "distance", "control_compute_time_s")):
            raise ValueError("Nonfinite realized trajectory")
        np.testing.assert_allclose(a["distance"], np.linalg.norm(a["y"][:,6:8]-a["final_goal"][6:8],axis=1))
        np.testing.assert_allclose(a["ref"], np.tile(a["final_goal"], (steps,1)))
        for key, value in true_error_metrics(a["distance"]).items():
            if not np.isclose(row[key], value):
                raise ValueError(f"Stale metric: {key}")
        solved = np.isin(a["status"], ["optimal", "optimal_inaccurate"])
        np.testing.assert_allclose(a["planned_y"][solved,0], a["pred"][solved])
        np.testing.assert_allclose(np.clip(a["planned_u"][solved,0],-1,1), a["u"][solved])
        np.testing.assert_allclose(a["stage_control_cost"].sum(), row["closed_loop_control_cost"])
    return dict(row, result_path=str(Path(path).resolve()))


def report(output=OUTPUT):
    """Read-only; never starts preparation or writes to an active run."""
    output = Path(output)
    result = dict(completed=0, requested=len(SEEDS), complete=False, results=[])
    manifest = output / "manifest.json"
    if not manifest.exists():
        return result
    protocol = json.loads(manifest.read_text())
    steps = protocol["steps"]
    checksum = file_digest(manifest)
    rows = [validate_trace(output/f"seed{s}_control_results.npz", checksum, steps)
            for s in SEEDS if (output/f"seed{s}_control_results.npz").exists()]
    result.update(completed=len(rows), complete=len(rows)==len(SEEDS), steps=steps, results=rows,
                  successes=sum(r["strict_success"] for r in rows) if steps>=100 else None)
    if rows:
        result.update(success_rate=result["successes"]/len(rows) if steps>=100 else None,
                      median_final100_error_mm=float(1000*np.median([r["mean_last_100"] for r in rows])) if steps>=100 else None,
                      median_rollout_rmse_mm=float(np.median([r["rollout_rmse_mm"] for r in rows])),
                      timing=summarize_timing([r["result_path"] for r in rows]))
    return result


def run(output=OUTPUT, *, steps=STEPS, batch_size=10, block_start=211, block_stop=310, prepare_only=False):
    if batch_size<1 or not 211<=block_start<=block_stop<=310:
        raise ValueError("Positive batch size and a block within development seeds 211--310 required")
    output = Path(output)
    with run_lock(output), threadpool_limits(limits=1), experiment.output_context(8):
        cfg, data_path, atlas_path = prepare(output, steps)
        checksum = file_digest(output/"manifest.json")
        current = report(output)
        missing = [s for s in range(block_start,block_stop+1)
                   if not (output/f"seed{s}_control_results.npz").exists()][:batch_size]
        print(f"Panel: {current['completed']}/100 complete; this batch: {missing}", flush=True)
        if prepare_only or not missing:
            return current
        H = experiment.load_hankel(data_path, 8)
        atlas = experiment.routed.load_atlas(atlas_path)
        with np.load(output/"source_episodes.npz") as metadata:
            targets, ids = metadata["targets"].copy(), metadata["episode_ids"].copy()
        cache = experiment.routed.precompute_chart_matrix_cache(H, atlas, cfg, geometry="linear")
        for matrix, _ in cache.values():
            if matrix["basis"].shape != (120,53):
                raise ValueError("A chart cannot support rank 53; no silent rank reduction is allowed")
            matrix.pop("g_min", None)
            matrix.pop("g_max", None)
        selector = LoggedSelector(cache, 2, 8, 12)
        # Preserve exact solver basis orientation for reconstruction from saved g.
        bases = np.stack([cache[(k,)][0]["basis"] for k in range(64)])
        np.savez_compressed(output/"controller_geometry.npz", bases=bases)
        np.savez_compressed(output/"selection_geometry.npz", bases=selector.bases)
        records = []
        original = experiment.routed.pipe.DeePCController
        started = datetime.now(timezone.utc).isoformat()
        try:
            with patch.object(experiment.routed.pipe, "DeePCController", comparison.logging_controller(original, records)):
                for seed in missing:
                    save_json(output/"progress.json", dict(state="running", seed=seed, started_utc=started,
                                                            completed=current["completed"], requested=100))
                    records.clear()
                    selector.reset()
                    policy = OriginalDataGoal(H, atlas, cfg, targets, ids)
                    pending = output/f"seed{seed}_control_results.pending.npz"
                    wall_start = time.perf_counter()
                    row = experiment.routed.run_routed_deepc(H, atlas, replace(cfg,seed=seed), pending,
                        matrix_cache=cache, goal_policy=policy, reference_mode="terminal", chart_selector=selector,
                        localization_history_length=12, summary_metadata=dict(dynamic_terminal_reference=False,
                            graph_waypoints_enabled=False, terminal_reference_policy="fixed_nearest_eligible_fingertip"))
                    with np.load(pending,allow_pickle=True) as saved:
                        arrays = {k:saved[k] for k in saved.files if k!="summary"}
                    if len(records)!=steps:
                        raise ValueError("Missing solver instrumentation")
                    arrays.update({k:np.asarray([r[k] for r in records]) for k in records[0]})
                    arrays.update(selection_windows=np.asarray(selector.windows), selection_residuals=np.asarray(selector.residual_history),
                                  selection_tie_count=np.asarray(selector.tie_history), target_xy=policy.target.copy())
                    arrays["warmup_u"] = arrays["selection_windows"][0,:24].reshape(12,2)
                    arrays["warmup_y"] = arrays["selection_windows"][0,24:].reshape(12,8)
                    arrays["y_before"] = arrays["selection_windows"][:,-8:]
                    q=np.array([cfg.q_data_goal_joint]*4+[cfg.q_dq]*2+[cfg.q_xy]*2)
                    arrays["stage_tracking_cost"]=np.sum((arrays["y"]-arrays["ref"])**2*q,axis=1)
                    arrays["stage_input_cost"]=cfg.r*np.sum(arrays["u"]**2,axis=1)
                    arrays["stage_control_cost"]=arrays["stage_tracking_cost"]+arrays["stage_input_cost"]
                    row.update(protocol_sha256=checksum, strict_success=strict_success(row) if steps>=100 else None,
                               rollout_rmse_mm=float(1000*np.sqrt(np.mean(arrays["distance"]**2))),
                               unsolved_steps=int(np.sum(~np.isin(arrays["status"],["optimal","optimal_inaccurate"]))),
                               elapsed_rollout_wall_s=time.perf_counter()-wall_start)
                    np.savez_compressed(pending,**arrays,summary=np.array(json.dumps(row)))
                    # Validate before making the final checkpoint visible to readers.
                    final = output/f"seed{seed}_control_results.npz"
                    validate_trace(pending,checksum,steps,pending=True)
                    pending.replace(final)
                    current=report(output)
                    save_json(output/"summary.json",current)
                    print(f"seed {seed}: success={row['strict_success']}, last={1000*row['last']:.2f} mm",flush=True)
            save_json(output/"progress.json",dict(state="complete" if current["complete"] else "batch_complete",
                                                  completed=current["completed"],requested=100,started_utc=started))
            return current
        except BaseException as exc:
            save_json(output/"progress.json",dict(state="interrupted" if isinstance(exc,KeyboardInterrupt) else "error",
                                                  error=str(exc),started_utc=started))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--steps",type=int,default=STEPS,help="300 for the comparison; use a separate directory for smoke tests")
    parser.add_argument("--batch-size",type=int,default=10)
    parser.add_argument("--block-start",type=int,default=211)
    parser.add_argument("--block-stop",type=int,default=310)
    parser.add_argument("--prepare-only",action="store_true")
    args=parser.parse_args()
    run(args.output,steps=args.steps,batch_size=args.batch_size,block_start=args.block_start,
        block_stop=args.block_stop,prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()
