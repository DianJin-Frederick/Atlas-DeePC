"""Paired, serial rank 59/62/64 experiment using the original frozen protocol."""
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile
import time

import linear_subspace_rank_ablation as e
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
VERSION = "selected_ranks_1000_clarabel_v1"
RANKS = (59, 62, 64)
SEEDS = tuple(range(311, 1311))
e.VERSION, e.RESULT_ROOT, e.RANKS, e.SEEDS = VERSION, ROOT, RANKS, SEEDS


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare():
    original = json.loads((ROOT.parent / "linear_subspace_rank_100_v1/manifest.json").read_text())
    for name, digest in original["source_sha256"].items():
        assert e.file_digest(Path(__file__).with_name(name)) == digest, name
    assert asdict(e.baseline.default_config()) == original["base_config"]
    signature = dict(original, version=VERSION, ranks=list(RANKS), seeds=list(SEEDS),
                     dependency_sha256={p.name: e.file_digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
                                        if p.name not in original["source_sha256"] and p.name != Path(__file__).name},
                     solver="CLARABEL", solver_options={"max_iter": 10000},
                     runner_sha256=e.file_digest(Path(__file__)),
                     package_versions={n: version(n) for n in ("numpy", "scipy", "cvxpy", "clarabel", "mujoco", "gymnasium")},
                     workers=1, numerical_threads=1, order="rotate ranks within each seed",
                     timing_discard_initial_steps=5, warmup_seed=210, warmup_steps=10)
    for key, path in (("data_sha256", e.baseline.DATA_PATH), ("atlas_sha256", e.baseline.ATLAS_PATH)):
        assert e.file_digest(path) == signature[key], path
    manifest = ROOT / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == signature, "Incompatible resume"
    else:
        assert not any(ROOT.glob("rank*/seed*.npz"))
        e.write_json(manifest, signature)
    return signature


def freeze(value):
    if isinstance(value, np.ndarray):
        assert not value.dtype.hasobject, "Unexpected mutable object array"
        value.flags.writeable = False
    elif isinstance(value, dict):
        for item in value.values():
            freeze(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            freeze(item)


def digests(value, prefix=""):
    result = {}
    if isinstance(value, np.ndarray):
        digest = hashlib.sha256()
        digest.update(str((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
        result[prefix] = digest.hexdigest()
    elif isinstance(value, dict):
        for key, item in value.items():
            result.update(digests(item, prefix + "/" + str(key)))
    elif isinstance(value, (tuple, list)):
        for key, item in enumerate(value):
            result.update(digests(item, prefix + "/" + str(key)))
    return result


def initialize():
    start = time.monotonic()
    e.initialize_worker()
    assert not e._H.flags.writeable
    freeze(e._atlas)
    controllers, setup = {}, {}
    for rank in RANKS:
        tick = time.monotonic()
        controllers[rank] = e.controller_for_rank(rank)
        setup[str(rank)] = time.monotonic() - tick
        freeze(controllers[rank].matrix_cache)
        freeze(controllers[rank].route_tables)
    reference = controllers[64]
    for rank, controller in controllers.items():
        assert controller.H is e._H and controller.atlas is e._atlas
        assert replace(controller.config, svd_rank=64) == reference.config
        for chart, (matrix, members) in controller.matrix_cache.items():
            full, full_members = reference.matrix_cache[chart]
            np.testing.assert_array_equal(members, full_members)
            np.testing.assert_array_equal(matrix["basis"], full["basis"][:, :rank])
            for bound in ("g_min", "g_max"):
                np.testing.assert_allclose(matrix[bound], full[bound][:rank], rtol=1e-12, atol=1e-12)
        for actual, expected in zip(controller.route_tables, reference.route_tables):
            np.testing.assert_array_equal(actual, expected)
    invariant = {"atlas_arrays": digests(e._atlas),
                 "rank_arrays": {str(r): digests((c.matrix_cache, c.route_tables)) for r, c in controllers.items()},
                 "fixed_memberships_and_routes": True, "nested_basis_prefixes": True,
                 "data_read_only": True, "atlas_and_matrix_arrays_read_only": True}
    path = ROOT / "offline_invariance.json"
    if path.exists():
        assert json.loads(path.read_text()) == invariant
    else:
        e.write_json(path, invariant)
    e.write_json(ROOT / f"setup_{time.time_ns()}.json", {
        "utc": now(), "matrix_and_route_setup_seconds_by_rank": setup,
        "total_load_and_setup_seconds": time.monotonic()-start})
    return controllers, invariant


def verify(controllers, invariant, signature):
    assert digests(e._atlas) == invariant["atlas_arrays"]
    for rank, controller in controllers.items():
        assert digests((controller.matrix_cache, controller.route_tables)) == invariant["rank_arrays"][str(rank)]
    for key, path in (("data_sha256", e.baseline.DATA_PATH), ("atlas_sha256", e.baseline.ATLAS_PATH)):
        assert e.file_digest(path) == signature[key]
    for name, digest in {**signature["source_sha256"], **signature["dependency_sha256"]}.items():
        assert e.file_digest(Path(__file__).with_name(name)) == digest
    assert e.file_digest(Path(__file__)) == signature["runner_sha256"]


def warmup(controllers):
    # Observe the actual requested solver on short discarded rollouts.
    solve = e.pipe.DeePCController.solve
    calls = Counter()
    def checked(controller, *args, **kwargs):
        assert controller.solver == "CLARABEL" and controller.lam_g == .05
        calls[str(controller.solver)] += 1
        return solve(controller, *args, **kwargs)
    e.pipe.DeePCController.solve = checked
    try:
        for rank, controller in controllers.items():
            config = controller.config
            try:
                controller.config = replace(config, steps=10)
                with tempfile.TemporaryDirectory(prefix="atlas_warmup_") as tmp:
                    row = controller.run(210, Path(tmp) / "trace.npz")
                assert row["steps"] == 10
            finally:
                controller.config = config
    finally:
        e.pipe.DeePCController.solve = solve
    assert calls == {"CLARABEL": 30}
    e.write_json(ROOT / "preflight.json", {"utc": now(), "passed": True,
        "actual_solver_calls": dict(calls), "lam_g": .05, "warmup_excluded_from_results": True})


def main(preflight_only=False):
    signature = prepare()
    controllers, invariant = initialize()
    if not (ROOT / "fairness_audit.json").exists():
        e.write_json(ROOT / "fairness_audit.json", e.routed.audit_iid_library(e._H, eval_seeds=SEEDS))
    warmup(controllers)
    verify(controllers, invariant, signature)
    print("Preflight passed: frozen data/charts, nested bases, actual Clarabel and lambda_g=.05", flush=True)
    if preflight_only:
        return
    start = time.monotonic()
    rows = {rank: [] for rank in RANKS}
    e.write_json(ROOT / f"session_{time.time_ns()}.json", {"utc": now(), "workers": 1,
        "numerical_threads": 1, "seed_count": len(SEEDS), "ranks": list(RANKS)})
    for index, seed in enumerate(SEEDS):
        shift = index % len(RANKS)
        order = RANKS[shift:] + RANKS[:shift]
        targets = []
        for rank in order:
            e._controller = controllers[rank]
            tick = time.monotonic()
            existed = e.checkpoint_path(rank, seed).exists()
            e.run_one(rank, seed)
            row = e.load_checkpoint(e.checkpoint_path(rank, seed), rank, seed)
            rows[rank].append(row)
            targets.append(row["exact_target_xy"])
            with (ROOT / "rollout_wall_times.jsonl").open("a") as stream:
                stream.write(json.dumps({"utc": now(), "rank": rank, "seed": seed,
                    "seconds": time.monotonic()-tick, "reused_checkpoint": existed}) + "\n")
            print(f"rank {rank} seed {seed}: success={e.strict_success(row)}, unsolved={row['unsolved_steps']}", flush=True)
        np.testing.assert_array_equal(targets, [targets[0]] * len(RANKS))
        if (index+1) % 50 == 0 or index+1 == len(SEEDS):
            verify(controllers, invariant, signature)
            e.write_json(ROOT / "invariance_latest.json", {"utc": now(), "through_seed": seed, "passed": True})
        progress = {"version": VERSION, "updated_utc": now(), "complete": False,
                    "paired_seeds_complete": index+1, "rollouts_complete": 3*(index+1),
                    "total_rollouts": 3000, "session_elapsed_seconds": time.monotonic()-start,
                    "ranks": [{"rank": r, "completed": len(rows[r]),
                               "successes": sum(e.strict_success(x) for x in rows[r]),
                               "unsolved_qps": sum(x["unsolved_steps"] for x in rows[r])} for r in RANKS]}
        e.write_json(ROOT / "progress.json", progress)
    completed = [e.summarize_rank(rank, SEEDS) for rank in RANKS]
    statuses = {str(r): dict(sum((Counter(x["statuses"]) for x in rows[r]), Counter())) for r in RANKS}
    summary = {"version": VERSION, "complete": True, "completed_utc": now(),
               "ranks": list(RANKS), "seeds": list(SEEDS), "rollouts": 3000,
               "completed": completed, "statuses_by_rank": statuses,
               "signature": signature, "paired_targets_verified": True,
               "offline_invariance_verified": True,
               "timing_note": "Serial, one numerical thread, rotating rank order per seed; shared-computer wall clock. First 5 steps excluded from pooled timing; all 320 retained in checkpoints. QP includes construction and solve. Offline setup and rollout wall time recorded separately."}
    e.write_json(ROOT / "summary.json", summary)
    e.write_json(ROOT / "progress.json", dict(progress, complete=True))
    print(json.dumps(summary["completed"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-only", action="store_true")
    main(parser.parse_args().preflight_only)
