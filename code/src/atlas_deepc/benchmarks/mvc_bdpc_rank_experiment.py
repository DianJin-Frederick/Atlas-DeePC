"""Isolated fixed-rank MVC experiment; never changes the default controller.

Rebuild all offline trajectory subspaces, re-cluster with the same NMF settings,
and use the resulting rank-r representative models for control. Online matching
is unchanged: its past-length Hankels have only one column, not r columns.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
from atlas_deepc.benchmarks import mvc_bdpc_terminal8 as m
from atlas_deepc.benchmarks import tune_mvc_bdpc_terminal8 as tune


def fixed_subspace(matrix, rank):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite matrix")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or not 1 <= rank <= min(matrix.shape):
        raise ValueError("Fixed rank must be an integer within the matrix dimensions")
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    if s[rank-1] <= np.finfo(float).eps * max(matrix.shape) * s[0]:
        raise ValueError("Requested rank exceeds numerical rank; no padding with null directions")
    return u[:, :rank], (u[:, :rank]*s[:rank]) @ vt[:rank], s


def rebuild(original, rank):
    lib = m.Library.load(original, m.Config())
    c = lib.config
    bases, matrices, spectra = [], [], []
    for trajectory in lib.trajectories:
        basis, matrix, spectrum = fixed_subspace(m.hankel(trajectory, c.past+1), rank)
        bases.append(basis); matrices.append(matrix); spectra.append(spectrum)
    d1 = m.squareform(m.pdist(lib.trajectories.reshape(len(bases), -1), metric="euclidean"))
    d2 = np.zeros_like(d1)
    for i in range(len(bases)):
        for j in range(i):
            d2[i, j] = d2[j, i] = m.subspace_distance(bases[i], bases[j])
    lib.distances = np.stack([d1, d2])
    lib.nmf = m.joint_nmf([d1, d2], c)
    if not lib.nmf["converged"]:
        raise RuntimeError("Fixed-rank NMF did not converge; do not accept unchecked clusters")
    lib.ranks = np.full(len(bases), rank, dtype=int)
    lib.spectra = np.stack(spectra)
    lib.representatives = lib.nmf["representatives"]
    lib.centers = lib.trajectories[lib.representatives]
    lib.models = np.stack([matrices[i] for i in lib.representatives])
    lib.audit = {
        "rank_policy": "fixed offline rank", "fixed_rank": rank,
        "online_subspace_policy": "unchanged one-column past Hankels",
        "offline_episodes": len(bases), "model_shape": list(lib.models.shape),
        "selected_ranks": lib.ranks[lib.representatives].tolist(),
        "representatives": lib.representatives.tolist(),
        "distinct_representatives": len(np.unique(lib.representatives)),
        "cluster_sizes": np.bincount(lib.nmf["labels"], minlength=c.clusters).tolist(),
        "nmf_converged": bool(lib.nmf["converged"]),
        "nmf_outer_iterations": len(lib.nmf["objective"])-1,
        "model_geometry": lib.geometry_audit(),
        "relative_retained_singular_values": [float(s[rank-1]/s[0]) for s in spectra],
    }
    return lib


def run(output, rank, seeds):
    if not seeds or len(seeds) != len(set(seeds)) or not all(211 <= s <= 310 for s in seeds):
        raise ValueError("Use unique development seeds within 211--310")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 1 <= rank <= 70:
        raise ValueError("Rank must be an integer from 1 to 70")
    original = tune.OFFLINE / "offline_model.npz"
    previous = json.loads((tune.OFFLINE / "manifest.json").read_text())
    if previous["source_sha256"] != m.utils.digest(m.__file__) or previous["data_sha256"] != m.utils.digest(m.DATA):
        raise ValueError("Controller or offline data differs from baseline")
    if previous["config"] != asdict(m.Config()):
        raise ValueError("Unexpected baseline configuration")
    references, provenance = m.read_references()
    candidates = {"original_weights": {}, "tuned_weights": {
        "q_joint": .04, "q_dq": 400., "q_xy": 400., "r": .0001, "lambda_epsilon": 200.}}
    configs = {name: tune.weight_config(m.Config(), weights) for name, weights in candidates.items()}
    identity = {"rank": rank, "steps": 300, "seeds": seeds,
        "source_sha256": m.utils.digest(__file__), "controller_sha256": m.utils.digest(m.__file__),
        "legacy_sha256": m.utils.digest(m.legacy.__file__), "utilities_sha256": m.utils.digest(m.utils.__file__),
        "tuning_utilities_sha256": m.utils.digest(tune.__file__),
        "original_model_sha256": m.utils.digest(original), "data_sha256": previous["data_sha256"],
        "reference_provenance": provenance, "configs": {n: asdict(c) for n,c in configs.items()},
        "policy": "Rebuild all offline rank-r subspaces and clustering; keep online matching, horizon, objective, constraints, normalization and references unchanged"}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError("Experiment identity changed; choose a fresh directory")
        elif any(output.glob("**/*.npz")):
            raise ValueError("Refusing unversioned results")
        else:
            m.utils.write_json(manifest, identity)
        started = time.perf_counter()
        try:
            cache = output / "offline_model.npz"
            m.utils.write_json(output / "progress.json", {"state": "preparing", "rank": rank})
            if not cache.exists():
                lib = rebuild(original, rank)
                lib.save(cache)
                m.utils.write_json(output / "offline_audit.json", lib.audit)
            else:
                lib = m.Library.load(cache, m.Config())
            print(json.dumps(lib.audit), flush=True)
            model_hash = m.utils.digest(cache)
            reports = {}
            for name, config in configs.items():
                lib.config = config
                directory = output / name; directory.mkdir(exist_ok=True)
                m.utils.write_json(directory / "manifest.json", {**identity, "config": asdict(config), "offline_model_sha256": model_hash})
                for seed in seeds:
                    path = directory / f"seed{seed}_control_results.npz"
                    if not path.exists():
                        with (directory / f"seed{seed}.log").open("a") as log, redirect_stdout(log):
                            m.run_seed(lib, seed, 300, directory, references[seed])
                    else:
                        with np.load(path, allow_pickle=False) as z:
                            row = json.loads(z["summary"].item())
                            if row["seed"] != seed or row["requested_steps"] != 300:
                                raise ValueError("Incompatible saved rollout")
                    reports[name] = tune.compact_report(directory, seeds)
                    m.utils.write_json(output / "report.json", reports)
                    m.utils.write_json(output / "progress.json", {"state": "running", "candidate": name, "seed": seed})
                    print(json.dumps({"candidate": name, "seed": seed, **{k:v for k,v in reports[name].items() if k != "results"}}), flush=True)
            m.utils.write_json(output / "progress.json", {"state": "complete", "elapsed_seconds": time.perf_counter()-started})
        except BaseException as exc:
            m.utils.write_json(output / "progress.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error", "message": str(exc)})
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=53)
    parser.add_argument("--seeds", type=int, nargs="+", default=[211, 212, 213, 222, 237, 258, 281, 305])
    args = parser.parse_args()
    run(args.output, args.rank, args.seeds)
