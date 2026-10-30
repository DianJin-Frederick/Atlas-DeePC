"""Build a cached contextual atlas from development-time query geometry.

The ordinary contextual-cluster atlas quantizes offline trajectories.  This
builder instead quantizes causal deployment queries [Y_p, final goal] collected
on development seeds.  Each query prototype owns the same top-N offline Hankel
columns that contextual-aware DPC would select for that representative query,
but its rank-reduced basis is cached once offline.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import contextual_cluster_atlas_experiment as contextual
import expert_pipeline_search as pipe
import routed_deepc_experiment as routed
from contextual_chart_expansion_experiment import baseline_config, output_cost
from online_selected_deepc_comparison import (
    prepare_contextual_library,
    selected_affine_basis,
)


def collect_development_queries(
    rollout_dirs: list[Path],
    seeds,
    library: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    past_queries, future_queries = [], []
    mean, scale = library["output_mean"], library["output_scale"]
    for seed in seeds:
        matches = [
            directory / f"seed{int(seed)}_control_results.npz"
            for directory in rollout_dirs
            if (directory / f"seed{int(seed)}_control_results.npz").exists()
        ]
        if not matches:
            raise FileNotFoundError(f"no rollout found for development seed {seed}")
        with np.load(
            matches[0],
            allow_pickle=True,
        ) as data:
            outputs = np.asarray(data["y"], dtype=float)
            references = np.asarray(data["ref"], dtype=float)
        # The first two online histories include unrecorded zero-action warmup
        # samples.  Omitting them leaves 318 exact causal histories per rollout.
        for step in range(pipe.T_INI, outputs.shape[0]):
            past = (
                (outputs[step - pipe.T_INI:step] - mean[None, :])
                / scale[None, :]
            ).reshape(-1)
            reference = np.tile(references[step].reshape(1, pipe.P), (pipe.T_F, 1))
            future = ((reference - mean[None, :]) / scale[None, :]).reshape(-1)
            past_queries.append(past)
            future_queries.append(future)
    return np.asarray(past_queries), np.asarray(future_queries)


def _query_embedding(
    past: np.ndarray,
    future: np.ndarray,
    Q: np.ndarray,
) -> np.ndarray:
    sqrt_q = np.sqrt(np.tile(np.diag(Q), pipe.T_F))
    return np.hstack([
        past / np.sqrt(past.shape[1]),
        future * sqrt_q[None, :] / np.sqrt(np.sum(sqrt_q**2)),
    ])


def _prototype_supports(
    prototype_past: np.ndarray,
    prototype_future: np.ndarray,
    library: dict[str, np.ndarray],
    Q: np.ndarray,
    support_columns: int,
    batch_size: int = 16,
) -> np.ndarray:
    offline_past = library["Y_p_normalized"].T
    sqrt_q = np.sqrt(np.tile(np.diag(Q), pipe.T_F))
    offline_future = library["Y_f_normalized"].T * sqrt_q[None, :]
    weighted_prototype_future = prototype_future * sqrt_q[None, :]
    supports = []
    for start in range(0, prototype_past.shape[0], batch_size):
        stop = min(start + batch_size, prototype_past.shape[0])
        past_distance = np.sqrt(np.maximum(
            np.sum(prototype_past[start:stop] ** 2, axis=1)[:, None]
            + np.sum(offline_past**2, axis=1)[None, :]
            - 2.0 * prototype_past[start:stop] @ offline_past.T,
            0.0,
        ))
        future_distance = np.sqrt(np.maximum(
            np.sum(weighted_prototype_future[start:stop] ** 2, axis=1)[:, None]
            + np.sum(offline_future**2, axis=1)[None, :]
            - 2.0 * weighted_prototype_future[start:stop] @ offline_future.T,
            0.0,
        ))
        past_score = (
            past_distance - past_distance.mean(axis=1, keepdims=True)
        ) / np.maximum(past_distance.std(axis=1, keepdims=True), 1e-12)
        future_score = (
            future_distance - future_distance.mean(axis=1, keepdims=True)
        ) / np.maximum(future_distance.std(axis=1, keepdims=True), 1e-12)
        score = past_score + future_score
        for row in score:
            selected = np.argpartition(row, support_columns - 1)[:support_columns]
            supports.append(np.sort(selected))
    return np.asarray(supports, dtype=int)


def build_query_amortized_atlas(
    H: np.ndarray,
    physical_atlas: dict,
    control: routed.RoutedDeePCConfig,
    rollout_dirs: list[Path],
    development_seeds,
    clusters: int = 512,
    support_columns: int = 200,
    cluster_seed: int = 91,
    prototype_mode: str = "medoid",
) -> dict[str, np.ndarray]:
    library = prepare_contextual_library(H)
    Q = output_cost(control)
    query_past, query_future = collect_development_queries(
        rollout_dirs, development_seeds, library
    )
    embedding = _query_embedding(query_past, query_future, Q)
    model = MiniBatchKMeans(
        n_clusters=clusters,
        random_state=cluster_seed,
        n_init=5,
        batch_size=2048,
        max_iter=400,
        reassignment_ratio=0.001,
    )
    assignment = model.fit_predict(embedding)
    assignment_counts = np.bincount(assignment, minlength=clusters)
    past_width = query_past.shape[1]
    if prototype_mode == "medoid":
        # A real causal query preserves the strong correlations between target,
        # configuration, and velocity that an arithmetic center can blur.
        representative_index = np.zeros(clusters, dtype=int)
        representative_distance = np.full(clusters, np.inf)
        for start in range(0, embedding.shape[0], 4096):
            stop = min(start + 4096, embedding.shape[0])
            distances = model.transform(embedding[start:stop])
            local = np.argmin(distances, axis=0)
            values = distances[local, np.arange(clusters)]
            improved = values < representative_distance
            representative_distance[improved] = values[improved]
            representative_index[improved] = start + local[improved]
        prototype_past = query_past[representative_index]
        prototype_future = query_future[representative_index]
    elif prototype_mode == "center":
        sqrt_q = np.sqrt(np.tile(np.diag(Q), pipe.T_F))
        prototype_past = (
            model.cluster_centers_[:, :past_width] * np.sqrt(past_width)
        )
        prototype_future = (
            model.cluster_centers_[:, past_width:]
            * np.sqrt(np.sum(sqrt_q**2))
            / sqrt_q[None, :]
        )
        representative_index = np.full(clusters, -1, dtype=int)
    else:
        raise ValueError(f"Unknown prototype mode: {prototype_mode}")
    supports = _prototype_supports(
        prototype_past, prototype_future, library, Q, support_columns
    )

    offsets, bases, g_mins, g_maxs = [], [], [], []
    physical_labels = np.asarray(physical_atlas["labels"], dtype=int)
    prototype_chart = []
    for index, support in enumerate(supports):
        matrix = selected_affine_basis(H, support, control)
        offsets.append(matrix["offset"])
        bases.append(matrix["basis"])
        g_mins.append(matrix["g_min"])
        g_maxs.append(matrix["g_max"])
        histogram = np.bincount(
            physical_labels[support],
            minlength=int(physical_atlas["centers"].shape[0]),
        )
        prototype_chart.append(int(np.argmax(histogram)))
        if (index + 1) % 32 == 0:
            print(f"cached query bases: {index + 1}/{clusters}", flush=True)

    config = contextual.ContextualClusterConfig(
        architecture="global_vq",
        contextual_clusters=clusters,
        support_columns=support_columns,
        cluster_seed=cluster_seed,
        activation_mode="always_on",
        reference_mode="final_only",
        distance_mode="balanced_embedding",
    )
    return {
        "prototype_past": prototype_past,
        "prototype_future": prototype_future,
        "prototype_chart": np.asarray(prototype_chart, dtype=int),
        "assignment_counts": assignment_counts,
        "representative_query_index": representative_index,
        "support_indices": supports,
        "matrix_offsets": np.asarray(offsets),
        "matrix_bases": np.asarray(bases),
        "matrix_g_mins": np.asarray(g_mins),
        "matrix_g_maxs": np.asarray(g_maxs),
        "output_mean": library["output_mean"],
        "output_scale": library["output_scale"],
        "q_diagonal": np.diag(Q),
        "basis_rank": np.array(control.svd_rank, dtype=int),
        "cluster_config": np.array(json.dumps(asdict(config)), dtype=object),
        "atlas_training": np.array(json.dumps({
            "method": "development_query_vq",
            "development_seed_start": int(min(development_seeds)),
            "development_seed_stop": int(max(development_seeds)),
            "queries": int(query_past.shape[0]),
            "prototype_mode": prototype_mode,
        }), dtype=object),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--rollout-dir", type=Path, action="append", required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/select_iid_offline_hankel.npy"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=111)
    parser.add_argument("--seed-stop", type=int, default=210)
    parser.add_argument("--clusters", type=int, default=512)
    parser.add_argument("--support-columns", type=int, default=200)
    parser.add_argument("--cluster-seed", type=int, default=91)
    parser.add_argument("--prototype-mode", choices=("medoid", "center"), default="medoid")
    args = parser.parse_args()
    started = time.perf_counter()
    baseline_payload = json.loads((args.baseline_dir / "summary.json").read_text())
    control = baseline_config(baseline_payload)
    H = np.load(args.data_path, mmap_mode="r")
    atlas = routed.load_atlas(args.baseline_dir / "behavioral_atlas.npz")
    artifact = build_query_amortized_atlas(
        H, atlas, control, args.rollout_dir,
        range(args.seed_start, args.seed_stop + 1),
        clusters=args.clusters,
        support_columns=args.support_columns,
        cluster_seed=args.cluster_seed,
        prototype_mode=args.prototype_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    contextual.save_contextual_cluster_atlas(args.output, artifact)
    print(json.dumps({
        "output": str(args.output),
        "clusters": args.clusters,
        "support_columns": args.support_columns,
        "prototype_mode": args.prototype_mode,
        "seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
