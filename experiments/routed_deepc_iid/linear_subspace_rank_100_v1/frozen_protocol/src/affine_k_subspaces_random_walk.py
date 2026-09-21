import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np

import expert_pipeline_search as pipe
import targeted_search as select_data


EXPERIMENT_DIR = Path("experiments/affine_k_subspaces_random_walk")


def save_atlas(path, fit, controller_matrix, component_slices, config):
    centers = np.stack([center[:, 0] for center, _ in fit["affine_bases"]], axis=0)
    bases = np.stack([basis for _, basis in fit["affine_bases"]], axis=0)
    offset, matrix = controller_matrix
    np.savez(
        path,
        labels=fit["labels"],
        centers=centers,
        bases=bases,
        row_mean=fit["row_mean"],
        row_scale=fit["row_scale"],
        objective_history=fit["objective_history"],
        change_history=fit["change_history"],
        controller_offset=offset,
        controller_matrix=matrix,
        component_slices=component_slices,
        config=np.array(json.dumps(config), dtype=object),
    )


def load_atlas(path):
    data = np.load(path, allow_pickle=True)
    return data["labels"].astype(int), (data["controller_offset"], data["controller_matrix"])


def run_experiment(args):
    output_dir = EXPERIMENT_DIR / ("smoke" if args.smoke else "full")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "random_walk_hankel.npy"
    atlas_path = output_dir / "affine_k_subspaces_atlas.npz"
    summary_path = output_dir / "summary.json"

    episodes = 8 if args.smoke else args.episodes
    episode_len = 60 if args.smoke else args.episode_len
    n_clusters = 3 if args.smoke else args.clusters
    chart_rank = 4 if args.smoke else args.chart_rank
    control_steps = 20 if args.smoke else args.control_steps
    eval_seeds = args.seeds[:1] if args.smoke else args.seeds

    config = {
        "collector": "select_dpc_random_walk",
        "input_process": {"a": args.input_a, "b": args.input_b},
        "seed": args.data_seed,
        "episodes": episodes,
        "episode_len": episode_len,
        "t_ini": pipe.T_INI,
        "t_f": pipe.T_F,
        "window_length": pipe.L,
        "clusters": n_clusters,
        "chart_rank": chart_rank,
        "cluster_seed": args.cluster_seed,
        "cluster_max_iter": args.cluster_max_iter,
        "standardize_for_clustering": True,
        "eval_seeds": eval_seeds,
        "control_steps": control_steps,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    if args.force_collect or not data_path.exists():
        H = select_data.collect_select_dpc_random_walk(
            seed=args.data_seed,
            episodes=episodes,
            episode_len=episode_len,
            walk_scale=args.input_b,
            momentum=args.input_a,
            save_path=data_path,
        )
    else:
        H = np.load(data_path)
    print("offline Hankel", H.shape, "input range", float(H[:pipe.M * pipe.L].min()), float(H[:pipe.M * pipe.L].max()))

    if args.force_cluster or not atlas_path.exists():
        fit = pipe.fit_affine_k_subspaces(
            H,
            n_clusters=n_clusters,
            rank=chart_rank,
            seed=args.cluster_seed,
            max_iter=args.cluster_max_iter,
            standardize_rows=True,
        )
        controller_matrix, component_slices = pipe.concatenate_affine_bases(
            fit["affine_bases"],
            global_offset=H.mean(axis=1, keepdims=True),
        )
        save_atlas(atlas_path, fit, controller_matrix, component_slices, config)
        labels = fit["labels"]
    else:
        labels, controller_matrix = load_atlas(atlas_path)

    offset, concatenated_basis = controller_matrix
    print("cluster sizes", np.bincount(labels, minlength=n_clusters).tolist())
    print("concatenated affine matrix", concatenated_basis.shape, "rank", int(np.linalg.matrix_rank(concatenated_basis)))

    controller_labels = np.zeros(H.shape[1], dtype=int)
    summaries = []
    for seed in eval_seeds:
        variant = pipe.Variant(
            name=f"random_walk_affine_concat_seed{seed}",
            mode="affine_subspace",
            rank=chart_rank,
            try_top=1,
            u_limit=0.5,
            q_xy=200000.0,
            q_dq=50.0,
            r=0.005,
            lam_g=0.05,
            lam_ini=100000.0,
            lam_du=5.0,
            solver="CLARABEL",
            steps=control_steps,
            warmup_scale=0.0,
            seed=seed,
            fixed_chart=0,
            use_ik_ref=False,
            feedback_blend=0.0,
            feedback_fallback=False,
        )
        result_path = output_dir / f"seed{seed}_control_results.npz"
        summary = pipe.run_atlas_deepc(
            H,
            controller_labels,
            variant,
            save_path=result_path,
            controller_matrices=[controller_matrix],
        )
        summary["result_path"] = str(result_path)
        summaries.append(summary)
        print("seed", seed, "mean30", summary["mean_last_30"], "last", summary["last"], "success", summary["success"])

    payload = {
        "config": config,
        "controller_matrix_shape": list(concatenated_basis.shape),
        "controller_matrix_rank": int(np.linalg.matrix_rank(concatenated_basis)),
        "cluster_sizes": np.bincount(labels, minlength=n_clusters).tolist(),
        "results": summaries,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Random-walk affine K-subspaces Atlas-DeePC experiment")
    parser.add_argument("--smoke", action="store_true", help="Run a small end-to-end verification experiment.")
    parser.add_argument("--force-collect", action="store_true")
    parser.add_argument("--force-cluster", action="store_true")
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--episode-len", type=int, default=200)
    parser.add_argument("--input-a", type=float, default=1.0)
    parser.add_argument("--input-b", type=float, default=0.1)
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--chart-rank", type=int, default=12)
    parser.add_argument("--cluster-seed", type=int, default=7)
    parser.add_argument("--cluster-max-iter", type=int, default=20)
    parser.add_argument("--control-steps", type=int, default=320)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5, 7, 8, 42])
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
