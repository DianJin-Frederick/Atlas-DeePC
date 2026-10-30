
if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np

from atlas_deepc.control import expert_pipeline_search as pipe
TARGET_H_PATH = Path("data/select_iid_offline_hankel.npy")
TARGET_LABEL_PATH = Path("data/select_iid_prediction_error_labels.npz")
TARGET_RESULTS_PATH = Path("select_iid_search_results.json")


def collect_select_dpc_hankel(
    seed=2026,
    episodes=200,
    episode_len=200,
    input_a=0.0,
    input_b=1.0,
    save_path=TARGET_H_PATH,
):
    """Collect a Hankel library using the Select-DPC input process.

    The process is ``u_t = a * u_{t-1} + b * n_t`` with independent
    ``n_t ~ Uniform(-1, 1)`` and clipping to the environment action bounds.
    """
    rng = np.random.default_rng(seed)
    columns = []
    for ep in range(episodes):
        env_seed = seed + ep
        env = gym.make(pipe.REACHER_ENV_ID, max_episode_steps=episode_len + 5)
        obs, _ = env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        previous_action = np.zeros(pipe.M, dtype=float)
        U = []
        Y = []
        for _ in range(episode_len):
            noise = rng.uniform(-1.0, 1.0, size=pipe.M)
            action = input_a * previous_action + input_b * noise
            action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, info = env.step(action)
            U.append(action.copy())
            Y.append(pipe.reacher_output_from_obs(obs))
            previous_action = action
            if terminated or truncated:
                break
        env.close()
        U = np.asarray(U, dtype=float)
        Y = np.asarray(Y, dtype=float)
        for start in range(U.shape[0] - pipe.L + 1):
            columns.append(np.concatenate([
                U[start:start + pipe.L].reshape(pipe.L * pipe.M),
                Y[start:start + pipe.L].reshape(pipe.L * pipe.P),
            ]))
        if (ep + 1) % 10 == 0:
            print("collected episodes", ep + 1, "columns", len(columns), flush=True)
    H = np.asarray(columns, dtype=float).T
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, H)
    return H


def collect_select_deepc_iid(seed=2026, episodes=200, episode_len=200, input_scale=1.0, save_path=TARGET_H_PATH):
    """
    Collect Reacher offline data using the Select-DPC Reacher protocol.

    The paper's Reacher ablation uses IID inputs rather than random-walk inputs:
        u_t = a u_{t-1} + b n_t, n_t ~ U(-1, 1)
    with (a, b) = (0, 1). It reports 200 simulations with 200 steps each.

    This intentionally does not oversample any test target seed and does not use
    an expert/IK controller. It is a fairer offline data source for comparing
    chart/subspace methods against Select-DPC-style data selection.
    """
    return collect_select_dpc_hankel(
        seed=seed,
        episodes=episodes,
        episode_len=episode_len,
        input_a=0.0,
        input_b=input_scale,
        save_path=save_path,
    )


def collect_select_dpc_random_walk(
    seed=2026,
    episodes=200,
    episode_len=200,
    walk_scale=0.1,
    momentum=1.0,
    save_path=Path("select_random_walk_offline_hankel.npy"),
):
    """Collect the Select-DPC random-walk profile, normally ``(a, b)=(1, 0.1)``."""
    return collect_select_dpc_hankel(
        seed=seed,
        episodes=episodes,
        episode_len=episode_len,
        input_a=momentum,
        input_b=walk_scale,
        save_path=save_path,
    )


def collect_targeted(*args, **kwargs):
    """Backward-compatible alias for the current fair Select-DPC IID collector."""
    return collect_select_deepc_iid(*args, **kwargs)


def main():
    started = time.time()
    H = np.load(TARGET_H_PATH) if TARGET_H_PATH.exists() else collect_targeted()
    print("H", H.shape, "u range", H[:pipe.M * pipe.L].min(), H[:pipe.M * pipe.L].max(), flush=True)
    if TARGET_LABEL_PATH.exists():
        labels = np.load(TARGET_LABEL_PATH)["labels"]
    else:
        labels = pipe.learn_prediction_error_atlas(H, seed=7, label_path=TARGET_LABEL_PATH)
    print("cluster sizes", np.bincount(labels, minlength=pipe.K).tolist(), flush=True)

    variants = []
    for mode in ["raw", "basis"]:
        for u_limit in [0.35, 0.5]:
            for q_xy in [1000.0, 5000.0, 20000.0]:
                variants.append(pipe.Variant(
                    name=f"targeted_{mode}_u{u_limit}_q{int(q_xy)}",
                    mode=mode,
                    rank=100,
                    max_cols=500,
                    try_top=8,
                    u_limit=u_limit,
                    q_xy=q_xy,
                    q_dq=0.1,
                    r=0.0001,
                    lam_g=0.01 if mode == "basis" else 0.1,
                    lam_ini=100000.0,
                    lam_du=0.05,
                    solver="OSQP",
                    steps=120,
                    warmup_scale=0.0,
                    seed=42,
                ))
    results = []
    for variant in variants:
        print("RUN", variant, flush=True)
        save_path = Path(f"{variant.name}.npz")
        summary = pipe.run_atlas_deepc(H, labels, variant, save_path=save_path)
        summary["save_path"] = str(save_path)
        results.append(summary)
        print(" ->", summary, flush=True)
        TARGET_RESULTS_PATH.write_text(json.dumps({"elapsed": time.time() - started, "results": results}, indent=2))
        if summary["success"]:
            break
    results.sort(key=lambda r: (r["mean_last_30"], r["max_last_30"], r["last"]))
    TARGET_RESULTS_PATH.write_text(json.dumps({"elapsed": time.time() - started, "best": results[:10], "results": results}, indent=2))
    print("BEST", flush=True)
    for row in results[:5]:
        print(row, flush=True)


if __name__ == "__main__":
    main()
