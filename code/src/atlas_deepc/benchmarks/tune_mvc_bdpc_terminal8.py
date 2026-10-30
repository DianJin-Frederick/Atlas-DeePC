"""Serial control-weight tuning with frozen p=8 MVC models and development goals.

Only the five existing objective weights may change. Never modifies the
controller, model rank, selection rule, constraints, or existing experiments.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import fcntl
import json
from pathlib import Path
import re
import time

import numpy as np
from threadpoolctl import threadpool_limits
from atlas_deepc.benchmarks import mvc_bdpc_terminal8 as m

OFFLINE = m.ROOT / "experiments/reacher/mvc_bdpc_p8_smoke_211_213_t300_v1"
ALLOWED = {"q_joint", "q_dq", "q_xy", "r", "lambda_epsilon"}


def weight_config(base, overrides):
    if not set(overrides) <= ALLOWED:
        raise ValueError("Only existing control-objective weights may change")
    config = replace(base, **overrides)
    config.validate()
    return config


def compact_report(directory, seeds):
    summary = m.summarize(directory, seeds)
    full = [row for row in summary["results"] if row["complete"]]
    actions, clips, predictions = [], [], []
    for row in full:
        with np.load(directory / f"seed{row['seed']}_control_results.npz", allow_pickle=False) as z:
            actions.append(float(np.mean(np.linalg.norm(z["u"], axis=1))))
            clips.append(float(np.mean(z["action_clip"] > 1e-8)))
            predictions.append(float(np.mean(z["prediction_error_xy"]) * 1000))
    return {"finished": summary["finished_seeds"], "requested": len(seeds),
            "full_rollouts": len(full), "aborted": summary["failed_rollouts"],
            "successes": summary["successes"],
            "mean_last100_mm": float(np.mean([r["mean_last_100"] for r in full])*1000) if full else None,
            "mean_last30_mm": float(np.mean([r["mean_last_30"] for r in full])*1000) if full else None,
            "mean_final_mm": float(np.mean([r["last"] for r in full])*1000) if full else None,
            "clip_fraction": float(np.mean(clips)) if clips else None,
            "mean_action_norm": float(np.mean(actions)) if actions else None,
            "mean_prediction_mm": float(np.mean(predictions)) if predictions else None,
            "results": summary["results"]}


def run(plan, output, seeds, steps, names=None):
    if not seeds or len(seeds) != len(set(seeds)) or not all(211 <= s <= 310 for s in seeds):
        raise ValueError("Use unique development seeds within 211--310 only")
    if not 100 <= steps <= 300:
        raise ValueError("Use 100--300 steps; only 300-step runs are full-protocol validation")
    candidates = json.loads(Path(plan).read_text())
    names = names or list(candidates)
    if len(names) != len(set(names)) or not names:
        raise ValueError("Candidate names must be unique and nonempty")
    if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in names):
        raise ValueError("Invalid candidate name")
    previous = json.loads((OFFLINE / "manifest.json").read_text())
    if previous["source_sha256"] != m.utils.digest(m.__file__):
        raise ValueError("p8 controller changed since the saved baseline")
    if previous["data_sha256"] != m.utils.digest(m.DATA):
        raise ValueError("Offline dataset changed")
    base = m.Config(**previous["config"])
    configs = {name: weight_config(base, candidates[name]) for name in names}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "tuning.lock").open("a+") as lock, threadpool_limits(limits=1):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        references, provenance = m.read_references()
        identity = {"controller_sha256": m.utils.digest(m.__file__),
                    "legacy_sha256": m.utils.digest(m.legacy.__file__),
                    "legacy_sha256_at_original_run": previous["legacy_algorithms_sha256"],
                    "utilities_sha256": m.utils.digest(m.utils.__file__),
                    "tuner_sha256": m.utils.digest(__file__),
                    "offline_model_sha256": m.utils.digest(OFFLINE / "offline_model.npz"),
                    "data_sha256": previous["data_sha256"], "references": provenance,
                    "note": "Weight-only serial experiment; shared frozen model; short runs are screening only"}
        plan_manifest = {**identity, "seeds": seeds, "steps": steps,
                         "candidates": {name: asdict(c) for name, c in configs.items()}}
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if json.loads(manifest_path.read_text()) != plan_manifest:
                raise ValueError("Existing experiment differs; choose a fresh directory")
        else:
            if list(output.glob("*/seed*_control_results.npz")):
                raise ValueError("Refusing unversioned result files")
            m.utils.write_json(manifest_path, plan_manifest)
        reports = {}
        started = time.perf_counter()
        try:
            for name, config in configs.items():
                directory = output / name
                directory.mkdir(exist_ok=True)
                m.utils.write_json(directory / "manifest.json", {**identity, "config": asdict(config), "seeds": seeds, "steps": steps})
                lib = m.Library.load(OFFLINE / "offline_model.npz", config)
                for seed in seeds:
                    path = directory / f"seed{seed}_control_results.npz"
                    if not path.exists():
                        with (directory / f"seed{seed}.log").open("a") as log, redirect_stdout(log):
                            m.run_seed(lib, seed, steps, directory, references[seed])
                    else:
                        with np.load(path, allow_pickle=False) as z:
                            old = json.loads(z["summary"].item())
                            if old["seed"] != seed or old["requested_steps"] != steps:
                                raise ValueError("Incompatible saved rollout")
                    report = compact_report(directory, seeds)
                    reports[name] = {"weights": {k: getattr(config, k) for k in sorted(ALLOWED)}, **report}
                    m.utils.write_json(output / "report.json", reports)
                    m.utils.write_json(output / "progress.json", {"state": "running", "candidate": name, "seed": seed,
                                                                   "elapsed_seconds": time.perf_counter()-started})
                    print(json.dumps({"candidate": name, "seed": seed, **{k:v for k,v in report.items() if k != "results"}}), flush=True)
            m.utils.write_json(output / "progress.json", {"state": "complete", "candidates": len(reports),
                                                           "elapsed_seconds": time.perf_counter()-started})
        except BaseException as exc:
            m.utils.write_json(output / "progress.json", {"state": "interrupted" if isinstance(exc, KeyboardInterrupt) else "error", "message": str(exc)})
            raise
    return reports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--candidates", nargs="+")
    args = parser.parse_args()
    run(args.plan, args.output, args.seeds, args.steps, args.candidates)
