"""Render one study trial with its original perturbation and verify state replay."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from pusher_contribution_study import (
    apply_condition, displace_object, load_npz, make_env, state_from_env, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = load_npz(args.rollout)
    condition = str(result["condition"])
    stage_dir = args.rollout.parents[2]
    if stage_dir.name == "recovery_pilot":
        spec = json.loads((stage_dir / "protocol.json").read_text())["condition"]
    else:
        protocol_name = "final_protocol.json" if stage_dir.name == "final" else "protocol.json"
        protocol = json.loads((args.rollout.parents[3] / protocol_name).read_text())
        spec = protocol["conditions"][condition]
    env, _ = make_env(int(result["seed"]), 250, "rgb_array")
    apply_condition(env, condition, spec)
    errors = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(args.output, fps=20, codec="libx264", quality=7) as writer:
            writer.append_data(env.render())
            for action in result["warmup_action"]:
                env.step(action.astype(np.float32))
                writer.append_data(env.render())
            for step, action in enumerate(result["action"]):
                if condition == "disturbance" and step == spec["step"]:
                    displace_object(env, result["disturbance_delta_xy"])
                env.step(action.astype(np.float32))
                errors.append(float(np.max(np.abs(state_from_env(env) - result["state"][step]))))
                writer.append_data(env.render())
    finally:
        env.close()
    report = {"rollout": str(args.rollout), "video": str(args.output),
              "max_state_error": max(errors), "condition": condition,
              "verified": max(errors) < 1e-9}
    write_json(args.output.with_suffix(".json"), report)
    assert report["verified"], "Video replay differs from experimental trajectory"
    print(json.dumps(report))


if __name__ == "__main__":
    main()
