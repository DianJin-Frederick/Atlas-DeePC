"""Independently audit the completed frozen panel; never run or retune control."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "experiments/pusher_atlas/contribution_study_20260903"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_trial(path):
    with np.load(path, allow_pickle=False) as archive:
        return {k: archive[k] for k in archive.files}


def main():
    p = json.loads((STUDY / "final_protocol.json").read_text())
    lock = json.loads((STUDY / "frozen_protocol.json").read_text())
    summary = json.loads((STUDY / "final/summary.json").read_text())
    seeds = p["final_seeds"]
    methods = list(p["methods"])
    conditions = list(p["conditions"])
    expected = {(condition, method, seed) for condition in conditions for method in methods for seed in seeds}
    actual = set()
    for path in (STUDY / "final").glob("*/*/seed*.npz"):
        actual.add((path.parents[1].name, path.parent.name, int(path.stem.removeprefix("seed"))))
    assert actual == expected, {"missing": sorted(expected - actual), "extra": sorted(actual - expected)}
    assert len(actual) == 2000 and len(seeds) == len(set(seeds)) == 100
    assert summary["completed_trials"] == len(actual)
    assert digest(STUDY / "final_protocol.json") == lock["protocol_sha256"]
    for category in ("source_sha256", "artifacts_sha256"):
        for path, checksum in lock["provenance"][category].items():
            assert digest(ROOT / path) == checksum, path
    csv_rows = list(csv.DictReader((STUDY / "final/per_trial.csv").open()))
    assert len(csv_rows) == len(actual)
    aggregates = {(r["condition"], r["method"]): r for r in summary["aggregates"]}
    assert len(aggregates) == 20
    recomputed = {}
    trial_checksums = {}
    observed_error_trials = []
    counts = {"trials": 0, "controller_steps": 0, "qp_calls": 0,
              "solver_nonoptimal": 0, "solver_inaccurate": 0, "solver_retries": 0,
              "fallback_steps": 0, "applied_torque_violations": 0,
              "joint_limit_violation_steps": 0}
    for condition in conditions:
        for method in methods:
            group_successes, finals, steps = [], [], []
            for seed in seeds:
                path = STUDY / "final" / condition / method / f"seed{seed}.npz"
                r = read_trial(path)
                row = json.loads(path.with_suffix(".json").read_text())
                assert str(r["condition"]) == condition and str(r["method"]) == method and int(r["seed"]) == seed
                assert np.array_equal(r["warmup_action"], np.zeros((2, 7)))
                assert float(r["dt"]) == .05
                n = len(r["action"])
                complete = n == p["base_config"]["rollout_steps"] and not str(r["error"])
                if not complete:
                    observed_error_trials.append({"condition": condition, "method": method, "seed": seed,
                                                  "steps": n, "error": str(r["error"])})
                assert np.all(np.isfinite(r["state"])) and np.all(np.isfinite(r["action"]))
                distance = np.linalg.norm(r["state"][:, 17:19] - r["goal_xy"], axis=1)
                assert np.allclose(distance, r["distance"], rtol=0, atol=1e-12)
                success = bool(complete and np.all(distance[-8:] <= .08))
                assert success == row["success"] and complete == row["complete"]
                assert abs(float(distance[-1]) - row["final_distance_m"]) <= 1e-12
                assert np.max(np.abs(r["action"])) <= 2.0
                if method == "A_policy":
                    assert np.array_equal(r["action"], np.clip(r["prior"], -2, 2))
                    assert np.all(r["chart"] == -1)
                elif method == "B_predictive":
                    assert np.array_equal(r["prior"], np.zeros((n, 7)))
                physics = json.loads(str(r["physics_json"]))
                assert physics["xml_sha256"] == lock["provenance"]["pusher_xml_sha256"]
                assert physics["all_contact_dimensions"] == [1]
                assert physics["gravity"] == [0., 0., 0.]
                assert physics["object_dof_armature"] == [.04, .04]
                assert physics["object_dof_damping"] == [.5, .5]
                if condition == "mass":
                    assert abs(physics["object_mass_kg"] - .05) < 1e-12
                else:
                    assert abs(physics["object_mass_kg"] - 7.853981633974484e-6) < 1e-15
                assert physics["object_dof_frictionloss"] == ([.02, .02] if condition == "friction" else [0., 0.])
                if condition == "disturbance":
                    assert abs(np.linalg.norm(r["disturbance_delta_xy"]) - .10) < 1e-12
                    onset = p["conditions"]["disturbance"]["step"]
                    pre = bool(n >= onset and np.all(distance[onset-8:onset] <= .08))
                    eligible = pre and float(r["disturbance_post_distance"]) > .08
                    assert pre == row["pre_disturbance_sustained_success"]
                    assert eligible == row["recovery_eligible"]
                statuses = r["status"].astype(str)
                qp = method != "A_policy"
                counts["trials"] += 1
                counts["controller_steps"] += n
                counts["qp_calls"] += n if qp else 0
                counts["solver_nonoptimal"] += int(np.sum(~np.char.startswith(statuses, "optimal"))) if qp else 0
                counts["solver_inaccurate"] += int(np.sum(np.char.find(statuses, "inaccurate") >= 0))
                counts["solver_retries"] += int(np.sum(np.char.find(statuses, "clarabel_retry") >= 0))
                counts["fallback_steps"] += int(np.sum(r["fallback"]))
                counts["applied_torque_violations"] += int(np.sum(r["actuator_violation"] > 1e-6))
                counts["joint_limit_violation_steps"] += int(np.sum(r["joint_violation_rad"] > 1e-4))
                group_successes.append(success)
                finals.append(float(distance[-1]))
                steps.append(n)
                trial_checksums[str(path.relative_to(STUDY))] = digest(path)
            agg = aggregates[(condition, method)]
            assert agg["n"] == 100 and agg["successes"] == sum(group_successes)
            assert abs(agg["mean_final_distance_m"] - np.mean(finals)) < 1e-12
            recomputed[(condition, method)] = np.asarray(group_successes, dtype=bool)

    # Intervention validation: paired nominal and disturbance runs must be
    # identical until the declared intervention. This uses states/actions, not rewards.
    max_prefix_state_error = max_prefix_action_error = 0.0
    onset = p["conditions"]["disturbance"]["step"]
    for method in methods:
        for seed in seeds:
            nominal = read_trial(STUDY / "final/nominal" / method / f"seed{seed}.npz")
            disturbed = read_trial(STUDY / "final/disturbance" / method / f"seed{seed}.npz")
            n = min(onset, len(nominal["state"]), len(disturbed["state"]))
            max_prefix_state_error = max(max_prefix_state_error, float(np.max(np.abs(nominal["state"][:n] - disturbed["state"][:n]))))
            max_prefix_action_error = max(max_prefix_action_error, float(np.max(np.abs(nominal["action"][:n] - disturbed["action"][:n]))))
    assert max_prefix_state_error < 1e-9 and max_prefix_action_error < 1e-9
    for condition in conditions:
        a = recomputed[(condition, "C_combined")]
        for comparator in ("A_policy", "B_predictive", "D_selection", "E_regularization"):
            b = recomputed[(condition, comparator)]
            win, lose = int(np.sum(a & ~b)), int(np.sum(~a & b))
            paired = summary["paired_comparisons"][f"{condition}:C_minus_{comparator}"]
            assert paired["n_pairs"] == 100
            assert paired["combined_only"] == win and paired["comparator_only"] == lose
            assert abs(paired["success"]["mean_difference"] - (win - lose)) < 1e-12
            expected_p = binomtest(win, win + lose, .5).pvalue if win + lose else 1.
            assert abs(expected_p - paired["mcnemar_exact_p"]) < 1e-14

    report = {"verified": True, "counts": counts, "incomplete_or_error_trials": observed_error_trials,
              "nominal_disturbance_matched_prefix_pairs": 500,
              "matched_prefix_steps": onset,
              "max_prefix_state_error": max_prefix_state_error,
              "max_prefix_action_error": max_prefix_action_error,
              "final_protocol_sha256": digest(STUDY / "final_protocol.json"),
              "analysis_script_sha256": digest(Path(__file__)), "trial_sha256": trial_checksums}
    output = STUDY / "final/independent_verification.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "trial_sha256"}, indent=2))


if __name__ == "__main__":
    main()
