"""Full-data p=8, K=64, rank-53 Atlas comparison on seeds 211--310."""

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import numpy as np

from atlas_deepc.benchmarks import reacher_heldout as runner
from atlas_deepc.benchmarks import reacher_mixed_k as selected

ROOT = runner.ROOT
RESULT_ROOT = ROOT / "experiments/reacher/atlas_comparison_p8_k64_r53_t300_v1"
CONTEXTUAL_ROOT = ROOT / "experiments/reacher/contextual_dpc_atlas_terminal8_v1/serial_211_310_tuned_t300_v1"
SELECT_ROOT = ROOT / "experiments/reacher/select_dpc_upstream_final_clarabel_211_310_t300_v1"
SEEDS = tuple(range(211, 311))
CONTROL_STEPS = 300
VERSION = "atlas_comparison_p8_k64_r53_t300_v1"
NOTE = ("Development comparison, not held-out performance: seeds 211--310 selected the rank. "
        "Full traces retained. Timings include logging overhead and shared-computer contention; "
        "the separate 500-seed evaluation may be running concurrently.")
_base_signature = runner.case_signature


def validate_seeds(seeds):
    if tuple(seeds) != SEEDS:
        raise ValueError("This comparison uses exactly seeds 211--310")


def signature(root, p, mode):
    result = _base_signature(root, p, mode)
    result.update(full_data_runner_sha256=runner.file_digest(__file__),
                  retained_data="All 300 steps, predicted horizons, g, sigma, initial histories, timings, geometry",
                  comparison_seeds=list(SEEDS), timing_caveat=NOTE)
    return result


@contextmanager
def protocol_context():
    with selected.protocol_context(), ExitStack() as stack:
        for name, value in {"CASES": ((8, "terminal", 53),), "SEEDS": SEEDS,
                            "VERSION": VERSION, "EVALUATION_NOTE": NOTE,
                            "validate_seeds": validate_seeds, "case_signature": signature}.items():
            stack.enter_context(patch.object(runner, name, value))
        yield


def prepare(root=RESULT_ROOT):
    with protocol_context():
        protocol = runner.prepare(root)
    for source in (Path(__file__), Path(selected.__file__)):
        shutil.copy2(source, Path(root) / "frozen_source" / source.name)
    return protocol


def config():
    return selected.config_for(8, "terminal")


def panel_directory(root=RESULT_ROOT):
    return Path(root) / "output8/terminal"


def atlas_path(root=RESULT_ROOT):
    return selected.atlas_path(root, 8, "terminal")


def logging_controller(original, records):
    class LoggedController(original):
        def solve(self, matrix, u_ini, y_ini, reference):
            answer = super().solve(matrix, u_ini, y_ini, reference)
            u, y, g, status, fit_residual = answer
            row = {"u_ini": u_ini.ravel().copy(), "y_ini": y_ini.ravel().copy(),
                   "planned_u": np.full((10, 2), np.nan), "planned_y": np.full((10, 8), np.nan),
                   "g_coefficients": np.full(53, np.nan), "slack": np.full(16, np.nan),
                   "qp_equality_residual": np.nan, "objective": np.nan,
                   "initial_fit_residual": fit_residual}
            if g is not None:
                offset, basis, _, _ = runner.experiment.routed.pipe.unpack_basis(matrix)
                behavior = offset + basis @ g
                sigma = behavior[24:40] - y_ini
                row.update(planned_u=u.reshape(10, 2).copy(), planned_y=y.reshape(10, 8).copy(),
                           g_coefficients=g.ravel().copy(), slack=sigma.ravel().copy(),
                           qp_equality_residual=float(np.linalg.norm(
                               behavior - np.vstack([u_ini, u, y_ini+sigma, y]))),
                           objective=float(np.sum(((y-reference).reshape(10, 8) @ self.Q)
                                                   * (y-reference).reshape(10, 8))
                                           + np.sum((u.reshape(10, 2) @ self.R)*u.reshape(10, 2))
                                           + self.lam_ini*np.sum(sigma**2)))
            records.append(row)
            return answer
    return LoggedController


def enrich(arrays, row, selector, records):
    assert len(records) == CONTROL_STEPS
    arrays.update({key: np.asarray([record[key] for record in records]) for key in records[0]})
    arrays.update(selection_windows=np.asarray(selector.windows),
                  selection_residuals=np.asarray(selector.residual_history),
                  selection_tie_count=np.asarray(selector.tie_history))
    arrays["warmup_u"] = arrays["selection_windows"][0, :24].reshape(12, 2).copy()
    arrays["warmup_y"] = arrays["selection_windows"][0, 24:].reshape(12, 8).copy()
    arrays["y_before"] = arrays["selection_windows"][:, -8:].copy()
    arrays["target_xy"] = arrays["final_goal"][6:8].copy()
    c = config()
    q = np.array([c.q_data_goal_joint]*4 + [c.q_dq]*2 + [c.q_xy]*2)
    arrays["stage_tracking_cost"] = np.sum((arrays["y"]-arrays["ref"])**2*q, axis=1)
    arrays["stage_input_cost"] = c.r*np.sum(arrays["u"]**2, axis=1)
    arrays["stage_control_cost"] = arrays["stage_tracking_cost"]+arrays["stage_input_cost"]
    row.update(full_data_retained=True, strict_success=runner.strict_success(row),
               rollout_rmse_mm=float(1000*np.sqrt(np.mean(arrays["distance"]**2))))
    return arrays, row


def validate_full_trace(path, signature_hash):
    with protocol_context():
        row, distance, statuses = runner.validate_trace(path, 8, "terminal", signature_hash)
    with np.load(path, allow_pickle=True) as trace:
        for key, shape in {"planned_u": (300, 10, 2), "planned_y": (300, 10, 8),
                           "g_coefficients": (300, 53), "slack": (300, 16),
                           "u_ini": (300, 4), "y_ini": (300, 16)}.items():
            assert trace[key].shape == shape, key
        solved = np.isin(trace["status"], ["optimal", "optimal_inaccurate"])
        np.testing.assert_allclose(trace["planned_y"][solved, 0], trace["pred"][solved])
        np.testing.assert_allclose(np.clip(trace["planned_u"][solved, 0], -1, 1), trace["u"][solved])
        np.testing.assert_allclose(trace["stage_control_cost"].sum(), row["closed_loop_control_cost"])
        np.testing.assert_allclose(trace["warmup_u"], 0)
    return row, distance, statuses


def audit_comparison(root=RESULT_ROOT):
    """Check reset/target pairing and disclose differing reference conventions."""
    with np.load(CONTEXTUAL_ROOT / "references.npz") as bundle:
        assert list(bundle["seeds"]) == list(SEEDS)
        references = bundle["references"].copy()
    matched, mismatches, select_initial_differences = [], [], []
    for index, seed in enumerate(SEEDS):
        path = panel_directory(root) / f"seed{seed}_control_results.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=True) as atlas,\
                np.load(CONTEXTUAL_ROOT / path.name) as contextual,\
                np.load(SELECT_ROOT / path.name) as select:
            np.testing.assert_allclose(atlas["target_xy"], contextual["target_xy"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(atlas["target_xy"], select["target_xy"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(atlas["warmup_y"], contextual["warmup_y"], rtol=0, atol=1e-12)
            initial_delta = atlas["y_before"][0]-select["y_before"][0]
            select_initial_differences.append({"seed": seed,
                "max_output_difference": float(np.max(np.abs(initial_delta))),
                "fingertip_difference_mm": float(1000*np.linalg.norm(initial_delta[6:8]))})
            (matched if np.allclose(atlas["final_goal"], references[index], rtol=0, atol=1e-12)
             else mismatches).append(seed)
    audit = {"checked_seeds": len(matched)+len(mismatches), "contextual_reference_matching_seeds": matched,
             "contextual_reference_mismatch_seeds": mismatches, "contextual_initial_states_match": True,
             "cartesian_targets_match_all_methods": True,
             "select_initial_state_differences": select_initial_differences,
             "note": "Select-DPC uses Reacher-v4 without physical warmup, versus Reacher-v5 and 12 zero-input steps here. It retains its original offline data, horizon, reference, constraints, and weights. This is not an otherwise-identical benchmark; native weighted costs are not directly comparable."}
    runner.save_json(Path(root) / "comparison_audit.json", audit)
    return audit


def report(root=RESULT_ROOT):
    root = Path(root)
    directory = panel_directory(root)
    files = [directory / f"seed{s}_control_results.npz" for s in SEEDS]
    completed = sum(p.exists() for p in files)
    result = {"completed": completed, "requested": len(SEEDS), "complete": completed == len(SEEDS),
              "steps": CONTROL_STEPS, "charts": 64, "rank": 53, "output_dimension": 8,
              "graph_guidance": False, "note": NOTE}
    if result["complete"]:
        signature_hash = runner.file_digest(directory / "manifest.json")
        for path in files:
            validate_full_trace(path, signature_hash)
        with protocol_context():
            result["metrics"] = runner.summarize_case(root, 8, "terminal")
        result["comparison_audit"] = audit_comparison(root)
    if root.exists():
        runner.save_json(root / "evaluation_report.json", result)
    return result


def run_all(root=RESULT_ROOT):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root)
        session = {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
                   "workers": 1, "numerical_threads": 1, "note": NOTE}
        runner.save_json(root / "run_status.json", session)
        directory = panel_directory(root)
        signature_hash = runner.file_digest(directory / "manifest.json")
        records = []
        original = runner.experiment.routed.pipe.DeePCController
        try:
            with protocol_context(), runner.controller_context(root, 8, "terminal") as (controller, selector, _, _),\
                    patch.object(runner.experiment.routed.pipe, "DeePCController", logging_controller(original, records)):
                np.savez_compressed(directory / "selection_geometry.npz", bases=selector.bases,
                                    chart_ids=selector.chart_ids, ranks=selector.ranks)
                for index, seed in enumerate(SEEDS, 1):
                    path = directory / f"seed{seed}_control_results.npz"
                    if path.exists():
                        validate_full_trace(path, signature_hash)
                        continue
                    if shutil.disk_usage(root).free < 2*2**30:
                        raise RuntimeError("Less than 2 GiB free; complete rollouts are retained")
                    selector.reset()
                    records.clear()
                    started = time.perf_counter()
                    pending = path.with_suffix(".pending.npz")
                    row = controller.run(seed, pending, reference_mode="terminal",
                                         chart_selector=selector, localization_history_length=12)
                    row.update(evaluation_version=VERSION, evaluation_signature_sha256=signature_hash,
                               solver="CLARABEL", output_dimension=8,
                               chart_selection_method="full_window_projection", graph_cost_method="not_used",
                               elapsed_rollout_wall_s=time.perf_counter()-started)
                    with np.load(pending, allow_pickle=True) as trace:
                        arrays = {k: trace[k] for k in trace.files if k != "summary"}
                    arrays, row = enrich(arrays, row, selector, records)
                    np.savez_compressed(pending, **arrays, summary=np.array(json.dumps(row)))
                    pending.replace(path)
                    validate_full_trace(path, signature_hash)
                    if index == 1:
                        audit_comparison(root)
                    runner.save_json(root / "progress.json", {"completed": index, "requested": len(SEEDS),
                                     "last_seed": seed, "updated_utc": datetime.now(timezone.utc).isoformat()})
                    print(f"{index}/{len(SEEDS)} seed={seed}, success={row['strict_success']}, "
                          f"final100={1000*row['mean_last_100']:.3f} mm", flush=True)
            result = report(root)
            session["status"] = "complete"
            return result
        except BaseException as error:
            session.update(status="stopped", error=str(error))
            raise
        finally:
            session["updated_utc"] = datetime.now(timezone.utc).isoformat()
            runner.save_json(root / "run_status.json", session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    print(prepare() if args.prepare_only else run_all())
