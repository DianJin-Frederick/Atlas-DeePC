"""Paired 500-seed guidance study with p=8, K=64, rank=53 fixed."""

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.stats import binomtest

from atlas_deepc.ablations import figure_guidance_matched as matched
from atlas_deepc.common.reacher_config import normalize_saved_config

baseline = matched.baseline
runner = matched.runner
SEEDS = tuple(range(10001, 10501))
VERSION = "matched_guidance_500_p8_k64_r53_t300_v1"
RESULT_ROOT = baseline.ROOT / "experiments/reacher" / VERSION
PAIRED_DIRECTORY = baseline.RESULT_ROOT / "output8/terminal"
NOTE = (
    "Paired repeated evaluation on seeds 10001--10500; not a new untouched test set. "
    "p=8, K=64, rank=53, T_sim=300, atlas, weights and reset seeds are fixed. "
    "Guidance includes chordal waypoints and iterative N=10 terminal selection; "
    "the unguided policy uses a fixed nearest eligible terminal configuration. "
    "This isolates that policy package, not waypoints alone. Timings come from separate sessions.")


@contextmanager
def protocol_context():
    with patch.object(matched, "SEEDS", SEEDS), patch.object(matched, "VERSION", VERSION), \
            patch.object(matched, "RESULT_ROOT", RESULT_ROOT):
        with matched.protocol_context():
            yield


def prepare(root=RESULT_ROOT):
    root = Path(root)
    original = PAIRED_DIRECTORY / "manifest.json"
    previous = json.loads(original.read_text())
    value = matched.config()
    atlas = baseline.atlas_path(baseline.RESULT_ROOT, 8, "terminal")
    assert normalize_saved_config(previous["config"]) == {k: v for k, v in asdict(value).items() if k != "seed"}
    assert previous["evaluation_seeds"] == list(SEEDS)
    assert previous["atlas_sha256"] == runner.file_digest(atlas)
    assert previous["data_sha256"] == runner.file_digest(runner.DATA_PATH)
    for seed in SEEDS:
        path = PAIRED_DIRECTORY / f"seed{seed}_control_results.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing paired unguided trace: {path}")
    with protocol_context():
        signature = runner.case_signature(root, 8, "graph")
    for key in ("solver", "solver_options", "package_versions", "warmup_steps", "chart_selection_method"):
        if previous[key] != signature[key]:
            raise ValueError(f"Paired baseline differs in {key}; review before running a controlled comparison")
    signature.update(matched_study_sha256=runner.file_digest(__file__),
        rollout_runner_sha256=runner.file_digest(matched.__file__),
        paired_manifest_sha256=runner.file_digest(original),
        paired_directory=str(PAIRED_DIRECTORY), atlas_path=str(atlas), caveat=NOTE)
    path = root / "output8/graph/manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != signature:
            raise ValueError("Matched 500-seed source/protocol changed; use a fresh result directory")
    else:
        if list(path.parent.glob("seed*_control_results.npz")):
            raise ValueError("Unversioned guided traces; refusing to reuse them")
        runner.save_json(path, signature)
    return path


def read_outcome(path, seed, mode, manifest_hash):
    with np.load(path, allow_pickle=True) as trace:
        metadata = json.loads(trace["summary"].item())
        assert metadata["evaluation_signature_sha256"] == manifest_hash
        assert (metadata["seed"], metadata["output_dimension"], metadata["clusters"],
                metadata["svd_rank"], metadata["steps"], metadata["reference_mode"]) == (
                    seed, 8, 64, 53, 300, mode)
        distance = trace["distance"]
        assert distance.shape == (300,) and np.isfinite(distance).all()
        assert trace["y"].shape == (300, 8)
        assert trace["selection_windows"].shape == (300, 120)
        target = trace["final_goal"].reshape(-1)[6:8].copy()
        np.testing.assert_allclose(distance, np.linalg.norm(trace["y"][:, 6:8] - target, axis=1))
        success = bool(runner.strict_success(runner.true_error_metrics(distance)))
        assert success == bool(runner.strict_success(metadata))
        return dict(success=success, final100_mm=float(1000 * distance[-100:].mean()),
                    rollout_rmse_mm=float(1000 * np.sqrt(np.mean(distance**2)))), \
            trace["selection_windows"][0].copy(), target


def paired_rows(root=RESULT_ROOT):
    root = Path(root)
    original = PAIRED_DIRECTORY / "manifest.json"
    original_hash = runner.file_digest(original)
    guided_manifest = root / "output8/graph/manifest.json"
    guided_hash = None
    if guided_manifest.exists():
        signature = json.loads(guided_manifest.read_text())
        assert signature["paired_manifest_sha256"] == original_hash
        assert signature["evaluation_seeds"] == list(SEEDS)
        guided_hash = runner.file_digest(guided_manifest)
    rows = []
    for seed in SEEDS:
        unguided_path = PAIRED_DIRECTORY / f"seed{seed}_control_results.npz"
        guided_path = root / f"output8/graph/seed{seed}_control_results.npz"
        unguided, initial, target = read_outcome(unguided_path, seed, "terminal", original_hash)
        row = dict(seed=seed, unguided_path=str(unguided_path), guided_path=str(guided_path),
                   **{f"unguided_{key}": value for key, value in unguided.items()},
                   guided_success=None)
        if guided_path.exists():
            if guided_hash is None:
                raise ValueError("Guided trace exists without its protocol manifest")
            guided, other_initial, other_target = read_outcome(guided_path, seed, "graph", guided_hash)
            np.testing.assert_array_equal(initial, other_initial)
            np.testing.assert_array_equal(target, other_target)
            row.update({f"guided_{key}": value for key, value in guided.items()})
        rows.append(row)
    return rows


def statistics(rows, expected_seeds=SEEDS):
    if sorted(row["seed"] for row in rows) != sorted(expected_seeds):
        raise ValueError("Paired rows must contain each expected seed exactly once")
    pairs = [row for row in rows if row["guided_success"] is not None]
    complete = len(pairs) == len(expected_seeds)
    categories = {
        "both_success": (True, True), "repaired": (False, True),
        "regressed": (True, False), "both_failed": (False, False),
    }
    groups = {name: [r["seed"] for r in pairs
                    if (r["unguided_success"], r["guided_success"]) == outcome]
              for name, outcome in categories.items()}
    counts = {name: len(seeds) for name, seeds in groups.items()}
    total = len(pairs)
    discordant = counts["repaired"] + counts["regressed"]
    return dict(complete=complete, expected_pairs=len(expected_seeds), completed_pairs=total,
        paired_counts=counts, paired_seeds=groups,
        unguided_success_rate=(100 * (counts["both_success"] + counts["regressed"]) / total if total else None),
        guided_success_rate=(100 * (counts["both_success"] + counts["repaired"]) / total if total else None),
        improvement_percentage_points=(100 * (counts["repaired"] - counts["regressed"]) / total if total else None),
        exact_mcnemar_pvalue=(float(binomtest(counts["repaired"], discordant, .5).pvalue)
                             if complete and discordant else (1.0 if complete else None)),
        rates_scope="Completed pairs only; provisional until all 500 pairs finish",
        caveat=NOTE)


def report(root=RESULT_ROOT):
    rows = paired_rows(root)
    return dict(statistics(rows), rows=rows)


def run(root=RESULT_ROOT):
    # Reuse the existing serial, locked, validated, per-seed resumable runner.
    # No patches affect its controller or the baseline's saved files.
    with patch.object(matched, "SEEDS", SEEDS), patch.object(matched, "VERSION", VERSION), \
            patch.object(matched, "prepare", prepare), patch.object(matched, "paired_rows", paired_rows):
        matched.run(root)
    result = report(root)
    assert result["complete"]
    runner.save_json(Path(root) / "paired_summary.json", {k: v for k, v in result.items() if k != "rows"})
    print(json.dumps({k: v for k, v in result.items() if k not in ("rows", "paired_seeds")}, indent=2))
    return result


if __name__ == "__main__":
    run()
