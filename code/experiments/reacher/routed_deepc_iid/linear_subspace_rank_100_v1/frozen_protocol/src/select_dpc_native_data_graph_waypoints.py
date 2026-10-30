"""Run native-data, original-form Select-DPC on Atlas graph waypoints."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from atlas_select_shared_waypoint_comparison import (
    DEFAULT_ARCHIVE,
    Protocol,
    build_select_controller,
    extract_reference_package,
    import_reference_select_dpc,
    load_native_select_trajectories,
    run_select_on_waypoints,
    sha256_file,
)


DEFAULT_REFERENCE = Path(
    "experiments/select_dpc_comparison/shared_data_waypoints/atlas_control_results.npz"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/select_dpc_comparison/native_data_graph_waypoints"
)


def trajectory_digest(inputs: list[np.ndarray], outputs: list[np.ndarray]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for u, y in zip(inputs, outputs):
        digest.update(np.ascontiguousarray(u, dtype=np.float64).tobytes())
        digest.update(np.ascontiguousarray(y, dtype=np.float64).tobytes())
    return digest.hexdigest()


def make_figure(output_dir: Path, atlas: np.lib.npyio.NpzFile, select: np.lib.npyio.NpzFile):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    for label, data, xy_indices, color in (
        ("Atlas-DeePC", atlas, (6, 8), "tab:blue"),
        ("Select-DPC native data", select, (4, 6), "tab:green"),
    ):
        axes[0].plot(1000 * data["target_error"], label=label, color=color)
        axes[1].plot(1000 * data["waypoint_error"], label=label, color=color)
        axes[2].plot(1000 * data["control_compute_time_s"], label=label, color=color)
    axes[0].set(title="Final-target tracking", xlabel="Control step", ylabel="Error [mm]")
    axes[1].set(title="Graph-waypoint tracking", xlabel="Control step", ylabel="Error [mm]")
    axes[2].set(title="End-to-end online compute", xlabel="Control step", ylabel="Time [ms]", yscale="log")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle("Select-DPC native offline data with the identical Atlas waypoint schedule")
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=180, bbox_inches="tight")
    fig.savefig(output_dir / "comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_data = np.load(args.reference_npz, allow_pickle=True)
    references = np.asarray(reference_data["ref"], dtype=float)
    atlas_initial = np.asarray(reference_data["initial_obs"], dtype=float)
    atlas_summary = json.loads(reference_data["summary"].item())
    protocol = Protocol(
        seed=args.seed,
        steps=int(references.shape[0]),
        target_x=args.target_x,
        target_y=args.target_y,
        select_columns=args.select_columns,
        select_refinements=args.select_refinements,
    )

    with tempfile.TemporaryDirectory(prefix="choose_wisely_native_") as temp:
        reference_root = extract_reference_package(args.select_code_zip, Path(temp))
        modules = import_reference_select_dpc(reference_root)
        inputs, outputs, native_metadata = load_native_select_trajectories(reference_root)
        controller, setup_time, controller_metadata = build_select_controller(
            inputs, outputs, protocol, modules
        )
        select_path = args.output_dir / "select_control_results.npz"
        select_summary, select_initial = run_select_on_waypoints(
            controller,
            references,
            protocol,
            atlas_initial,
            setup_time,
            select_path,
        )
        select_summary["controller"] = (
            "original Select-DPC formulation with its supplied native IID trajectories"
        )

    select_data = np.load(select_path, allow_pickle=True)
    reference_mapping_error = float(
        np.max(np.abs(references[:, [0, 1, 2, 3, 6, 7, 4, 5]] - select_data["ref"]))
    )
    latency_ratio = (
        select_summary["median_control_compute_ms"]
        / atlas_summary["median_control_compute_ms"]
    )
    payload = {
        "benchmark": "Native-data original-form Select-DPC on the frozen Atlas graph-waypoint schedule",
        "protocol": {
            **asdict(protocol),
            "environment": "Reacher-v4-custom (no rendering)",
            "reference_source": str(args.reference_npz.resolve()),
            "reference_protocol": "exact replay of the causal 100-step Atlas waypoint schedule",
            "timing_boundary": "complete controller compute; MuJoCo step excluded",
        },
        "select_native_data": {
            **native_metadata,
            **controller_metadata,
            "trajectory_sha256_after_notebook_transformation": trajectory_digest(inputs, outputs),
        },
        "source_provenance": {
            "archive": str(args.select_code_zip.resolve()),
            "archive_sha256": sha256_file(args.select_code_zip),
            "source_use": (
                "source and IID dataset loaded from the supplied archive; original "
                "controller settings plus disclosed exhausted-fallback safety guard"
            ),
        },
        "fairness_checks": {
            "initial_observation_max_abs_difference": float(
                np.max(np.abs(atlas_initial - select_initial))
            ),
            "reference_mapping_max_abs_error": reference_mapping_error,
            "same_online_seed_target_initial_state_and_reference": True,
            "same_offline_data": False,
        },
        "results": [atlas_summary, select_summary],
        "comparison": {
            "median_latency_ratio_select_over_atlas": float(latency_ratio),
            "qp_count_ratio_select_over_atlas": float(
                select_summary["qp_count"] / atlas_summary["qp_count"]
            ),
            "interpretation": (
                "Native implementation comparison with identical online task and "
                "reference, but intentionally method-specific offline datasets."
            ),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    make_figure(args.output_dir, reference_data, select_data)
    print(json.dumps(payload["comparison"], indent=2), flush=True)
    print(
        f"Select-DPC native data: median={select_summary['median_control_compute_ms']:.2f} ms, "
        f"p95={select_summary['p95_control_compute_ms']:.2f} ms, "
        f"final={1000 * select_summary['final_target_error_m']:.2f} mm, "
        f"statuses={select_summary['final_iteration_statuses']}",
        flush=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-npz", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--select-code-zip", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-x", type=float, default=0.043)
    parser.add_argument("--target-y", type=float, default=0.092)
    parser.add_argument("--select-columns", type=int, default=200)
    parser.add_argument("--select-refinements", type=int, default=5)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
