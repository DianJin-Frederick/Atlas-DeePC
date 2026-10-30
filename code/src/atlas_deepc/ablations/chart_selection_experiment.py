"""Paired full-window chart-selection study for the eight-output direct baseline."""

from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from atlas_deepc.atlas.chart_selection import FullWindowProjectionSelector
from atlas_deepc.baselines import unstandardized_atlas_deepc as experiment


METHODS = ("nearest_centroid", "full_window_projection")


def prepare_selector(config, atlas_path, data_path=experiment.baseline.DATA_PATH):
    with experiment.output_context(8), threadpool_limits(limits=1):
        hankel = experiment.load_hankel(data_path, output_dimension=8)
        atlas = experiment.routed.load_atlas(atlas_path)
        cache = experiment.routed.precompute_chart_matrix_cache(
            hankel, atlas, config, geometry="linear"
        )
        pipe = experiment.routed.pipe
        return FullWindowProjectionSelector(cache, pipe.M, 8, pipe.L)


def panel_directories(root, config):
    root = Path(root) / f"chart_selection_full_window_rank{config.svd_rank:03}_steps{config.steps}"
    return {method: root / method for method in METHODS}


def load_paired_panel(seeds, config, atlas_path, root,
                      data_path=experiment.baseline.DATA_PATH):
    panels = {}
    for method, directory in panel_directories(root, config).items():
        panels[method] = experiment.load_panel(
            seeds, config, atlas_path, directory, data_path,
            output_dimension=8, reference_mode="terminal",
            chart_selection_method=method, warmup_steps=experiment.routed.pipe.L,
        )
    return panels


def run_paired_panel(seeds, config, atlas_path, root,
                     data_path=experiment.baseline.DATA_PATH):
    """Run or resume both selectors with the same L-step zero-input warmup."""
    results = {}
    for method, directory in panel_directories(root, config).items():
        results[method] = experiment.run_panel(
            seeds, config, atlas_path, directory, data_path,
            output_dimension=8, reference_mode="terminal",
            chart_selection_method=method, warmup_steps=experiment.routed.pipe.L,
        )
    return results


def rank_audit(selector):
    return {
        "charts": len(selector.chart_ids),
        "window_entries": selector.window_dimension,
        "minimum_chart_rank": int(np.min(selector.ranks)),
        "maximum_chart_rank": int(np.max(selector.ranks)),
        "all_charts_proper_subspaces": bool(np.all(selector.ranks < selector.window_dimension)),
    }
