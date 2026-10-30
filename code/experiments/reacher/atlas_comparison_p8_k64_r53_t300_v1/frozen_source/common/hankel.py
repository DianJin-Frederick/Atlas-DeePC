"""Reconstruct overlapping Reacher output windows without duplicating noise."""

import numpy as np
from atlas_deepc.atlas import routed_deepc_experiment as routed
pipe = routed.pipe


def remove_joint_sum_channels(H10):
    """Drop the two derived channels at each lag, retaining every sample/input."""
    H10 = np.asarray(H10, dtype=float)
    input_rows = pipe.M * pipe.L
    expected_rows = (pipe.M + 10) * pipe.L
    if H10.ndim != 2 or H10.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in the 10-output Hankel matrix.")
    outputs = H10[input_rows:].reshape(pipe.L, 10, H10.shape[1])
    return np.vstack([H10[:input_rows], outputs[:, :8].reshape(pipe.L * 8, H10.shape[1])])

def episode_outputs(H):
    """Recover unique time samples rather than independently corrupting overlaps."""
    windows = routed.WINDOWS_PER_EPISODE
    assert H.shape[1] % windows == 0
    blocks = H[pipe.M*pipe.L:].reshape(pipe.L, pipe.P, -1, windows).transpose(2, 3, 0, 1)
    outputs = np.concatenate([blocks[:, :, 0], blocks[:, -1, 1:]], axis=1)
    rebuilt = output_hankel(outputs)
    np.testing.assert_array_equal(rebuilt, H[pipe.M*pipe.L:])
    return outputs


def output_hankel(outputs):
    windows = outputs.shape[1] - pipe.L + 1
    blocks = np.stack([outputs[:, lag:lag+windows] for lag in range(pipe.L)])
    return blocks.transpose(0, 3, 1, 2).reshape(pipe.P*pipe.L, -1)
