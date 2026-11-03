"""Select the physical Reacher output channels from the offline Hankel matrix."""

import numpy as np
from atlas_deepc.control import expert_pipeline_search as pipe


def remove_joint_sum_channels(H10):
    """Drop the two derived channels at each lag, retaining every sample/input."""
    H10 = np.asarray(H10, dtype=float)
    input_rows = pipe.M * pipe.L
    expected_rows = (pipe.M + 10) * pipe.L
    if H10.ndim != 2 or H10.shape[0] != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows in the 10-output Hankel matrix.")
    outputs = H10[input_rows:].reshape(pipe.L, 10, H10.shape[1])
    return np.vstack([H10[:input_rows], outputs[:, :8].reshape(pipe.L * 8, H10.shape[1])])
