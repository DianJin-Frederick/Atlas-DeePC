"""Shared success and true tracking-error metrics."""

import numpy as np

def strict_success(row):
    return bool(row["success_30"] and row["success_100"])


def true_error_metrics(distance):
    result = {"minimum": float(distance.min()), "minimum_step": int(distance.argmin()), "last": float(distance[-1])}
    for n in (30, 100):
        tail = distance[-n:]
        result.update({f"mean_last_{n}": float(tail.mean()), f"max_last_{n}": float(tail.max()),
                       f"success_{n}": bool(tail.mean() < .01 and tail.max() < .02)})
    return result


def full_rollout_rmse(distances):
    errors = np.asarray(distances, dtype=float)
    if errors.ndim != 2 or not errors.size or not np.all(np.isfinite(errors)):
        raise ValueError("Expected a finite, nonempty seeds-by-control-steps error array")
    per_seed = 1000 * np.sqrt(np.mean(errors ** 2, axis=1))
    pooled = float(1000 * np.sqrt(np.mean(errors ** 2)))
    return pooled, per_seed

