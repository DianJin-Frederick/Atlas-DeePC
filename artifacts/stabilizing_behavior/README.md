# Stabilizing Behavior Artifacts

These artifacts use stabilizing/expert-style offline data.

They are a positive-control sanity check: if successful reaching behavior is present in the offline library, DeePC can recover/reuse it and reach the target.

Main verified video:

- `seed42_control.mp4`

Main verified result:

- `seed42_control_results.npz`

Saved metrics for seed 42:

- `success = true`
- `mean_last_30 = 0.005343914380203674`
- `max_last_30 = 0.01337696219260556`
- `last = 0.005655749407056843`

This is useful evidence for reusing successful behavior through atlas/subspace DeePC. It is not the clean fair IID random-data claim, but it is still an intentional and meaningful sanity check.
