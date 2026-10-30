# Retained Routed-DeePC Artifacts

This directory contains retained historical experiment dependencies, not the
current baseline protocol. Keep each study's data and media inside its own folder.

- `baseline_linear_iterative_n10_v1/videos/`: historical baseline videos used by
  `website/video-sources.json`; their locations are unchanged.
- `clustering_jointconfig48_fresh211_310/media/`: physical-configuration K=48
  dashboards, posters, metadata, and figures previously loose in this directory.
- `linear_subspace_rank_100_v1/`: retained historical rank-study artifacts.
- `unstandardized_shared_runner_v6_output8/` and
  `unstandardized_shared_runner_v6_output10/`: retained shared-runner artifacts.

Other loose historical media have moved to
`../../../archive/experiments/routed_deepc_iid/`, grouped by originating study.
That archive also holds the old README as `historical_README.md` and the move
manifest `media_relocation_2026-10-22.json` with before/after checksums.

Dashboard metadata now points to the relocated videos and posters. Original
source-result paths and hashes remain unchanged for provenance; some source
rollouts and atlases were deleted in earlier cleanup and are not restored here.
The two `measurement_*_study.lock` files remain at their established runtime
locations because noise-runner code uses them for mutual exclusion.
