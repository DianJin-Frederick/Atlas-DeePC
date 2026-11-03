# Retained Experiment Artifacts

The local Reacher inventory was screened against `AtlasDeePC_v3.pdf` and the
retained notebooks on 2026-11-03. A directory can support a paper result without
being a separately reported experiment. Measurements and original provenance
remain unchanged. Files linked to Crucial X10 were not edited or deleted.

| Reacher directory | Reason retained |
|---|---|
| `four_baselines_500_mixed_k_t300_v1` | Main four-method 500-seed evaluation, Table 1 |
| `contextual_dpc_k64_rank53_terminal8_10001_10500_t300_v1` | Contextual-DPC 500-seed comparison, Table 2 |
| `select_dpc_upstream_final_clarabel_10001_10500_t300_v1` | Select-DPC 500-seed comparison, Table 2 |
| `matched_guidance_500_p8_k64_r53_t300_v1` | Controlled guidance comparison, Table 3 |
| `figure_guidance_p8_k64_r53_t300_v1` | Selected matched trajectories and baseline notebook section |
| `projection_chordal_rank_lamg0_lamdu0_clarabel_v1` | K=48 rank curves; includes the retained `rank060/` controls |
| `four_baselines_k64_rank_projection_chordal_v1` | K=64 rank curves and development-selected atlases |
| `large_k_rank_compact_v1` | K=80/96 rank curves and development-selected atlases |
| `chart_count_rank_projection_chordal_v1` | Original guided p=10 cluster-count curve |
| `low_k_missing_batches_t300_v3` | Completed missing low-K curves |
| `atlas_comparison_p8_k64_r53_t300_v1` | Full-data development comparison notebook and integrity checks |
| `contextual_dpc_atlas_terminal8_v1` | Frozen development weight selection and reference provenance |
| `select_dpc_upstream_final_clarabel_211_310_t300_v1` | Development comparison used by the full-data notebook |
| `select_dpc_our_data_clarabel_211_310_t100_v1` | Unsuccessful own-data comparison discussed in Appendix B.3.2 |

F1TENTH artifacts are retained for Appendix C. KDE-BDPC result storage is on the
external drive and was outside the local cleanup. Its active and suspended
experiment processes were not interrupted.

Removed locally: the older four-case baseline, standalone contextual tuning
report, abandoned low-K predecessor, historical `routed_deepc_iid` tree,
outcome-selected guidance debug reruns, superseded two-K plot exports, empty
300-step own-data Select run log, and the superseded Select development output.

A local cleanup manifest is in `docs/paper_artifact_cleanup_20261103.json`.
