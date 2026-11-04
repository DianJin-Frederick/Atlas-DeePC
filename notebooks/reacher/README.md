# Reacher Experiments

The retained notebooks use the original offline data and explicit saved-result
paths. Paper evaluation, development studies, and historical result audits are
kept distinct. Other comparison algorithms live in `src/comparison_methods/`,
separate from Atlas-DeePC controllers.

## Baselines

- [Main Reacher baseline](baselines/baseline_graph_routed_atlas_deepc.ipynb)
- [Slow-motion videos and matched guidance repairs](baselines/video_display.ipynb)

## Comparisons

- [Full-data Atlas-DeePC comparison](comparisons/atlas_deepc.ipynb)
- [Contextual-DPC comparison](comparisons/contextual_dpc.ipynb)
- [Historical KDE tuning results (read-only audit)](comparisons/kde_bdpc_multistep_tuning_audit.ipynb)
- [Faithful KDE-BDPC](comparisons/kde_bdpc_reacher.ipynb)
- [Select-DPC with original data](comparisons/select_dpc_original.ipynb)
- [Select-DPC with Atlas offline data (development)](comparisons/select_dpc_our_data_development.ipynb)

## Ablations

- [Complete chart-count and rank ablation (all K)](ablations/ablation_chart_count_and_rank.ipynb)
- [Effect of graph guidance](ablations/effect_of_graph_guidance.ipynb)

## Main Baseline

The four configurations use p=8 without/with guidance and p=10 without/with
guidance, respectively: K=64/96/80/64 and rank=53/57/64/62. Evaluation uses
seeds 10001--10500 with 300 controlled steps. Development seeds 211--310
were used for configuration selection. Previously evaluated test seeds are
not a new untouched test set.

Run experiments deliberately using the notebook controls; keep timing runs
serial and avoid concurrent heavy jobs. Existing protocol/source hashes guard
against mixing changed code or settings into saved runs.

## Data and Implementation

- [Offline data](../../data/reacher/)
- [Saved experiment results](../../experiments/reacher/)
- [Code map](../../src/README.md)

Many saved-result files are links to the external SSD. Local archive removal
does not change their targets or the retained experiments.
