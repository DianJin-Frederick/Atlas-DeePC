# F1TENTH experiments

[Paper notebook](baselines/baseline_f1tenth_atlas_deepc.ipynb): one frozen Atlas-DeePC configuration per noise level versus Contextual-DPC with 30/60/90 columns, using the same noisy offline data. The table uses seven noise levels and 20 fresh paired seeds per setting. `RUN_ALL_EXPERIMENTS=False` reads saved results; enabling it reproduces the 560 paper rollouts into a fresh folder.

[Protocol](../../docs/f1tenth/README.md) · [Paper table](../../experiments/f1tenth/paper_nsr_20261027/report.md)

The configuration selection is recorded separately from the fresh evaluation. Superseded raw experiments are removed after verification; retained raw records include every successful and failed paper rollout.
