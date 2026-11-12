# F1TENTH experiments

[Paper notebook](baselines/baseline_f1tenth_atlas_deepc.ipynb): one frozen Atlas-DeePC configuration per noise level versus Contextual-DPC with 30/60/90 columns, using the same noisy offline data. The table uses seven noise levels and 20 fresh paired seeds per setting. `RUN_ALL_EXPERIMENTS=False` reads saved results; enabling it reproduces the 560 paper rollouts into a fresh folder.

[Protocol](../../docs/f1tenth/README.md) · [Paper table](../../experiments/f1tenth/paper_nsr_20261027/report.md)

The configuration selection is recorded separately from the fresh evaluation. Superseded raw experiments are removed after verification; retained raw records include every successful and failed paper rollout.

[Video notebook](baselines/video_display.ipynb): a retained collision-free Atlas-DeePC lap with the official track, true-scale car close-up, speed reference, and tracking error. Run All displays the saved MP4; regeneration reads saved states without rerunning the controller.

[Noiseless longer-history trial](baselines/history_tini8_tf10_noiseless.ipynb): T_ini=8, T_f=10, K=12, d=40, seed 542; rebuilt atlas and chart selection/residuals over time, including the initial failed rollout and a tuned collision-free K=13, d=44 run with full chart-selection diagnostics. Paper results remain unchanged.

Current code uses applied input histories offline and online. The longer-history notebook preserves its two historical commanded-input examples and adds the corrected applied-input rerun. The old paper table is historical; RUN_ALL_EXPERIMENTS now evaluates its frozen settings with the corrected convention in a separate folder.
