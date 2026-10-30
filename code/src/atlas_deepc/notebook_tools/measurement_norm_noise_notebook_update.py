"""Adapt the established noise report to per-measurement Gaussian scaling."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import nbformat as nbf


def update_cells(cells):
    for c in cells:
        c.source = (c.source.replace('measurement_noise_rank59_ablation', 'measurement_norm_noise_rank59')
            .replace('measurement_noise_rank62_ablation', 'measurement_norm_noise_rank62')
            .replace('render_measurement_noise_videos', 'render_measurement_norm_noise_videos')
            .replace('noise SD / offline signal SD (%)', 'relative noise amplitude ρ (%)')
            .replace('noise SD / signal SD (%)', 'relative noise amplitude ρ (%)')
            .replace('noise SD / signal SD', 'relative noise amplitude ρ')
            .replace('Noise SD (%)', 'Relative amplitude (%)')
            .replace('four rollout workers', 'the configured rollout worker count (one by default)')
            .replace('four workers', 'the configured workers')
            .replace('WORKERS = 4', 'WORKERS = 1')
            .replace('import os\nos.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")', 'import os, tempfile\nos.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "atlas_deepc_mpl"))'))
    cells[0].source = r'''# Atlas-DeePC: Measurement-Norm Gaussian Noise

**Ready for a new local run; no results for this model are included yet.**
Seven relative-amplitude levels **0%, 1%, 2%, 3%, 4%, 5%, 10%**, with **100 paired
seeds 211–310 for each level and each rank (59 and 62): 1,400 full rollouts**.
Keep λg=0.05, Clarabel, 48 charts, linear bases, graph waypoints, iterative N=10
terminal references, and all 320 steps. No controller retuning is introduced.

The Gaussian scale now follows each clean measurement's norm, both offline and
online. The original fixed offline-SD noise results are archived, not reused or
relabeled. All outputs are cleared pending your local run.

**To run locally:** choose your `gym-robotics-mujoco` Python kernel (or an environment
with the project's dependencies), then click Run All.
`RUN_EXPERIMENTS=True` is already set. The subprocess uses that kernel's Python,
runs rank 59 then rank 62, logs progress, and resumes compatible completed seeds.
Default `WORKERS=1` limits load and makes timing easier to interpret. Keep the
whole project, including `src/`, offline data and the original atlas, together.
After completion, set `RUN_EXPERIMENTS=False` to regenerate reports only.

Videos are optional: keep `RENDER_VIDEOS=False` for the initial run. No simulation,
atlas fitting, or notebook execution was performed while preparing this update.
'''
    setup = next(c for c in cells if c.cell_type=='code' and 'RUN_EXPERIMENTS = False' in c.source)
    setup.source = setup.source.replace("RUN_EXPERIMENTS = False", "RUN_EXPERIMENTS = True")
    setup.source = setup.source.replace('if RUN_EXPERIMENTS:\n    experiment.run_all(SD_RATIOS, SEEDS, WORKERS)', '''if RUN_EXPERIMENTS:
    import subprocess, signal
    from collections import deque
    from datetime import datetime
    log_dir = ROOT / "experiments/reacher/routed_deepc_iid/measurement_norm_noise_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (datetime.now().strftime("run_%Y%m%d_%H%M%S_%f") + ".log")
    command = [sys.executable, "-u", str(ROOT / "src/atlas_deepc/noise/run_measurement_norm_noise.py"),
               "--workers", str(WORKERS), "--ratios", *map(str, SD_RATIOS)]
    print("Run log:", log_path, flush=True)
    tail = deque(maxlen=40)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=os.name != "nt")
        try:
            for line in process.stdout:
                log.write(line); log.flush(); tail.append(line)
                print(line, end="", flush=True)
            if process.wait() != 0:
                raise RuntimeError("Noise sweep stopped. Completed checkpoints are retained. "
                                   + str(log_path) + "\\n" + "".join(tail))
        except KeyboardInterrupt:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGINT)
            else:
                process.terminate()
            process.wait()
            raise
''')
    setup.source = setup.source.replace('enable RUN_EXPERIMENTS to run/resume.', 'set RUN_EXPERIMENTS=True to run/resume both ranks locally.')
    for c in cells:
        if c.cell_type=='markdown' and '### Noise Definition and Placement' in c.source:
            c.source=r'''### Noise Definition and Placement

For the full clean output $y_t\in\mathbb R^{10}$, independently draw
$z_t\sim\mathcal N(0,I_{10})$ and use

$$\widetilde y_t=y_t+\epsilon_t,\qquad
\epsilon_t=\frac{\rho\|y_t\|_2}{\sqrt{10}}z_t.$$

Thus $\mathbb E[\|\epsilon_t\|_2^2\mid y_t]=\rho^2\|y_t\|_2^2$.
The existing amplitude levels $\rho$ stay unchanged; **power NSR is $\rho^2$**
and SNR is $-20\log_{10}\rho$ dB for nonzero $y_t$. The factor $1/\sqrt{10}$
avoids multiplying total noise power by the output dimension. Gaussian draws
are not normalized or clipped; realized NSR fluctuates around its expectation.
Exactly zero measurements receive exactly zero noise, with no fixed noise floor.

This is the raw **ten-channel norm**, containing six sine/cosine features,
two velocities and fingertip x/y, in the baseline's original coordinates.
It mixes units and does not give each coordinate the same relative error.
The clean periodic features contribute exactly 3 to the squared norm, so
$\|y_t\|_2\ge\sqrt{3}$ even when the fingertip or velocity is near zero.
Consequently this model scales with full-output energy, not target error or
fingertip magnitude alone. Scaling uses clean simulator output before injection;
the controller receives only the corrupted measurement, never its scale or truth.

**Offline:** corrupt each unique clean sample once, then rebuild the overlapping
Hankel windows with identical copies of that corrupted sample. Inputs remain exact.
At each nonzero level, refit the noisy atlas, medoids, graph and bases as before;
zero noise uses the original atlas. The noisy atlas is shared by ranks 59 and 62.

**Online:** use the same formula on each current true output, including reset and
initial history. Localization, past matching, and the latch use noisy measurements.
Targets and actions stay exact; evaluation uses true fingertip positions.

Offline and per-seed online standard-normal streams are reproducible and paired
across levels/ranks. **The z draws match, but noise magnitudes can differ across
ranks because their closed-loop measurements differ.** No sensor filtering,
projection, clipping, process noise or actuator noise is added.
'''
        if c.cell_type=='code' and c.source.startswith('calibration = []'):
            c.source='''calibration = []
for ratio, payload in payloads.items():
    artifact = payload["artifact"]
    calibration.append({"Relative amplitude (%)": 100*ratio,
        "Expected power NSR": ratio**2, "Expected power NSR (%)": 100*ratio**2,
        "SNR (dB)": np.inf if ratio == 0 else -20*np.log10(ratio),
        "Offline conditional coordinate SD min": artifact["offline_sigma_min"],
        "Offline conditional coordinate SD median": artifact["offline_sigma_median"],
        "Offline conditional coordinate SD max": artifact["offline_sigma_max"],
        "Mean realized sample power NSR": artifact["empirical_offline_power_nsr"]})
display(pd.DataFrame(calibration).round(6))
print("Coordinate SD varies with each sample's norm; there is no fixed per-channel calibration.")
'''
        if c.cell_type=='code' and c.source.startswith('zero_matches = 0'):
            c.source='''from candidate_goal_routing_experiment import file_digest
atlas_rows = []
for ratio, payload in payloads.items():
    artifact = payload["artifact"]
    assert artifact["signature"]["noise_model"] == "measurement_norm_gaussian_v1"
    assert artifact["overlap_consistency_verified"] and artifact["applied_inputs_unchanged"]
    assert artifact["offline_unique_samples"] == 40000
    for name, digest in artifact["artifact_sha256"].items():
        assert file_digest(RESULT_ROOT / experiment.level_name(ratio) / name) == digest
    with np.load(RESULT_ROOT / experiment.level_name(ratio) / "chart_cache.npz") as cache:
        assert cache["basis"].shape == (48, 144, 59) and not np.any(cache["offset"])
    relative_power = []
    for row in payload["results"]:
        path = RESULT_ROOT / experiment.level_name(ratio) / f"seed{row['seed']}_control_results.npz"
        with np.load(path, allow_pickle=True) as trace:
            truth, measured = trace["sensor_true"], trace["sensor_measured"]
            assert truth.shape == (323, 10)
            np.testing.assert_array_equal(measured[-320:], trace["y_measured"])
            np.testing.assert_array_equal(truth[-320:], trace["y"])
            np.testing.assert_allclose(trace["distance"], np.linalg.norm(trace["y"][:, 6:8]-row["exact_target_xy"], axis=1))
            z = np.random.default_rng(np.random.SeedSequence([experiment.noise.ONLINE_NOISE_SEED, row["seed"]])).normal(size=truth.shape)
            np.testing.assert_array_equal(z, trace["sensor_standard_normal"])
            sigma = experiment.noise.measurement_sigma(truth, ratio)
            np.testing.assert_allclose(trace["sensor_noise_sigma"], sigma[:, 0])
            np.testing.assert_allclose(measured, truth + sigma*z, rtol=0, atol=1e-14)
            if ratio == 0:
                np.testing.assert_array_equal(truth, measured)
            energy = np.sum(truth**2, axis=-1)
            relative_power.extend((np.sum((measured-truth)**2, axis=-1)[energy>0]/energy[energy>0]).tolist())
    sizes = artifact["chart_sizes"]
    atlas_rows.append({"Relative amplitude (%)": 100*ratio,
        "Expected power NSR": ratio**2, "Mean realized online power NSR": np.mean(relative_power),
        "Charts": len(sizes), "Smallest chart": min(sizes), "Largest chart": max(sizes),
        "Unreachable chart pairs": artifact["unreachable_chart_pairs"],
        "Offline fit (s)": artifact["offline_preparation_s"]["chart_fit"],
        "Offline bases/routes (s)": artifact["offline_preparation_s"]["svd_cache_and_routes"]})
print(f"Verified rank-59 charts and all {len(frame)} measurement-norm noise streams.")
display(pd.DataFrame(atlas_rows).set_index("Relative amplitude (%)").round(6))
'''
        if c.cell_type=='code' and c.source.startswith('import measurement_norm_noise_rank62'):
            c.source=c.source.replace('RUN_RANK62_EXPERIMENTS = False\n','').replace('if RUN_RANK62_EXPERIMENTS:\n    rank62_experiment.run_all(SD_RATIOS, SEEDS, workers=WORKERS)\n','')
            c.source=c.source.replace('enable RUN_RANK62_EXPERIMENTS.', 'set RUN_EXPERIMENTS=True in the first cell.')
            c.source=c.source.replace('''                np.testing.assert_allclose(current["sensor_measured"]-current["sensor_true"],
                    previous["sensor_measured"]-previous["sensor_true"], rtol=0, atol=1e-13)''','''                np.testing.assert_array_equal(current["sensor_standard_normal"], previous["sensor_standard_normal"])
                sigma = experiment.noise.measurement_sigma(current["sensor_true"], ratio)
                np.testing.assert_allclose(current["sensor_noise_sigma"], sigma[:, 0])
                np.testing.assert_allclose(current["sensor_measured"],
                    current["sensor_true"] + sigma*current["sensor_standard_normal"], rtol=0, atol=1e-14)''')
        if c.cell_type=='code' and 'RENDER_VIDEOS = False' in c.source:
            # Skip absent optional videos without stopping the rank-62 report.
            start=c.source.index('entries = json.loads')
            before,after=c.source[:start],c.source[start:]
            c.source=before+'if not (VIDEO_DIR / "manifest.json").exists():\n    print("No new-model videos yet. Set RENDER_VIDEOS=True to render saved traces (optional).")\nelse:\n'+''.join('    '+line+'\n' for line in after.splitlines())
        if c.cell_type=='markdown':
            c.source=c.source.replace('same seed and noise realization', 'same seed and standard-normal draws')
            c.source=c.source.replace('Online Gaussian draws and targets are\npaired across ranks.', 'Online standard-normal draws and targets are\npaired across ranks; their conditional noise scales follow each trajectory.')
        if c.cell_type=='markdown' and '## 8. Reproduction' in c.source:
            c.source='''## 8. Local Reproduction and Preservation

Click Run All (`RUN_EXPERIMENTS=True` is already set), or use the notebook kernel's
Python from the project root:

```bash
python src/atlas_deepc/noise/run_measurement_norm_noise.py --workers 1
```

This launches both ranks on all seven levels and 100 seeds (1,400 rollouts).
Use the existing gym-robotics-mujoco environment, or install the compatible
packages listed in `requirements-control-parameter.txt`. No absolute machine path
is needed by the notebook or launcher. Resume rejects incompatible data, code or
settings. Restart the kernel after source changes. Complete checkpoints are
retained on interruption; a seed interrupted during writing is rerun.

Fresh results go to `measurement_norm_noise_rank59_lamg005_100_v1` and
`measurement_norm_noise_rank62_lamg005_100_v1` under `experiments/reacher/routed_deepc_iid`.
The previous fixed-scale experiment data and scripts remain unchanged. Its notebook
is preserved at `archive/notebooks/ablation_measurement_noise_fixed_scale_before_20260923.ipynb`.
The new notebook has no saved experiment outputs and does not display old-model videos.

This remains a combined offline/online noise study: it does not isolate those
contributions. Nonzero noise changes offline charts and reference consistency,
as well as online sensing. The raw lifted-output norm does not suppress noise in
one near-zero coordinate if the other coordinates are large.
'''
    for c in cells:
        if c.cell_type=='code':
            compile(c.source, 'measurement_norm_noise_notebook', 'exec')
            c.outputs=[]; c.execution_count=None
    return cells
