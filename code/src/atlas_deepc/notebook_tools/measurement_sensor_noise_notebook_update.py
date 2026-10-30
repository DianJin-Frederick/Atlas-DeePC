"""Physical-sensor noise report adapter; never executes notebook cells."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


def update_cells(cells):
    for c in cells:
        s=c.source.replace('measurement_norm_noise','measurement_sensor_noise').replace('measurement_norm_gaussian_v1','physical_sensor_gaussian_v1')
        s=s.replace('noise_sd_ratio','noise_multiplier').replace('SD_RATIOS','NOISE_MULTIPLIERS')
        s=s.replace('NOISE_MULTIPLIERS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.10]', '''NOISE_MULTIPLIERS = [0., 1., 2., 3., 4., 5., 10.]
# Illustrative fixed physical standard deviations at 1x; edit before running.
ANGLE_SD_DEG = 0.1
POSITION_SD_MM = 1.0
VELOCITY_SD_RAD_S = 0.01
experiment.noise.configure(ANGLE_SD_DEG, POSITION_SD_MM, VELOCITY_SD_RAD_S)''')
        s=s.replace('"--ratios", *map(str, NOISE_MULTIPLIERS)', '''"--multipliers", *map(str, NOISE_MULTIPLIERS),
               "--angle-sd-deg", str(ANGLE_SD_DEG), "--position-sd-mm", str(POSITION_SD_MM),
               "--velocity-sd-rad-s", str(VELOCITY_SD_RAD_S)''')
        s=s.replace('100*np.array(NOISE_MULTIPLIERS)','np.array(NOISE_MULTIPLIERS)')
        s=s.replace('100*ratio','ratio').replace('100 * ratio','ratio').replace('{100*r:g}', '{r:g}').replace('100*r,', 'r,')
        s=s.replace('100*row["noise_multiplier"]','row["noise_multiplier"]').replace('100*example_table.noise_multiplier','example_table.noise_multiplier')
        s=s.replace('Relative amplitude (%)','Noise multiplier').replace('relative noise amplitude ρ (%)','noise multiplier').replace('relative noise amplitude ρ','noise multiplier')
        s=s.replace('"Power NSR": ratio**2,','')
        s=s.replace('("sd_ratio", "power_nsr")','("multiplier",)')
        s=s.replace('ATLAS_NOISE_RATIO = 0.05','ATLAS_NOISE_RATIO = 5.0').replace('100*ATLAS_NOISE_RATIO','ATLAS_NOISE_RATIO')
        s=s.replace('% SD','x').replace('% SD-noise','x sensor-noise').replace('% noise','x noise')
        s=s.replace('f"{ratio:g}%"','f"{ratio:g}x"')
        s=s.replace('((212, .01, .05), (222, .01, .05), (237, .02, .10))','((212, 1., 5.), (222, 1., 5.), (237, 2., 10.))')
        s=s.replace('physical_sensor_gaussian_v1','physical_sensor_gaussian_v1')
        if c.cell_type=='code' and 'primitive' not in s and s.startswith('calibration = []'):
            s='''calibration = []
for multiplier, payload in payloads.items():
    artifact = payload["artifact"]
    sigma = np.asarray(artifact["primitive_noise_sigma"])
    calibration.append({"Noise multiplier": multiplier, "Angle SD (deg)": np.rad2deg(sigma[0]),
        "Position SD per axis (mm)": 1000*sigma[4], "Velocity SD (rad/s)": sigma[2]})
display(pd.DataFrame(calibration))
print("Fixed SDs in physical units; these levels are not NSR percentages.")
'''
        if c.cell_type=='code' and s.startswith('from candidate_goal_routing_experiment import file_digest\natlas_rows'):
            a=s.index('    relative_power = []');s=s[:a]+s[a:].replace('    relative_power = []\n','')
            s=s.replace('normal(size=truth.shape)','normal(size=truth.shape[:-1]+(6,))')
            s=s.replace('sigma = experiment.noise.measurement_sigma(truth, ratio)','sigma = experiment.noise.measurement_sigma(ratio)')
            s=s.replace('np.testing.assert_allclose(trace["sensor_noise_sigma"], sigma[:, 0])','np.testing.assert_allclose(trace["sensor_noise_sigma"], np.broadcast_to(sigma, (323,6)))')
            s=s.replace('truth + sigma*z','experiment.noise.corrupt_outputs(truth, ratio, z)')
            a=s.index('            energy =');b=s.index('    sizes =',a);s=s[:a]+s[b:]
            s=s.replace('"Expected power NSR": ratio**2, "Mean realized online power NSR": np.mean(relative_power),','"Angle SD (deg)": np.rad2deg(sigma[0]), "Position SD (mm)": 1000*sigma[4], "Velocity SD (rad/s)": sigma[2],')
            s=s.replace('measurement-norm noise streams','physical-sensor noise streams')
        if c.cell_type=='code' and s.startswith('import measurement_sensor_noise_rank62'):
            s=s.replace('sigma = experiment.noise.measurement_sigma(current["sensor_true"], ratio)','sigma = experiment.noise.measurement_sigma(ratio)')
            s=s.replace('np.testing.assert_allclose(current["sensor_noise_sigma"], sigma[:, 0])','np.testing.assert_allclose(current["sensor_noise_sigma"], np.broadcast_to(sigma, (323,6)))')
            s=s.replace('current["sensor_true"] + sigma*current["sensor_standard_normal"]','experiment.noise.corrupt_outputs(current["sensor_true"], ratio, current["sensor_standard_normal"])')
        if c.cell_type=='code' and 'clean_signature = payloads' in s:
            s=s.replace('clean_signature = payloads[0.0]["artifact"]["signature"]', 'clean_signature = payloads[0.0]["artifact"]["signature"]\nassert clean_signature["sensor_settings"] == experiment.noise.sensor_settings(), "Notebook scales differ from saved results"')
        c.source=s
    cells[0].source='''# Atlas-DeePC: Fixed-Scale Physical Sensor Noise

**Ready for local Run All; experiments have not been run for this model.**
Ranks **59 and 62**, with **100 paired seeds 211–310 per level**, λg=0.05,
Clarabel, 48 linear charts, graph waypoints, iterative N=10 references and
320 steps. Both offline data and online sensing use the new model.

At **1×**, the editable defaults are **0.1° angle SD, 1 mm position SD per axis,
and 0.01 rad/s velocity SD**. These are illustrative values, not calibrated sensor
specifications. Levels **0×, 1×, 2×, 3×, 4×, 5×, 10×** give 1,400 rollouts.
Each scale is constant throughout its experiment and independent of signal energy.
These multipliers replace the former NSR percentages; no old results are reused.

Choose the `gym-robotics-mujoco` kernel, reopen/restart after code changes, and
click **Run All**. `RUN_EXPERIMENTS=True` is already set. The first cell launches
both ranks, retains logs, and resumes compatible checkpoints. Default `WORKERS=1`.
The three base SDs and multiplier list are visible in the first cell.
Keep `RENDER_VIDEOS=False` unless optional saved-trace videos are wanted.
'''
    for c in cells:
        if c.cell_type=='markdown' and '### Noise Definition and Placement' in c.source:
            c.source=r'''### Noise Definition and Placement

For multiplier $a$, draw six independent standard-normal values per sample:
$z_{q_1},z_{q_2},z_{v_1},z_{v_2},z_x,z_y$. Use fixed physical scales:

$$\tilde q_i=q_i+a\sigma_q z_{q_i},\quad
\tilde{\dot q}_i=\dot q_i+a\sigma_v z_{v_i},\quad
\tilde p_j=p_j+a\sigma_p z_{p_j}.$$

The defaults at $a=1$ are $\sigma_q=0.1^\circ$, $\sigma_v=0.01$ rad/s,
and $\sigma_p=0.001$ m per position axis. Gaussian errors are unbounded; a quoted
SD is not a hard error bound. There is no norm scaling, clipping or signal-dependent
floor. Zero readings still receive the same fixed-scale noise.

Rebuild all six periodic features from the **same two noisy angles**:
$\cos\tilde q_1,\cos\tilde q_2,\sin\tilde q_1,\sin\tilde q_2,
\cos(\tilde q_1+\tilde q_2),\sin(\tilde q_1+\tilde q_2)$.
No separate noise is added to these derived channels. They remain on their unit
circles and retain angle-sum consistency. Only the primitive sensor errors are
independent Gaussian draws; lifted-feature errors are correlated and non-Gaussian.

This assumes independently sensed fingertip position and velocity, not position
computed by forward kinematics or velocity obtained by differencing noisy encoders.
Thus noisy position need not exactly agree with forward kinematics of noisy angles.

**Offline:** recover angles with atan2 from the clean cosine/sine pairs, corrupt
unique physical samples once, and rebuild every overlapping Hankel window from
the same corrupted samples. Applied inputs and original source files remain exact.
At each nonzero level refit the noisy atlas, medoids, graph and bases; rank 62
reuses the rank-59 atlas and verifies nested bases. Save six-channel primitive
noise RMS and source/cache hashes. Worker reconstruction is deterministic.

**Online:** apply the identical model at every observation, including reset and
initial history. Only noisy outputs reach localization, goal selection and control.
Evaluate against true fingertip positions. Store truth, measured outputs, primitive
draws and fixed six-channel SD vectors for audit. Noise draws are independent
across samples; use common random numbers across levels and ranks. Targets stay exact.
'''
        elif c.cell_type=='markdown':
            c.source=c.source.replace('**5% noise atlas**','**5× noise atlas**').replace('same diagnostic examples as before','same fixed diagnostic seeds as before')
            c.source=c.source.replace('**noisy observed configurations**, so link lengths and trigonometric consistency\nneed not be exact.', '**noisy observed configurations**. Periodic features remain consistent; independently\nnoisy position need not exactly match the forward-kinematic fingertip.')
            c.source=c.source.replace('of independent output-channel noise','of independently sensed position and angle noise')
            c.source=c.source.replace('paired across ranks; their conditional noise scales follow each trajectory.', 'paired across ranks, with the same fixed physical noise scales.')
            c.source=c.source.replace('1% versus 5%', '1× versus 5×').replace('2% versus 10%', '2× versus 10×')
            if '## 8. Local Reproduction' in c.source:
                c.source='''## 8. Local Reproduction and Preservation

Click **Run All** with `RUN_EXPERIMENTS=True` (already enabled). You can edit the
three SDs in the first cell before your first run. Changing scales after results
exist requires a fresh output directory; incompatible caches are rejected.

Equivalent command in the project environment:

```bash
python src/atlas_deepc/noise/run_measurement_sensor_noise.py --workers 1 --angle-sd-deg 0.1 --position-sd-mm 1 --velocity-sd-rad-s 0.01
```

Both ranks run on the same 100 seeds and seven multipliers. Fresh results use
`measurement_sensor_noise_rank59_lamg005_100_v1` and
`measurement_sensor_noise_rank62_lamg005_100_v1` under `experiments/reacher/routed_deepc_iid`.
The complete project, clean offline Hankel file and baseline atlas are needed.
The existing `gym-robotics-mujoco` environment is suitable.

Historical fixed-output-scale and measurement-norm studies are preserved.
The previous notebook is archived as
`archive/notebooks/ablation_measurement_noise_norm_before_sensor_model.ipynb`.
No simulation or atlas fitting was run while preparing this notebook. New outputs
are cleared. Videos are optional and use only new physical-sensor traces.
'''
        if c.cell_type=='code':
            compile(c.source,'sensor_noise_notebook','exec');c.outputs=[];c.execution_count=None
    return cells
