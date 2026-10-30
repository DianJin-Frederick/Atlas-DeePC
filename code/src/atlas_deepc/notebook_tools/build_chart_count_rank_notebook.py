"""Build the user-run low-K batch notebook without executing it."""

from pathlib import Path
import nbformat
from atlas_deepc.ablations import low_k_completion as study

NOTEBOOK = study.ROOT / "notebooks/reacher/ablations/ablation_chart_count_and_rank.ipynb"


def build(root=study.RESULT_ROOT):
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb = nbformat.v4.new_notebook(cells=[
        md("""# Reacher: Complete the Missing Low-K Curves

K = 1, 2, 4, 8, 16, 24, 32; p=8 without guidance, p=8 with guidance,
and p=10 without guidance. Every integer rank 28--70 is tested on development
seeds 211--310 with T_sim=300: 21 searches, 903 rank panels, 90,300 rollouts.

The current baseline uses projection-residual chart selection, unstandardized
linear charts, Clarabel, and lambda_ini=100000. Guided cases use chordal costs
and iterative N=10 terminal selection; unguided cases use a fixed nearest eligible
terminal configuration. At K=1 there are no inter-chart waypoints.

Set BATCH_SIZE to the maximum number of **new rollouts** per invocation and
WORKERS to the number of concurrent rank panels. Set RUN_BATCH=True and run
the batch cell. Rerunning that cell resumes progress and runs one more bounded
batch. Interrupting preserves completed seeds. Avoid concurrent timing studies.

Only success counts/rates are retained. After all 43 ranks finish for a method/K,
select highest success (ties: lowest rank), and delete losing rank checkpoints.
Temporary trajectories are removed after each seed. Selection is exploratory,
not held-out evaluation. Historical Figure 6 curves remain separate, with their
original measurement horizons preserved; this notebook does not recompute them.
No simulation runs unless RUN_BATCH=True.
"""),
        code('''import os
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
from pathlib import Path
import sys
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from atlas_deepc.ablations import low_k_completion as study
'''),
        code(f'''RESULT_ROOT = ROOT / {str(Path(root).relative_to(study.ROOT))!r}
T_SIM = study.CONTROL_STEPS
assert T_SIM == 300
BATCH_SIZE = 100
WORKERS = 1
RUN_BATCH = False
print(f"T_sim={{T_SIM}}; seeds 211--310; results: {{RESULT_ROOT}}")
'''),
        md("## Run One Batch\nThe first batch builds the offline atlases. Later batches reuse them. BATCH_SIZE and WORKERS may change between invocations; the experiment protocol may not."),
        code('''if RUN_BATCH:
    batch = study.run_batch(RESULT_ROOT, batch_size=BATCH_SIZE, workers=WORKERS)
    print("New rollouts in this batch:", batch["batch_completed_rollouts"])
else:
    print("Report only. Set RUN_BATCH=True to execute one batch.")
'''),
        md("## Saved Progress"),
        code('''payload = study.report(RESULT_ROOT)
print("Completed rollouts:", payload["completed_rollouts"], "/ 90300")
print("Complete:", payload["complete"])
display(pd.DataFrame(payload["progress"]))
best = pd.DataFrame(payload["selected"])
if not best.empty:
    display(best[["clusters", "output_dimension", "reference_mode", "rank", "success_rate"]])
'''),
        md("## Completed Best-Rank Results\nOnly fully searched method/K combinations appear. Incomplete searches are not final winners."),
        code('''if not best.empty:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for (p, mode), group in best.groupby(["output_dimension", "reference_mode"]):
        group = group.sort_values("clusters")
        label = f"p={p}, " + ("with guidance" if mode == "graph" else "without guidance")
        ax.plot(group.clusters, group.success_rate, "o-", label=label)
    ax.set(xscale="log", xlabel="Number of charts K", ylabel="Success rate (%)", ylim=(0, 100))
    ax.set_xticks(study.COUNTS, labels=study.COUNTS)
    ax.grid(alpha=.2)
    ax.legend()
    plt.show()
else:
    print("No complete method/K searches yet.")
'''),
    ])
    nb.metadata.kernelspec = {"display_name": "gym-robotics-mujoco", "language": "python", "name": "gym-robotics-mujoco"}
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    build()
