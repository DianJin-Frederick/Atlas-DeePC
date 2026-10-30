"""Build the standalone, opt-in matched graph-guidance evaluation notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = ROOT / "notebooks/reacher/ablations/effect_of_graph_guidance.ipynb"


def matched_guidance_500_cells():
    cells = [nbf.v4.new_markdown_cell(r"""# Effect of Graph Guidance: 500 Paired Seeds

Fix **p=8, K=64, d=53, T_sim=300** and evaluate guidance on **10001--10500**.
Reuse the existing 500 unguided baseline rollouts and their exact frozen atlas;
only the guided side runs. No reclustering, rank selection, or weight tuning.

This compares the full guidance policy (chordal waypoints and iterative N=10
terminal-reference selection) against a fixed nearest eligible terminal reference.
It isolates that policy package at fixed p/K/d, not waypoints alone. These are
repeated evaluation seeds, not a new untouched test set. Figure-selected seeds
are not used as a separate statistical sample, and their existing files are unchanged.

Run the setup cell below, then set **RUN_MATCHED_GUIDANCE_500=True**
in the run cell. No baseline experiments or figure-seed panels are launched. It runs serially with one numerical thread and resumes
complete seed files after interruption. Pause the low-K sweep and other heavy
experiments if the new timings will be reported; old/new sessions are separate.

Full guided traces and metrics are stored separately in
`experiments/reacher/matched_guidance_500_p8_k64_r53_t300_v1`.
The paired table reports both-success, repaired, regressed, and both-failed counts,
success-rate difference, and an exact two-sided McNemar test after all 500 finish.
Initial input/output windows and target positions are checked seed by seed.
"""), nbf.v4.new_code_cell('''from atlas_deepc.ablations import matched_guidance_500 as guidance500
importlib.reload(guidance500)
RUN_MATCHED_GUIDANCE_500 = False
print("Paired seeds:", guidance500.SEEDS[0], "--", guidance500.SEEDS[-1])
print("Guided results:", guidance500.RESULT_ROOT)
if RUN_MATCHED_GUIDANCE_500:
    import signal, subprocess
    child = subprocess.Popen(
        [sys.executable, "-u", "-m", "atlas_deepc.ablations.matched_guidance_500"],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True)
    try:
        for line in child.stdout:
            print(line, end="", flush=True)
        if child.wait():
            raise RuntimeError("Guidance study stopped; completed seeds are retained. See output above.")
    except KeyboardInterrupt:
        try:
            os.killpg(child.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
        print("Stopped. Rerun this cell to resume the 500-seed guided panel.")
    finally:
        child.stdout.close()
else:
    print("No rollouts launched. Enable RUN_MATCHED_GUIDANCE_500 when ready.")
'''), nbf.v4.new_code_cell('''paired500 = guidance500.report()
print(f"Matched pairs complete: {paired500['completed_pairs']}/500")
print("FINAL" if paired500["complete"] else "PARTIAL: completed pairs only; not a final success-rate estimate")
display(pd.Series(paired500["paired_counts"], name="Seed count").to_frame())
if paired500["completed_pairs"]:
    display(pd.Series({key: paired500[key] for key in (
        "unguided_success_rate", "guided_success_rate", "improvement_percentage_points",
        "exact_mcnemar_pvalue")}, name="Paired comparison").to_frame())
print("Repaired seeds:", paired500["paired_seeds"]["repaired"])
print("Regressed seeds:", paired500["paired_seeds"]["regressed"])
display(pd.DataFrame(paired500["rows"]).reindex(columns=[
    "seed", "unguided_success", "guided_success", "unguided_final100_mm", "guided_final100_mm",
    "unguided_rollout_rmse_mm", "guided_rollout_rmse_mm"]))
print("Guided error/timing metrics:", guidance500.RESULT_ROOT / "output8/graph/metrics.json")
''')]
    for cell in cells:
        cell.metadata["study"] = "matched_guidance_500"
    return cells



def build(output=NOTEBOOK):
    cells = matched_guidance_500_cells()
    setup = nbf.v4.new_code_cell('''from pathlib import Path
import os, sys
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "src/atlas_deepc").is_dir())
sys.path.insert(0, str(ROOT / "src"))
import importlib
import pandas as pd
from IPython.display import display
''')
    cells.insert(1, setup)
    notebook = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {
        "display_name": "gym-robotics-mujoco", "language": "python", "name": "gym-robotics-mujoco"}})
    nbf.validate(notebook)
    for cell in cells:
        if cell.cell_type == "code":
            compile(cell.source, str(output), "exec")
    nbf.write(notebook, output)
    return Path(output)


if __name__ == "__main__":
    print(build())
