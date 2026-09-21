"""Execute and validate the finished rank report; never launch simulations."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_100_v1"
NOTEBOOK = ROOT / "notebooks/ablations/ablation_linear_subspace_rank.ipynb"


def main(result=RESULT, notebook_path=NOTEBOOK, expected_figures=6):
    summary = json.loads((result / "summary.json").read_text())
    assert summary["complete"] and summary["ranks"] == list(range(20, 81))
    assert summary["seeds"] == list(range(211, 311))
    assert [r["rank"] for r in summary["completed"]] == list(range(20, 81))
    notebook = nbformat.read(notebook_path, as_version=4)
    setup = next(c.source for c in notebook.cells if c.cell_type == "code")
    assert "RUN_EXPERIMENTS = False" in setup
    NotebookClient(notebook, timeout=900, kernel_name="gym-robotics-mujoco",
                   resources={"metadata": {"path": str(notebook_path.parent)}}).execute()
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type == "error" for c in code_cells for o in c.outputs)
    figures = sum("image/png" in o.get("data", {}) for c in code_cells for o in c.outputs)
    assert figures == expected_figures
    report = json.loads((result / "notebook_report.json").read_text())
    assert report["complete"] and report["rollouts"] == 6100
    nbformat.validate(notebook)
    temporary = notebook_path.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(notebook_path)
    completion = {"executed_utc": datetime.now(timezone.utc).isoformat(),
                  "notebook": str(notebook_path), "executed_code_cells": len(code_cells),
                  "saved_inline_figures": figures, "rollouts": 6100,
                  "visual_inspection": "pending"}
    (result / "notebook_execution.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
