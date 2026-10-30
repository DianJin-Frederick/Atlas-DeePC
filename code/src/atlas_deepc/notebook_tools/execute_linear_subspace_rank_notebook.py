"""Execute and validate the finished rank report; never launch simulations."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root

import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")

import nbformat
from nbclient import NotebookClient

ROOT = project_root()
NOTEBOOK = ROOT / "notebooks/reacher/ablations/ablation_linear_subspace_rank.ipynb"


def main(notebook_path=NOTEBOOK, expected_figures=5):
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
    nbformat.validate(notebook)
    temporary = notebook_path.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(notebook_path)
    completion = {"executed_utc": datetime.now(timezone.utc).isoformat(),
                  "notebook": str(notebook_path), "executed_code_cells": len(code_cells),
                  "saved_inline_figures": figures, "sweep_evaluations": 68800,
                  "mode": "saved-results only; no simulations"}
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
