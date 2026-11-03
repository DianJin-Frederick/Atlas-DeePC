"""Check saved-result notebooks without rerunning trials or overwriting outputs."""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def check_one(path):
    os.chdir(ROOT)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["MPLCONFIGDIR"] = "/tmp/atlas_cleanup_matplotlib"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    sys.dont_write_bytecode = True
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from IPython.core.interactiveshell import InteractiveShell
    from IPython.utils.capture import capture_output

    notebook = json.loads(path.read_text())
    shell = InteractiveShell.instance()
    shell.enable_gui = lambda *a, **kw: None
    shell.user_ns["__name__"] = "__main__"
    checked, skipped = [], []

    original_run = subprocess.run
    original_csv = pd.DataFrame.to_csv
    original_json = pd.DataFrame.to_json

    def report_csv(frame, path_or_buf=None, *args, **kwargs):
        if path_or_buf is None:
            return original_csv(frame, None, *args, **kwargs)

    def report_json(frame, path_or_buf=None, *args, **kwargs):
        if path_or_buf is None:
            return original_json(frame, None, *args, **kwargs)

    def no_subprocess(command, *args, **kwargs):
        if isinstance(command, str) and command in {"lscpu", "uname"}:
            return original_run(command, *args, **kwargs)
        if isinstance(command, (list, tuple)) and Path(str(command[0])).name in {"lscpu", "sysctl", "uname", "fc-list"}:
            return original_run(command, *args, **kwargs)
        if any("cylinder_paper_figure.py" in str(arg) for arg in command):
            return subprocess.CompletedProcess(command, 0, "Existing paper figure retained.", "")
        raise RuntimeError(f"Report-only validation blocked a subprocess: {command}")

    with ExitStack() as stack:
        for owner, name in [(Path, "write_text"), (Path, "write_bytes"),
                            (Path, "replace"),
                            (np, "save"), (np, "savez"), (np, "savez_compressed"),
                            (Figure, "savefig"), (plt, "savefig")]:
            stack.enter_context(patch.object(owner, name, lambda *a, **kw: None))
        stack.enter_context(patch.object(subprocess, "run", no_subprocess))
        stack.enter_context(patch.object(pd.DataFrame, "to_csv", report_csv))
        stack.enter_context(patch.object(pd.DataFrame, "to_json", report_json))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            code = "".join(cell.get("source", []))
            code = re.sub(r"(?m)^(RUN_[A-Z_0-9]+|RENDER_[A-Z_0-9]+)\s*=\s*True\b", r"\1 = False", code)
            with capture_output() as captured:
                result = shell.run_cell(code, store_history=False)
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                return {"notebook": str(path.relative_to(ROOT)), "passed": False,
                        "failed_cell": index, "error": f"{type(error).__name__}: {error}",
                        "output_tail": (captured.stdout + captured.stderr)[-5000:],
                        "checked_cells": checked, "skipped_cells": skipped}
            checked.append(index)
            plt.close("all")
    return {"notebook": str(path.relative_to(ROOT)), "passed": True,
            "checked_cells": checked, "skipped_cells": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--one", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.one:
        start = time.monotonic()
        try:
            result = check_one(args.one.resolve())
        except Exception as exc:
            result = {"notebook": str(args.one), "passed": False,
                      "error": f"{type(exc).__name__}: {exc}"}
        result["elapsed_seconds"] = time.monotonic() - start
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "output_tail"}))
        return
    notebooks = sorted((ROOT / "notebooks").rglob("*.ipynb")) + sorted(ROOT.glob("*.ipynb"))
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, path in enumerate(notebooks):
        report = args.output / f"{index:02d}_{path.stem}.json"
        try:
            run = subprocess.run([sys.executable, __file__, "--one", str(path), "--output", str(report)],
                                 capture_output=True, text=True, timeout=240, cwd=ROOT)
            result = json.loads(report.read_text()) if report.exists() else {
                "notebook": str(path.relative_to(ROOT)), "passed": False,
                "error": run.stderr[-3000:]}
        except subprocess.TimeoutExpired:
            result = {"notebook": str(path.relative_to(ROOT)), "passed": False, "error": "240s timeout"}
        results.append(result)
        print(f"{'PASS' if result['passed'] else 'FAIL'} {path.relative_to(ROOT)} "
              f"{result.get('error', '')}", flush=True)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
