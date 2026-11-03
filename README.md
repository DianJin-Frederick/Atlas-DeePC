# Atlas-DeePC: Reacher and F1TENTH

Research code for Atlas-DeePC. Run commands from this repository's root
(the local `atlas_deepc/code/` directory). The website is maintained separately
and is not included in this code-only repository.

## Start Here

- [Current notebook index](notebooks/README.md)
- [Main Reacher baseline](notebooks/reacher/baselines/baseline_graph_routed_atlas_deepc.ipynb)
- [Reacher comparisons and ablations](notebooks/reacher/README.md)
- [F1TENTH paper notebook](notebooks/f1tenth/baselines/baseline_f1tenth_atlas_deepc.ipynb)
- [F1TENTH reproduction protocol](docs/f1tenth/README.md)
- [Implementation map](src/README.md)

## Project Layout

| Folder | Purpose |
|---|---|
| `notebooks/` | Current baselines, comparisons, ablations, and illustrations |
| `src/atlas_deepc/` | Atlas-DeePC controllers, task utilities, and experiment runners |
| `src/comparison_methods/` | Contextual-DPC, faithful KDE-BDPC, and external Select-DPC adapters |
| `data/` | Original offline datasets and F1TENTH map assets |
| `configs/` | Explicit settings and dependency locks |
| `third_party/` | Licensed simulator dependencies |
| `experiments/` | Retained results, protocols, source snapshots, and figures |
| `docs/` | Current reproduction documentation |
| `scripts/` | Setup, maintenance, and report utilities |

## Running Notebooks

Use the scientific Python environment specified by each notebook. Notebooks
locate `src/` from the project root. Review run/render switches before using
Run All; enable them deliberately to launch experiments.

The Reacher baseline uses projection-residual chart selection, chordal graph
costs for guided cases, Clarabel, and a tracking/input/history-slack objective.
Its mixed-K evaluation settings are documented in the [Reacher index](notebooks/reacher/README.md).
The [F1TENTH protocol](docs/f1tenth/README.md) documents noise, tuning, and evaluation.

Source/data/protocol hashes protect saved runs. Changed implementations must
use a fresh result directory, not overwrite the provenance of earlier runs.

## Experiment Storage

Many saved arrays, videos, JSON summaries, and protocol files in `experiments/`
are local links to `/Volumes/Crucial X10/atlas_deepc_data/`, through
`.external_experiment_data`. Connect that drive before reading linked results.
Original offline inputs and current notebooks/code remain on the computer.
New outputs are local unless their runner explicitly uses the SSD.

The local historical archive and obsolete cleanup/migration records have been
removed. This does not delete SSD targets or change retained experiment results.
Historical source snapshots inside retained experiment directories remain
part of those experiments' provenance. Keep a separate backup of important data.

## Optional Select-DPC Comparison

Select-DPC is an optional external dependency, not part of the Atlas-DeePC
controller. The setup script downloads the authors' source at commit
`8afa3b36ca47570400b0b8ff82328ab25ee3fdee` from
[choose-wisely-paper](https://github.com/naefjo/choose-wisely-paper), retaining
its license, and creates a separate Python environment outside this project.
No experiment is launched by setup.

From the repository root, using Python 3.10 and Git:

```sh
python3.10 scripts/setup_select_dpc.py
python3.10 scripts/setup_select_dpc.py --check
python3.10 scripts/run_select_dpc.py --help
```

The default location is `~/.local/share/atlas-deepc/select-dpc/`: `source/`
contains the pinned upstream checkout and `venv/` contains isolated dependencies.
To choose another location, set `ATLAS_DEEPC_SELECT_DPC_HOME` before setup and
before starting your notebook kernel. Do not move an existing virtual environment;
create it at the final location. This reduces the project/download size, but an
environment on the internal drive still consumes internal storage.

Use [the original-data comparison](notebooks/reacher/comparisons/select_dpc_original.ipynb)
or [the Atlas-data comparison](notebooks/reacher/comparisons/select_dpc_our_data_development.ipynb).
Both default to `RUN_EXPERIMENT = False`. Reading saved reports requires only
the notebook's usual NumPy/pandas/Matplotlib environment and the saved files,
not Select-DPC or PyTorch. Detailed trace cells still need the experiment SSD.
For a new run, select a fresh `RESULT_DIR`, enable `RUN_EXPERIMENT`, and choose
the batch size. Existing result directories retain their frozen source hashes;
the adapter path change does not authorize mixing new runs into old protocols.

The command-line equivalent is `python3.10 scripts/run_select_dpc.py` with the
runner's seed, step, and output-directory arguments. Add `--our-data` for the
Atlas-data adapter. Neither launcher imports Select-DPC into the Atlas controller.

`configs/select_dpc_macos_arm64_py310.lock` pins the complete dependency closure
for the validated Apple Silicon/Python 3.10 environment. Other Unix platforms
use `requirements-select-dpc.txt` (pinned direct dependencies) and require their
own validation. Setup records resolved package artifacts in the external
`pip-install-report.json`. Matching versions do not guarantee identical timings
on different hardware. Upstream tracked-source edits are rejected; our solver
and logging adaptations remain visible in the comparison adapters.

## Retained Data

The Reacher input is `data/reacher/select_iid_offline_hankel.npy`.
Consult current notebooks for result paths and the [experiment inventory](experiments/README.md).
Saved results are not necessarily included with a code-only download.
