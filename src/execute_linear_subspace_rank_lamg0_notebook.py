"""Execute the completed lambda_g=0 rank sweep and matched comparison report."""

from pathlib import Path
from execute_linear_subspace_rank_notebook import main

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    main(result=ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_lamg0_100_v1",
         notebook_path=ROOT / "notebooks/ablations/ablation_linear_subspace_rank_lamg0.ipynb",
         expected_figures=7)
