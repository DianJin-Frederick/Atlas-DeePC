"""Frozen settings for the two retained paper experiments."""
from dataclasses import dataclass
import numpy as np

TI, TF, DT = 5, 10, 0.1
SELECTION_METHOD = 'full_window_projection'
WARMUP_STEPS = TI + TF
Q_PHYSICAL = np.array([100., 100., 1., 1.])
R_PHYSICAL = np.array([0.1, 10.])
LIMIT = np.array([3., 0.4189])  # acceleration, steering angle
NOISE = np.array([0.05, 0.05, 0.05, np.deg2rad(0.6)])
PAPER_SEEDS = tuple(range(511, 531))
ATLAS_RANKS = (24, 28, 32, 36, 40, 44, 48, 56, 64)
# Search support only: changing these lists does not change the controller.
# Each requested rank must also fit within every chart's sample count.
ATLAS_CLUSTER_COUNTS = tuple(range(1, 25))
ATLAS_SEARCH_RANKS = tuple(range(16, 87))
CLEAN_DATA_SHA256 = 'c230b93e913cebf203cb98ee836258e61eab180746ce8218f9f51ba803e98146'
OFFLINE_NOISE_SEED = 3408  # Independent of collection (3407) and evaluation seeds.
DEFAULT_NOISE_SIGMA = 0.01
NOISE_SIGMAS = (0., 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05)

@dataclass(frozen=True)
class AtlasConfig:
    clusters: int = 2
    rank: int = 40
    lam_ini: float = 100.
    weights: str = 'position'

