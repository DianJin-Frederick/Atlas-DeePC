"""Canonical assets, paper results and the licensed F1TENTH simulator."""
import sys
from atlas_deepc.common.paths import ROOT

DATA_DIR = ROOT / 'data/f1tenth'
RESULTS_DIR = ROOT / 'experiments/f1tenth'
NOISE_SWEEP_DIR = RESULTS_DIR / 'noise_ratio_sweep_20261027'
PAPER_DIR = NOISE_SWEEP_DIR / 'sigma_0p01'
CONTEXTUAL_DIR = PAPER_DIR
CLEAN_OFFLINE_DATA_FILE = DATA_DIR / 'offline_300_seconds.npz'
def noise_tag(sigma):
    return format(float(sigma),'.12g').replace('.','p')

def noisy_data_file(sigma):
    return DATA_DIR / f'offline_300_seconds_sigma_{noise_tag(sigma)}_seed3408.npz'

OFFLINE_DATA_FILE = noisy_data_file(.01)
WAYPOINT_FILE = DATA_DIR / 'example_waypoints.csv'
MAP_BASE = DATA_DIR / 'maps/example_map'
SIMULATOR_ROOT = ROOT / 'third_party/f1tenth_gym'
if str(SIMULATOR_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMULATOR_ROOT))
