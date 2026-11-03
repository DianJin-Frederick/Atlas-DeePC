"""Verified development references and clustering helpers for Reacher comparisons."""
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from comparison_methods.common import reacher as utils
from comparison_methods.common import multiview as clustering
from comparison_methods.common.multiview import hankel, joint_nmf

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data/reacher/select_iid_offline_hankel.npy"
RESULT_STORE = Path('/Volumes/Crucial X10/atlas_deepc_data')

REFERENCE_SOURCE = ROOT / "experiments/reacher/four_baselines_k64_rank_projection_chordal_v1/selected_serial/rank053/output8/terminal"
REFERENCE_SUMMARY_SHA256 = "e6e3179ed68a04ee8d853c6259139d5d213aabb670687f52273114c2131daf19"
REFERENCE_MANIFEST_SHA256 = "193cacc035044fd4bb41fa7718279db9d7f2f0999be30b98638e18ca79b254a9"


def read_references(source=REFERENCE_SOURCE,data=DATA):
    """Recover fixed development goals from saved indices, NOT test rollouts.

    The old traces were deleted. The retained summary stores each selected
    offline column. The original policy uses that column's latest past posture,
    zero velocity, and the seeded environment target (not the sample's XY).
    No controller simulation, graph routing, reference selection or tuning.
    """
    source=Path(source)
    manifest_path=source/"manifest.json"; summary_path=source/"summary.json"
    if utils.digest(manifest_path)!=REFERENCE_MANIFEST_SHA256 or utils.digest(summary_path)!=REFERENCE_SUMMARY_SHA256:
        raise ValueError("Development reference source changed; refusing unverified references")
    manifest=json.loads(manifest_path.read_text())
    if (manifest["output_dimension"]!=8 or manifest["config"]["clusters"]!=64
            or manifest["config"]["svd_rank"]!=53
            or manifest["terminal_reference_policy"]!="fixed_nearest_eligible_fingertip"
            or utils.digest(data)!=manifest["data_sha256"]):
        raise ValueError("Wrong p8/K64/rank53 reference source or offline dataset")
    rows=json.loads(summary_path.read_text())["results"]
    if len(rows)!=100 or sorted(r["seed"] for r in rows)!=list(range(211,311)):
        raise ValueError("Reference source must contain exactly development seeds 211--310")
    h=np.load(data,mmap_mode="r")
    if h.shape!=(144,37800):
        raise ValueError("Unexpected offline Hankel layout")
    samples=h[24:].reshape(12,10,h.shape[1])[1].T[:,:8]
    references={}; indices={}
    env=gym.make("Reacher-v5")
    try:
        for row in rows:
            seed=int(row["seed"]); index=int(row["data_goal_index"])
            if not 0<=index<len(samples):
                raise ValueError("Invalid saved goal column")
            obs,_=env.reset(seed=seed); target=obs[4:6].copy()
            goal=samples[index].copy()
            if not np.isfinite(goal).all() or not np.allclose(goal[:2]**2+goal[2:4]**2,1,atol=1e-10):
                raise ValueError("Invalid offline posture")
            if not np.isclose(np.linalg.norm(goal[6:8]-target),row["data_goal_xy_error"],atol=1e-10,rtol=0):
                raise ValueError("Saved goal index and seeded target do not agree")
            goal[4:6]=0.; goal[6:8]=target
            references[seed]=goal; indices[str(seed)]=index
    finally:
        env.close()
    return references,{"source_summary":str(summary_path.relative_to(ROOT)),
        "source_summary_sha256":REFERENCE_SUMMARY_SHA256,
        "source_manifest_sha256":REFERENCE_MANIFEST_SHA256,
        "offline_data_sha256":manifest["data_sha256"],
        "atlas_sha256":manifest["atlas_sha256"],"saved_goal_indices":indices,
        "method":"Reconstructed from saved development column indices: H_y at past index 1, velocity=0, XY=seeded target",
        "control_timing_excludes_reference_setup":True}

