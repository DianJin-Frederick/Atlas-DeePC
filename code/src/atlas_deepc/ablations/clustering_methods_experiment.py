"""Frozen-data, fresh serial seven-method chart-construction experiment."""

if __package__ in (None, ""):
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))


from atlas_deepc.common.paths import project_root
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_mpl")
from pathlib import Path
import sys
ROOT = project_root()
RESULT = ROOT / "experiments/routed_deepc_iid/clustering_methods_100_clarabel_v1"
FROZEN = RESULT / "frozen_protocol/src"
sys.path[:0] = [str(FROZEN), str(RESULT / "dependencies")]
import argparse
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import time
import traceback
from unittest.mock import patch
import joblib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from threadpoolctl import threadpool_limits
import baseline_atlas_deepc as baseline
import routed_deepc_experiment as routed
from baseline_1000_benchmark import summarize_timing
from candidate_goal_routing_experiment import file_digest, TIMING_KEYS, strict_success
from atlas_deepc.atlas.clustering_methods_models import ChartGate, METHODS, LANDMARK_SEED
from atlas_deepc.atlas import clustering_methods_models

VERSION = "clustering_methods_100_clarabel_v1"
DATA = ROOT / "data/select_iid_offline_hankel.npy"
OLD_ATLAS = ROOT / "experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz"
SEEDS = tuple(range(211,311))
CONFIG = replace(baseline.default_config(), lam_g=.05, svd_rank=60, clusters=48, steps=300)
baseline.BASELINE_VERSION = "linear_iterative_n10_clarabel_lamg005"


def utc():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    tmp = path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def signature():
    assert CONFIG.lam_g == .05 and CONFIG.svd_rank == 60 and CONFIG.steps == 300
    old = json.loads((ROOT / "experiments/routed_deepc_iid/linear_subspace_rank_100_v1/manifest.json").read_text())
    assert asdict(CONFIG) == old["base_config"]
    assert file_digest(DATA) == old["data_sha256"]
    return {"version": VERSION, "config": asdict(CONFIG), "solver": "CLARABEL", "max_iter":10000,
        "seeds": list(SEEDS), "methods": METHODS, "landmark_seed": LANDMARK_SEED,
        "dataset": str(DATA), "data_sha256": file_digest(DATA), "original_atlas_sha256": file_digest(OLD_ATLAS),
        "feature_channels": ["cos(q1)","cos(q2)","sin(q1)","sin(q2)","cos(q1+q2)","sin(q1+q2)"],
        "feature_preprocessing": "baseline per-channel mean/std, std floor 1e-8; four input-history coordinates have zero weight",
        "sources": {str(p.relative_to(ROOT)): file_digest(p) for p in [*sorted(FROZEN.glob("*.py")),
            Path(__file__), ROOT/"src/atlas_deepc/atlas/clustering_methods_models.py"]},
        "versions": {n:version(n) for n in ("numpy","scipy","scikit-learn","cvxpy","clarabel","gymnasium","mujoco","umap-learn","pynndescent","numba","joblib")},
        "workers": 1, "numerical_threads":1, "rollout_order":"rotate methods within each seed",
        "timing_discard_initial_steps":5, "warmup_seed":210, "warmup_steps":10,
        "medoids":"nearest original member to mean in baseline normalized features; baseline gate centroids for K-means",
        "graph":"original within-episode transition counts; mean feature distance/sqrt(count), threshold3",
        "rank_policy":"require all 48 supports >=61 and all behavioral numerical ranks >=60; no rank reduction",
        "exploratory_panel": True}


def prepare():
    RESULT.mkdir(parents=True,exist_ok=True)
    sig = signature()
    path = RESULT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == sig, "Incompatible cached experiment"
    else:
        write(path,sig)
    return sig


def construct(H, features, mean, scale, labels, centers):
    counts = np.zeros((48,48),dtype=int)
    w = routed.WINDOWS_PER_EPISODE
    for start in range(0,H.shape[1],w):
        np.add.at(counts,(labels[start:start+w-1],labels[start+1:start+w]),1)
    Y = H[routed.pipe.M*routed.pipe.L:].reshape(routed.pipe.L,10,H.shape[1])
    medoids = []
    for k in range(48):
        idx = np.flatnonzero(labels==k)
        if not len(idx):
            raise ValueError(f"Empty chart {k}")
        medoids.append(int(idx[np.argmin(np.mean((features[idx]-centers[k])**2,axis=1))]))
    states = Y[routed.pipe.T_INI-1,:,medoids].copy()
    states[:,4:6]=0
    return {"labels": labels, "centers": centers, "feature_mean":mean, "feature_scale":scale,
        "clustering_method":np.array("minibatch_kmeans"), "clustering_feature_mode":np.array("plant_joint_configuration"),
        "transition_counts":counts, "center_distance":np.sqrt(np.mean((centers[:,None]-centers[None,:])**2,axis=2)),
        "medoid_indices":np.array(medoids), "medoid_states":states}


def atlas_health(H, atlas):
    counts = np.bincount(atlas["labels"],minlength=48)
    graph = csr_matrix((atlas["transition_counts"]>=3)&~np.eye(48,dtype=bool))
    weak = connected_components(graph,directed=True,connection="weak")[0]
    strong = connected_components(graph,directed=True,connection="strong")[0]
    singular = []
    for k in range(48):
        X = H[:,atlas["labels"]==k]
        singular.append(np.linalg.svd(X,compute_uv=False) if X.shape[1] else np.array([]))
    ranks = [int(np.sum(s>s[0]*max(144,int(n))*np.finfo(float).eps)) if len(s) else 0 for s,n in zip(singular,counts)]
    return {"chart_sizes":counts.tolist(),"numerical_ranks":ranks,
        "undersized_charts":np.flatnonzero(counts<61).tolist(),
        "degenerate_charts":[k for k,r in enumerate(ranks) if r<60],
        "weak_components":int(weak),"strong_components":int(strong),
        "directed_edges":int(graph.nnz)}, singular


def fit_method(method,H,features,mean,scale,landmarks,sig):
    dest = RESULT / method
    dest.mkdir(exist_ok=True)
    if (dest/"artifact_manifest.json").exists():
        meta=json.loads((dest/"artifact_manifest.json").read_text())
        assert meta["experiment_signature"]==sig
        for name,digest in meta["files"].items():
            assert file_digest(dest/name)==digest, name
        # Existing joblib artifacts record the pre-package module name.
        with patch.dict(sys.modules, {"clustering_methods_models": clustering_methods_models}):
            gate=joblib.load(dest/"gate.joblib")
        atlas=routed.load_atlas(dest/"atlas.npz")
        cache=joblib.load(dest/"controller_cache.joblib")
        return gate,atlas,cache,meta
    start=time.perf_counter()
    if method=="kmeans":
        atlas=routed.fit_behavioral_atlas(H,CONFIG)
        original=routed.load_atlas(OLD_ATLAS)
        for key in ("labels","centers","medoid_states","transition_counts","center_distance","feature_mean","feature_scale"):
            np.testing.assert_allclose(atlas[key],original[key],rtol=1e-12,atol=1e-12,err_msg=key)
        labels=atlas["labels"]
        gate=ChartGate(method)
        from sklearn.cluster import MiniBatchKMeans
        gate.kmeans=MiniBatchKMeans(n_clusters=48)
        gate.kmeans.cluster_centers_=atlas["centers"][:,4:].copy()
        gate.kmeans._n_threads=1
        gate.kmeans.n_features_in_=6
        gate.labels=labels
        gate.arrays={"centroids":atlas["centers"][:,4:]}
        atlas=construct(H,features,mean,scale,labels,atlas["centers"])
        gate.diagnostics={"matches_original_baseline_atlas":True}
    else:
        gate=ChartGate(method).fit(features[:,4:],landmarks)
        labels=gate.labels
        sizes=np.bincount(labels,minlength=48)
        write(dest/"membership_diagnostics.json",{"sizes":sizes.tolist(),"gate":gate.diagnostics})
        np.savez_compressed(dest/"memberships.npz",column_indices=np.arange(H.shape[1]),labels=labels,
            **{f"chart_{k:02d}":np.flatnonzero(labels==k) for k in range(48)})
        joblib.dump(gate,dest/"gate.joblib")
        np.savez_compressed(dest/"fitted_parameters.npz",**gate.arrays)
        centers=np.array([features[labels==k].mean(axis=0) for k in range(48)])
        atlas=construct(H,features,mean,scale,labels,centers)
    fit_time=time.perf_counter()-start
    np.savez_compressed(dest/"memberships.npz",column_indices=np.arange(H.shape[1]),labels=labels,
        **{f"chart_{k:02d}":np.flatnonzero(labels==k) for k in range(48)})
    np.savez_compressed(dest/"fitted_parameters.npz",**gate.arrays)
    joblib.dump(gate,dest/"gate.joblib")
    np.savez_compressed(dest/"atlas.npz",**atlas)
    health,singular=atlas_health(H,atlas)
    write(dest/"health.json",health)
    if health["undersized_charts"] or health["degenerate_charts"]:
        raise ValueError(f"Cannot keep rank60: {health}")
    tick=time.perf_counter()
    cache=routed.precompute_chart_matrix_cache(H,atlas,CONFIG,geometry="linear")
    build_time=time.perf_counter()-tick
    arrays={}
    for key,(matrix,members) in cache.items():
        k=key[0]
        assert matrix["basis"].shape==(144,60) and not matrix["offset"].any() and "g_sum" not in matrix
        arrays.update({f"chart_{k:02d}_{n}":v for n,v in matrix.items()})
        arrays[f"chart_{k:02d}_singular_values"]=singular[k]
        arrays[f"chart_{k:02d}_support_indices"]=members
    np.savez_compressed(dest/"behavioral_bases.npz",**arrays,actual_basis_ranks=np.full(48,60))
    joblib.dump(cache,dest/"controller_cache.joblib")
    assert file_digest(DATA)==sig["data_sha256"]
    meta={"method":method,"experiment_signature":sig,"data_sha256":file_digest(DATA),
        "fit_and_atlas_seconds":fit_time,"basis_cache_seconds":build_time,"gate_diagnostics":gate.diagnostics,
        "health":health,"files":{p.name:file_digest(p) for p in dest.iterdir() if p.is_file() and p.suffix in (".npz",".joblib")}}
    write(dest/"artifact_manifest.json",meta)
    return gate,atlas,cache,meta


def controller_from_cache(H,atlas,cache):
    # Original initialization uses exactly this cache constructor; supply the
    # verified saved cache to make reloading independent of reclustering/SVD.
    with patch.object(routed,"precompute_chart_matrix_cache",lambda *a,**k:cache):
        return baseline.BaselineAtlasDeePC(H,atlas,CONFIG)


def assignment(gate,atlas):
    original=routed.assign_atlas_charts
    def assign(a,features):
        assert a is atlas
        if gate.method=="kmeans":
            return original(a,features)
        return gate.predict(np.asarray(features).reshape(-1,10)[:,4:])
    return assign


def validate(path,method,seed,sig):
    with np.load(path,allow_pickle=True) as t:
        row=json.loads(t["summary"].item())
        assert row["clustering_experiment_version"]==VERSION and row["chart_method"]==method
        assert row["data_sha256"]==sig["data_sha256"] and row["solver"]=="CLARABEL"
        assert all(row[k]==v for k,v in asdict(replace(CONFIG,seed=seed)).items())
        assert row["reference_mode"]=="graph" and row["graph_waypoints_enabled"]
        assert row["matrix_geometry"]=="linear" and row["candidate_count"]==10
        assert row["minimum_selected_goal_source_separation"]>=.06
        assert not row["fitted_dynamics_model"] and not row["online_nearest_trajectory_retrieval"]
        assert t["y"].shape==(300,10)
        np.testing.assert_allclose(t["distance"],np.linalg.norm(t["y"][:,6:8]-t["final_goal"].reshape(-1)[6:8],axis=1))
        for key in TIMING_KEYS:
            assert t[key].shape==(300,) and np.isfinite(t[key]).all() and np.all(t[key]>=0)
        for window in (30,100):
            d=t["distance"][-window:]
            np.testing.assert_allclose(row[f"mean_last_{window}"],d.mean())
            np.testing.assert_allclose(row[f"max_last_{window}"],d.max())
            assert row[f"success_{window}"]==bool(d.mean()<.01 and d.max()<.02)
        statuses=Counter(str(s) for s in t["status"])
        assert dict(statuses)==row["statuses"]
        assert row["unsolved_steps"]==sum(v for s,v in statuses.items() if s not in ("optimal","optimal_inaccurate"))
    return row


def run_trial(method,seed,controller,gate,atlas,sig):
    path=RESULT/method/f"seed{seed}.npz"
    if path.exists():
        return validate(path,method,seed,sig)
    temporary=path.with_suffix(".partial.npz")
    tick=time.perf_counter()
    with patch.object(routed,"assign_atlas_charts",assignment(gate,atlas)):
        row=controller.run(seed,temporary,reference_mode="graph")
    with np.load(temporary,allow_pickle=True) as t:
        arrays={key:t[key] for key in t.files if key!="summary"}
    row.update(clustering_experiment_version=VERSION,chart_method=method,solver="CLARABEL",
        data_sha256=sig["data_sha256"],result_path=str(path),rollout_wall_seconds=time.perf_counter()-tick,
        exact_target_xy=arrays["final_goal"].reshape(-1)[6:8].tolist())
    np.savez_compressed(temporary,**arrays,summary=np.array(json.dumps(row)))
    validate(temporary,method,seed,sig)
    temporary.replace(path)
    return row


def main(fit_only=False):
    RESULT.mkdir(parents=True,exist_ok=True)
    lock=(RESULT/"run.lock").open("a")
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit("Clustering experiment already active")
    write(RESULT/"launch.json",{"pid":os.getpid(),"utc":utc(),"fit_only":fit_only})
    with threadpool_limits(limits=1):
        routed.pipe.P=10
        sig=prepare()
        H=np.load(DATA,mmap_mode="r")
        tick=time.perf_counter()
        features,mean,scale=routed.behavioral_features(H,0,"plant_joint_configuration")
        feature_time=time.perf_counter()-tick
        assert H.shape==(144,37800) and features.shape==(37800,10) and not features[:,:4].any()
        landmarks=np.sort(np.random.default_rng(LANDMARK_SEED).choice(H.shape[1],1500,replace=False))
        if not (RESULT/"common_features.npz").exists():
            np.savez_compressed(RESULT/"common_features.npz",features=features[:,4:],mean=mean,scale=scale,
                column_indices=np.arange(H.shape[1]),landmark_indices=landmarks)
        write(RESULT/"feature_preparation.json",{"seconds":feature_time,"data_sha256":file_digest(DATA)})
        if not (RESULT/"fairness_audit.json").exists():
            write(RESULT/"fairness_audit.json",routed.audit_iid_library(H,eval_seeds=SEEDS))
        ready={};failures={}
        for method in METHODS:
            print(f"Fitting/loading {method}",flush=True)
            try:
                gate,atlas,cache,meta=fit_method(method,H,features,mean,scale,landmarks,sig)
                # Validate online gate repeatability and saved labels on a
                # fixed offline sample; no evaluation outcomes are inspected.
                sample=np.arange(0,H.shape[1],379)
                assign=assignment(gate,atlas)
                a=np.array([int(assign(atlas,features[i])[0]) for i in sample])
                b=np.array([int(assign(atlas,features[i])[0]) for i in sample])
                np.testing.assert_array_equal(a,b)
                np.testing.assert_array_equal(a,atlas["labels"][sample])
                controller=controller_from_cache(H,atlas,cache)
                ready[method]=(gate,atlas,controller,meta)
                print(f"READY {method}: sizes {min(meta['health']['chart_sizes'])}–{max(meta['health']['chart_sizes'])}, SCC={meta['health']['strong_components']}",flush=True)
            except Exception as exc:
                failures[method]={"stage":"offline fitting/validation","error":str(exc),"traceback":traceback.format_exc(),"utc":utc()}
                write(RESULT/method/"failure.json",failures[method]);print(f"METHOD FAILURE {method}: {exc}",flush=True)
        write(RESULT/"offline_status.json",{"ready":list(ready),"failures":failures,"utc":utc()})
        if fit_only:
            return
        solve=routed.pipe.DeePCController.solve
        def checked(controller,*args,**kwargs):
            assert controller.solver=="CLARABEL"
            routed.pipe.validate_objective_options(CONFIG.lam_g, CONFIG.lam_du)
            return solve(controller,*args,**kwargs)
        rows={m:[] for m in ready};trial_failures={m:[] for m in ready}
        methods=list(ready)
        with patch.object(routed.pipe.DeePCController,"solve",checked):
            for method,(gate,atlas,controller,meta) in ready.items():
                config=controller.config
                try:
                    controller.config=replace(config,steps=10)
                    with patch.object(routed,"assign_atlas_charts",assignment(gate,atlas)):
                        controller.run(210,reference_mode="graph")
                except Exception as exc:
                    write(RESULT/method/"warmup_failure.json",{"error":str(exc),"utc":utc()})
                finally: controller.config=config
            for index,seed in enumerate(SEEDS):
                shift=index%len(methods) if methods else 0
                for method in methods[shift:]+methods[:shift]:
                    gate,atlas,controller,meta=ready[method]
                    try:
                        row=run_trial(method,seed,controller,gate,atlas,sig)
                        rows[method].append(row)
                        print(f"{method} seed {seed}: success={strict_success(row)}, unsolved={row['unsolved_steps']}",flush=True)
                    except Exception as exc:
                        failure={"seed":seed,"error":str(exc),"traceback":traceback.format_exc(),"utc":utc()}
                        trial_failures[method].append(failure)
                        write(RESULT/method/f"seed{seed}_failure.json",failure)
                        print(f"TRIAL FAILURE {method} seed {seed}: {exc}",flush=True)
                targets=[next((r["exact_target_xy"] for r in rows[m] if r["seed"]==seed),None) for m in methods]
                targets=[t for t in targets if t is not None]
                if targets: np.testing.assert_array_equal(targets,[targets[0]]*len(targets))
                if (index+1)%10==0:
                    assert signature()==sig
                    for method,(_,_,_,meta) in ready.items():
                        for name,digest in meta["files"].items(): assert file_digest(RESULT/method/name)==digest
                write(RESULT/"progress.json",{"utc":utc(),"complete":False,"seeds_attempted":index+1,
                    "rollouts_completed":sum(map(len,rows.values())),"offline_failures":failures,
                    "methods":{m:{"completed":len(rows[m]),"successes":sum(strict_success(r) for r in rows[m]),
                                   "trial_failures":len(trial_failures[m])} for m in methods}})
        panels={}
        for method in methods:
            items=sorted(rows[method],key=lambda r:r["seed"])
            panels[method]={"method":method,"complete":len(items)==100,"results":items,
                "successes":sum(strict_success(r) for r in items),"trial_failures":trial_failures[method],
                "timing":summarize_timing([r["result_path"] for r in items]) if items else None,
                "offline":ready[method][3]}
            write(RESULT/method/"summary.json",panels[method])
        assert signature()==sig
        write(RESULT/"summary.json",{"version":VERSION,"complete":True,"all_methods_succeeded":len(ready)==7 and all(p["complete"] for p in panels.values()),
            "completed_utc":utc(),"signature":sig,"panels":panels,"offline_failures":failures,
            "rollouts":sum(len(p["results"]) for p in panels.values()),"data_hash_final":file_digest(DATA)})
        print("EXPERIMENT FINISHED",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-only",action="store_true")
    main(parser.parse_args().fit_only)
