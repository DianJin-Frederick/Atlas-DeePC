"""KDE-BDPC Eqs. (14), (18), (24), (27)--(34), one-sample prediction.

Separate from the historical T_ini=2/T_f=10 adaptation. Follows the PRINTED
connection objectives (27a)/(28a), which are not generally equivalent to (26).
No claim of author validation or reproduction of the paper's battery example.
"""
import argparse
from dataclasses import asdict, dataclass
import fcntl
import importlib.metadata
import json
from pathlib import Path
import time

from comparison_methods.kde_bdpc import core
from comparison_methods.kde_bdpc.routing import solve_minimum_mean_route
from comparison_methods.kde_bdpc import routing as minimum_mean_route

np, cp = core.np, core.cp
OUTPUT = core.SSD/'experiments/reacher/kde_bdpc_eq34_p8_l3_v1'


@dataclass(frozen=True)
class Config(core.Config):
    past: int = 2  # L-1 overlap; with future=1 this sets L=3 for Reacher.
    future: int = 1  # Eq.34 predicts exactly one new sample.
    clusters: int = 4
    beta_bound: float = .05  # Paper's battery value; not tuned for Reacher.
    lambda_epsilon: float = 1000.
    lambda_J: float = 1.
    connection_max_steps: int | None = None  # None: repeat until Eq.29/stagnation.
    same_initial_cluster: str = 'reject'  # Paper Sec.7.1 assumes distinct clusters.

    def validate(self):
        values = asdict(self)
        extras = ('lambda_J', 'connection_max_steps', 'same_initial_cluster')
        for key in extras:
            values.pop(key)
        # Parent checks numeric offline/solver fields. Eq.34 allows PSD weights,
        # including zero, so validate the actual cost weights separately.
        for key in ('q_joint', 'q_dq', 'q_xy', 'r'):
            if not np.isfinite(values[key]) or values[key] < 0:
                raise ValueError('Cost weights must be nonnegative')
            values[key] = max(values[key], 1e-30)
        core.Config(**values).validate()
        if self.future != 1:
            raise ValueError('Eq.34 requires future=1 and past=L-1; multistep is a different controller')
        if not np.isfinite(self.lambda_J) or self.lambda_J < 0:
            raise ValueError('lambda_J must be nonnegative')
        if self.connection_max_steps is not None and (type(self.connection_max_steps) is not int or self.connection_max_steps < 1):
            raise ValueError('connection_max_steps must be None or a positive integer')
        if self.same_initial_cluster not in ('reject', 'direct_reference'):
            raise ValueError('Unknown same-initial-cluster policy')


@dataclass(frozen=True)
class AffineCluster:
    """Eq.14: c + Fg. The coefficient set is imposed by the relevant problem."""
    center: np.ndarray
    basis: np.ndarray

    def reconstruct(self, g):
        return self.center + self.basis @ np.asarray(g)


class Library(core.Library):
    # Reuse the already-checked global scaling, Eq.9 MVC, mean-centered Eq.6
    # offline bases, Eq.15 KDE, and Theorem3/Eq.24 rank search without alteration.
    @property
    def affine_clusters(self):
        return [AffineCluster(c, f) for c, f in zip(self.centers, self.bases)]

    def connection(self, center, target, source=None):
        """Algorithms1--2, printed27a/28a; no default iteration budget."""
        center = np.asarray(center, float).copy()
        threshold = self.log_density(self.centers[target], target) + np.log(self.config.beta_s)
        points, ranks = [], []
        iteration = 0
        while True:
            if source is not None and iteration == 0:
                nxt = self.first_cluster_step(source, target)
                rank = self.bases[source].shape[1]
            else:
                f, _, info = self.interpolate(center)
                rank = info['rank']
                # Positive rescaling/constant removal of printed Eq.28a:
                # minimize distance to target sample mean in the local ball.
                g = f.T @ (self.sample_means[target]-center)
                norm = np.linalg.norm(g)
                if norm > self.config.beta_bound:
                    g *= self.config.beta_bound/norm
                nxt = center+f@g
            if not np.isfinite(nxt).all():
                raise core.PlanningError('Nonfinite cluster connection')
            points.append(nxt.tolist()); ranks.append(int(rank)); iteration += 1
            ratio = self.log_density(nxt, target)-self.log_density(self.centers[target], target)
            if self.log_density(nxt, target) >= threshold-self.config.accept_residual:
                return dict(points=points, ranks=ranks,
                            weight=float(np.mean([self.entropy(np.array(p)) for p in points])),
                            end_log_density_ratio=ratio)
            if np.linalg.norm(nxt-center) < self.config.path_tol:
                raise core.PlanningError('Connection stalled before Eq.29 was satisfied')
            if self.config.connection_max_steps is not None and iteration >= self.config.connection_max_steps:
                raise core.PlanningError('Explicit connection resource guard reached before Eq.29')
            center = nxt

    def plan(self, center, reference, initial_cluster=None):
        current = int(np.argmax(self.membership(center)))
        destination = int(np.argmax(self.membership(reference)))
        initial = current if initial_cluster is None else initial_cluster
        if initial == destination:
            if self.config.same_initial_cluster == 'reject':
                raise core.PlanningError('Outside Sec.7.1 setup: initial and reference dominant clusters coincide')
            return reference, dict(route=[-1,destination], mean_entropy=self.entropy(center),
                next_cluster=destination, target_cluster=destination, failed_connections={},
                route_certificate={'method':'explicit_same_initial_cluster_extension'})
        edges = {tuple(map(int,k.split(':'))):v['weight'] for k,v in self.graph.items()}
        failures = {}
        for j in range(len(self.bases)):
            if j == initial:
                continue
            try:
                edges[-1,j] = self.connection(center,j)['weight']
            except core.PlanningError as exc:
                failures[str(j)] = str(exc)
        try:
            route, cost, certificate = solve_minimum_mean_route(edges,-1,destination)
        except (ValueError, RuntimeError) as exc:
            raise core.PlanningError(f'Eq.32 route rejected: {exc}') from exc
        next_cluster = route[1]
        target = reference if next_cluster == destination else self.centers[next_cluster]
        return target, dict(route=route, mean_entropy=cost, next_cluster=next_cluster,
            target_cluster=destination, failed_connections=failures, route_certificate=certificate)


def psd_weight(value, size, name):
    value = np.asarray(value, float)
    if value.ndim == 0:
        value = np.eye(size)*value
    if value.ndim == 1:
        value = np.diag(value)
    if value.shape != (size,size) or not np.isfinite(value).all():
        raise ValueError(f'Invalid {name} shape/values')
    if not np.allclose(value,value.T,atol=1e-12,rtol=0) or np.linalg.eigvalsh(value).min() < -1e-12:
        raise ValueError(f'{name} must be symmetric positive semidefinite')
    return value


def build_problem(basis, measured_window, target, config, Lambda_u=None, Lambda_y=None, extra_cost=None):
    """Eq.34a--c in time-major [u_t,y_t] order (a fixed row permutation).

    v = w[k-L+1:k] + Fbar*g represents w[k-L+2:k+1].
    epsilon = v[:-1] - w[k-L+2:k]; both u and y have slack.
    v[-1] is the only future sample, with no future epsilon.
    extra_cost(v) is an explicit convex nonnegative application-specific J(k).
    None means J(k)=0, not an omitted coefficient regularizer.
    """
    config.validate()
    length = config.length
    history, target, basis = map(lambda x:np.asarray(x,float), (measured_window,target,basis))
    if history.shape != (length,10) or target.shape != (length,10) or basis.ndim != 2 or basis.shape[0] != 10*length:
        raise ValueError('Expected L measured/target samples and an L*10-row basis')
    if basis.shape[1] < 1 or not all(np.isfinite(x).all() for x in (history,target,basis)):
        raise ValueError('Empty/nonfinite control data')
    if not np.allclose(basis.T@basis,np.eye(basis.shape[1]),atol=1e-8,rtol=0):
        raise ValueError('Eq.34 requires an orthonormal interpolated basis')
    Lu = psd_weight(config.r if Lambda_u is None else Lambda_u,2*length,'Lambda_u')
    Ly = psd_weight(np.tile(config.output_weights,length) if Lambda_y is None else Lambda_y,8*length,'Lambda_y')
    g = cp.Variable(basis.shape[1],name='g')
    future_window = cp.reshape(history.ravel()+basis@g,(length,10),order='C')
    epsilon = future_window[:-1]-history[1:]
    du = cp.reshape(future_window[:,:2]-target[:,:2],(2*length,),order='C')
    dy = cp.reshape(future_window[:,2:]-target[:,2:],(8*length,),order='C')
    J = cp.Constant(0.) if extra_cost is None else cp.Expression.cast_to_const(extra_cost(future_window))
    if J.shape != () or not J.is_convex() or not J.is_nonneg():
        raise ValueError('J(k) must be a scalar convex nonnegative expression for this conic implementation')
    terms = dict(output_tracking=cp.quad_form(dy,cp.psd_wrap(Ly)),
                 input_tracking=cp.quad_form(du,cp.psd_wrap(Lu)),
                 past_mismatch=config.lambda_epsilon*cp.sum_squares(epsilon),
                 application_cost=config.lambda_J*J)
    problem = cp.Problem(cp.Minimize(sum(terms.values())),[cp.norm(g,2)<=config.beta_bound])
    return problem,g,epsilon,future_window,terms


def compute_action(lib,u_history,y_history,reference8,initial_cluster=None):
    started = time.perf_counter(); c=lib.config
    physical_history = np.c_[u_history,y_history]
    if physical_history.shape != (c.length,10):
        raise ValueError('Need exactly L measured input-output pairs')
    history = (physical_history-lib.mean)/lib.scale
    reference = np.tile((np.r_[np.zeros(2),reference8]-lib.mean)/lib.scale,c.length)
    t=time.perf_counter(); basis,weights,rank_info=lib.interpolate(history.ravel())
    interpolation_seconds=time.perf_counter()-t
    t=time.perf_counter(); target,plan=lib.plan(history.ravel(),reference,initial_cluster)
    planning_seconds=time.perf_counter()-t
    t=time.perf_counter()
    problem,g,epsilon,window,terms=build_problem(basis,history,target.reshape(c.length,10),c)
    preparation_seconds=time.perf_counter()-t
    t=time.perf_counter()
    try:
        problem.solve(solver='CLARABEL',max_threads=1,max_iter=c.solver_max_iter,
                      tol_gap_abs=c.solver_tol,tol_gap_rel=c.solver_tol,tol_feas=c.solver_tol,warm_start=False)
    except cp.error.SolverError as exc:
        raise core.PlanningError(f'Eq.34 solve failed: {exc}') from exc
    solve_seconds=time.perf_counter()-t
    if problem.status not in ('optimal','optimal_inaccurate') or window.value is None or not np.isfinite(window.value).all():
        raise core.PlanningError(f'Eq.34 solve rejected: {problem.status}')
    violation=float(max(np.max(x.violation()) for x in problem.constraints))
    if violation>c.accept_residual:
        raise core.PlanningError(f'Eq.34 coefficient-ball violation: {violation}')
    physical=np.asarray(window.value)*lib.scale+lib.mean
    row=dict(command_u=physical[-1,:2].copy(),command_u_horizon=physical[-1:,:2].copy(),
        pred_y=physical[-1,2:].copy(),pred_y_horizon=physical[-1:,2:].copy(),
        reconstructed_window=physical,affine_offset=history.copy(),interpolated_basis=basis.copy(),
        membership=weights,rank=rank_info['rank'],rank_objective=rank_info['cost'],
        rank_candidate_costs=rank_info['candidate_costs'],g=g.value.copy(),past_slack=epsilon.value.copy(),
        planned_target=target.reshape(c.length,10)*lib.scale+lib.mean,
        route=plan['route'],next_cluster=plan['next_cluster'],target_cluster=plan['target_cluster'],
        path_mean_entropy=plan['mean_entropy'],failed_connections=plan['failed_connections'],
        route_certificate=json.dumps(plan['route_certificate']),status=problem.status,
        max_constraint_violation=violation,objective=float(problem.value),
        objective_terms=np.array([float(x.value) for x in terms.values()]),
        interpolation_time_s=interpolation_seconds,planning_time_s=planning_seconds,
        matrix_preparation_time_s=preparation_seconds,solve_time_s=solve_seconds,
        control_compute_time_s=time.perf_counter()-started)
    # Rank varies online, so keep basis as JSON rather than a ragged NPZ array.
    row['interpolated_basis']=json.dumps(row['interpolated_basis'].tolist())
    return physical[-1,:2].copy(),row


CONVENTIONS = dict(
    cluster='Eq.14: affine c_i + F_i g; center is mean length-L segment, F_i fitted to deviations.',
    online='Eq.18: offset is the entire latest measured L-sample window, not a cluster center or zero.',
    rank='Theorem3/Eq.24: weighted projector eigendecomposition and rank search. Offline SVD cutoff is explicit user choice.',
    controller='Eq.34: one-sample shift, L-1 past u/y samples with squared slack, one future u/y sample without slack, coefficient ball.',
    J='Reacher application cost J(k)=0. Generic builder supports explicit convex nonnegative J.',
    connection='PRINTED27a/28a sum of Gaussian exponents; not generally equivalent to KDE argmax26. Eq27 SLSQP is local.',
    graph='Directed verified connections; minimum mean weight over simple paths. Simple-path/direction conventions made explicit; no enumeration cap or approximate fallback.',
    initial_cluster='Default rejects same initial/reference dominant cluster, outside Sec7.1 assumption; direct_reference is an explicit optional extension.',
    normalization='Global channel z-score; weights and coefficient radius apply in standardized coordinates.',
    actuation='No additional optimizer bounds. Reacher physical clipping to +/-1 is recorded separately.',
    scope='Paper-equation implementation for Reacher; not author-verified or battery-example reproduction.')


def run(config=Config(),output=OUTPUT,seeds=(211,),steps=150,prepare_only=False):
    config.validate(); output=Path(output); seeds=list(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or not set(seeds)<=set(core.SEEDS) or not 1<=steps<=300:
        raise ValueError('Distinct development seeds 211--310 and 1..300 steps required')
    if not core.SSD.is_dir():
        raise FileNotFoundError('Connect Crucial X10')
    if not output.resolve().is_relative_to((core.SSD/'experiments/reacher').resolve()):
        raise ValueError('Store paper-equation runs in the external-drive experiment directory')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'run.lock').open('a+') as lock, core.threadpool_limits(limits=1):
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        refs, provenance=core.shared.read_references()
        signature=dict(version='kde_bdpc_eq34_one_sample_v1',config=asdict(config),
            conventions=CONVENTIONS,data_sha256=core.shared.utils.digest(core.DATA),reference_provenance=provenance,
            paper_sha256=core.shared.utils.digest(core.PAPER),
            sources={str(p):core.shared.utils.digest(p) for p in
                     (__file__,core.__file__,core.shared.__file__,core.shared.clustering.__file__,core.shared.utils.__file__,minimum_mean_route.__file__)},
            packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','cvxpy','clarabel','gymnasium','mujoco')})
        signature['geometry_reuse']=None
        manifest=output/'manifest.json'
        if manifest.exists():
            if json.loads(manifest.read_text())!=signature:
                raise ValueError('Changed manifest; use a new output directory')
        elif any(output.glob('*.npz')):
            raise ValueError('Refusing unversioned data reuse')
        else:
            core.shared.utils.write_json(manifest,signature)
        try:
            cache=output/'offline_model.npz'
            if cache.exists():
                lib=Library.load(cache,config)
            else:
                core.shared.utils.write_json(output/'progress.json',dict(state='preparing_offline'))
                lib=Library(*core.shared.utils.recover_episodes(np.load(core.DATA,mmap_mode='r')),config)
                lib.prepare_graph()
                lib.audit.update(conventions=CONVENTIONS)
                lib.save(cache); core.shared.utils.write_json(output/'offline_audit.json',lib.audit)
            if prepare_only:
                core.shared.utils.write_json(output/'progress.json',dict(state='offline_ready'))
                return lib.audit
            panel=dict(seeds=seeds,steps=steps)
            if (output/'panel.json').exists() and json.loads((output/'panel.json').read_text())!=panel:
                raise ValueError('Panel mismatch; use another output directory')
            core.shared.utils.write_json(output/'panel.json',panel)
            # Pass the paper controller explicitly to the shared rollout loop.
            for seed in seeds:
                path=output/f'seed{seed}_control_results.npz'
                if not path.exists():
                    core.run_seed(lib,seed,steps,output,refs[seed],action_fn=compute_action)
                else:
                    with np.load(path) as z:
                        row=json.loads(z['summary'].item())
                        if row['seed']!=seed or row['requested_steps']!=steps or row['future']!=1:
                            raise ValueError('Existing trace does not match Eq.34 protocol')
                result=core.summarize(output,seeds)
            core.shared.utils.write_json(output/'progress.json',dict(state='complete',finished_seeds=len(seeds)))
            return result
        except BaseException as exc:
            core.shared.utils.write_json(output/'progress.json',dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'error',message=str(exc)))
            raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--seeds',type=int,nargs='+',default=[211])
    parser.add_argument('--steps',type=int,default=150)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    config=Config(**json.loads(args.config.read_text())) if args.config else Config()
    print(json.dumps(run(config,args.output,args.seeds,args.steps,args.prepare_only),indent=2))
