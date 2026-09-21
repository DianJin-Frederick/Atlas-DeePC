"""Verified paper figures for the saved waypoint-enabled seed 573 rollout."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
from plot_no_waypoint_success import output, arm, style, ORANGE, INK, BLUE, RED
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'experiments/routed_deepc_iid/baseline_linear_iterative_n10_v1/evaluation_1000/seed573_control_results.npz'
ATLAS = ROOT / 'experiments/routed_deepc_iid/clustering_jointconfig48_fresh211_310/behavioral_atlas.npz'
DEST = ROOT / 'paper/figures/reacher_waypoint_seed573'


def save(fig, name):
    fig.savefig(DEST / (name + '.pdf'))
    fig.savefig(DEST / (name + '.png'), dpi=300)
    plt.close(fig)


def combined_figure(a, atlas, states, target, chart, error, stop, terminal_start, initial_route, curve, curve_points, arc):
    """One integrated figure: motion/route and time-aligned controller diagnostics."""
    # Preserve the paper's orange; use Paul Tol Vibrant blue/teal accents.
    ORANGE = '#df7b20'
    INK = '#28343D'
    RED = '#AF5147'
    route_color = '#0077BB'
    initial_color = '#009988'
    # Use actual recorded one-step graph requests, not a planned path snapshot.
    active_steps = np.flatnonzero(np.linalg.norm(a['ref'][:,6:8]-target, axis=1) >= 1e-10)
    requests = []
    for t in active_steps:
        edge = (int(a['chart'][t]), int(a['next_chart'][t]))
        if not requests or edge != requests[-1]['edge']:
            requests.append({'start':int(t), 'stop':int(t)+1, 'edge':edge})
        else:
            requests[-1]['stop'] = int(t)+1
    used_edges = list(dict.fromkeys(r['edge'] for r in requests))
    waypoint_nodes = list(dict.fromkeys(v for _, v in used_edges))
    reference_steps = [r['start'] for r in requests]
    width, height = 6., 2.75
    fig = plt.figure(figsize=(width, height))
    def panel(left, bottom, w, h):
        ax = fig.add_axes([left/width, bottom/height, w/width, h/height])
        style(ax)
        return ax
    motion = panel(.48, .40, 2.70, 1.72)
    selection = panel(3.82, 1.43, 2.05, .69)
    err = panel(3.82, .40, 2.05, .63)
    snapshots = np.unique(np.searchsorted(arc, np.linspace(0, arc[-1], 8)[1:-1]))
    for t in snapshots:
        motion.plot(*arm(states[t]).T, color=ORANGE, alpha=.45, lw=.8, zorder=1)
    for t in reference_steps:
        # Each intermediate reference is an observed chart representative;
        # do not render the separately completed terminal reference as a pose.
        np.testing.assert_allclose(a['ref'][t,[0,1,2,3,6,7,8,9]],
            atlas['medoid_states'][a['next_chart'][t],[0,1,2,3,6,7,8,9]])
        motion.plot(*arm(a['ref'][t]).T, color=route_color, alpha=.85, lw=.9,
            marker='o', ms=2, markeredgewidth=.5, markerfacecolor='white', zorder=2)
    motion.plot(*curve_points.T, color=ORANGE, lw=1.45, zorder=4)
    for fraction in [.22, .62]:
        t = np.searchsorted(arc, fraction*arc[-1])
        motion.annotate('', xy=curve(t+.3), xytext=curve(t-.3),
            arrowprops={'arrowstyle':'-|>', 'color':ORANGE, 'lw':.8, 'mutation_scale':7})
    for state, color, ls in [(states[0], initial_color, '--'), (states[stop], INK, '-')]:
        motion.plot(*arm(state).T, color=color, ls=ls, lw=1.3, marker='o', ms=3,
            markerfacecolor='white', markeredgewidth=.8, zorder=5)
    points = atlas['medoid_states'][:,6:8]*100
    offsets = {43:(10,1),16:(10,0),44:(5,10),5:(0,-11),26:(-12,7),48:(-8,7),30:(0,-13)}
    for k in waypoint_nodes:
        motion.scatter(*points[k], s=22, marker='D', facecolor='white', edgecolor=route_color, lw=1, zorder=7)
        motion.annotate(str(k+1), points[k], xytext=offsets[k+1], textcoords='offset points',
            fontsize=8, ha='center', va='center', color=route_color, zorder=9,
            bbox={'facecolor':'white','edgecolor':'none','pad':.3,'alpha':.9})
    motion.scatter(*target*100, marker='x', s=27, color=RED, lw=1.1, zorder=10)
    motion.scatter(0,0,s=12,color=INK,zorder=8)
    motion.set(xlim=(-12,-12+24*2.70/1.72), ylim=(-6,18), xticks=[-10,0,10,20], yticks=[0,10],
        xlabel=r'$p_{\mathrm{f}}^x$ (cm)', ylabel=r'$p_{\mathrm{f}}^y$ (cm)')
    motion.set_aspect('equal',adjustable='box')
    motion.set_title('(a) Motion and waypoint references', loc='left', pad=7)
    selection.axvspan(terminal_start,stop,color='#f3f4f5',zorder=0)
    selection.stairs(chart[:stop],np.arange(stop+1),baseline=None,color=ORANGE,lw=1.3,zorder=4)
    selection.stairs(a['next_chart'][:terminal_start]+1,np.arange(terminal_start+1),baseline=None,
        color=route_color,ls=':',lw=1.2,zorder=5)
    selection.axvline(terminal_start,color='#8e969c',ls=':',lw=.8)
    selection.text(58,53,'Terminal reference active',fontsize=8,ha='center')
    selection.set(xlim=(0,stop),ylim=(0,63),xticks=[0,24,60,90],yticks=[1,24,48],ylabel='Chart ID')
    selection.tick_params(axis='x',labelbottom=False)
    selection.set_title('(b) Controller and waypoint charts',loc='left',pad=7)
    selection.legend(handles=[Line2D([],[],color=ORANGE,lw=1.3,label='Controller chart'),
        Line2D([],[],color=route_color,lw=1.2,ls=':',label='Waypoint chart')],loc='lower right',
        frameon=False,ncol=1,fontsize=8,handlelength=1.8,handletextpad=.4,labelspacing=.15,borderpad=.1)
    err.axvline(terminal_start,color='#8e969c',ls=':',lw=.8)
    err.semilogy(np.arange(stop+1),error[:stop+1],color=ORANGE,lw=1.3)
    err.axhline(10,color='#8e969c',lw=.7,ls=(0,(4,3)))
    err.text(stop-2,14,'10 mm',fontsize=8,ha='right',color='#777777')
    err.scatter(stop,error[stop],s=11,color=ORANGE,clip_on=False)
    err.annotate(f'{error[stop]:.2f} mm',xy=(stop,error[stop]),xytext=(-4,-13),textcoords='offset points',ha='right',fontsize=8)
    err.set(xlim=(0,stop),ylim=(1,350),xticks=[0,24,60,90],yticks=[1,10,100],
        xlabel='Control step',ylabel='Error (mm)')
    err.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f'{v:g}'))
    err.yaxis.set_minor_locator(ticker.NullLocator())
    err.set_title('(c) Tracking error',loc='left',pad=7)
    handles=[Line2D([],[],color=ORANGE,lw=1.3,label='Path (interp.)'),
        Line2D([],[],color=initial_color,lw=1.3,ls='--',label='Initial pose'),
        Line2D([],[],color=INK,lw=1.3,label='Final pose'),
        Line2D([],[],color=ORANGE,lw=1.3,alpha=.45,label='Intermediate poses'),
        Line2D([],[],color=route_color,lw=1.3,alpha=.85,marker='D',mfc='white',ms=3,label='Waypoint poses'),
        Line2D([],[],color=RED,marker='x',ms=4,ls='',label='Target')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.994),ncol=3,
        frameon=False,fontsize=8,handlelength=2.2,handletextpad=.45,columnspacing=.9,labelspacing=.5)
    fig.canvas.draw()
    # All diagnostic axes have exactly the same horizontal extent.
    np.testing.assert_allclose(selection.get_position().extents[[0,2]], err.get_position().extents[[0,2]])
    save(fig,'reacher_waypoint_combined')
    metadata = {'seed':573, 'source':str(SOURCE),
        'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'edge_definition':'Recorded current-chart to next-waypoint-chart requests, only while intermediate reference is active',
        'online_requests':[{'start_step':r['start'], 'stop_step_exclusive':r['stop'],
            'source_chart':r['edge'][0]+1,'waypoint_chart':r['edge'][1]+1} for r in requests],
        'waypoint_pose_steps':reference_steps,'terminal_reference_first_step':terminal_start,
        'requested_edges_match_initial_route':used_edges==list(zip(initial_route[:-1],initial_route[1:])),
        'display_steps':stop,'figure_inches':[width,height],
        'graph_edges_displayed':False,
        'palette':{'path':ORANGE,'waypoint':route_color,'initial_pose':initial_color,'final_pose':INK},
        'waypoint_chart_ids_displayed':[int(k)+1 for k in waypoint_nodes],
        'graph_edges_are_not_physical_trajectories':True}
    (DEST/'combined_plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    with np.load(SOURCE, allow_pickle=True) as f:
        a = {k: f[k] for k in f.files}
    with np.load(ATLAS, allow_pickle=True) as f:
        atlas = {k: f[k] for k in f.files}
    summary = json.loads(a['summary'].item())
    assert summary['reference_mode'] == 'graph' and summary['lam_g'] == .05
    assert summary['success_30'] and summary['success_100']
    target = a['final_goal'][6:8]
    env = gym.make('Reacher-v5', max_episode_steps=327)
    obs, _ = env.reset(seed=573)
    np.testing.assert_allclose(obs[4:6], target)
    for _ in range(2):
        obs, *_ = env.step(np.zeros(2))
    initial = output(obs)
    replay = []
    for u in a['u']:
        obs, *_ = env.step(u)
        replay.append(output(obs))
    env.close()
    np.testing.assert_allclose(replay, a['y'], rtol=1e-8, atol=1e-9)
    states = np.vstack([initial, a['y']])
    error = 1000*np.linalg.norm(states[:, 6:8]-target, axis=1)
    np.testing.assert_allclose(error[1:], 1000*a['distance'])
    # Display settling is independent of the full-rollout success test.
    unsettled = np.flatnonzero((a['distance'] > .01) | (np.linalg.norm(a['y'][:, 4:6], axis=1) > .01))
    settled = int(unsettled[-1]+2)
    stop = min(320, int(np.ceil((settled+20)/10)*10))
    chart = a['chart']+1
    changes = np.r_[0, np.flatnonzero(np.diff(chart))+1]
    terminal = np.linalg.norm(a['ref'][:, 6:8]-target, axis=1) < 1e-10
    terminal_start = int(np.flatnonzero(terminal)[0])
    assert np.all(terminal[terminal_start:])

    counts = atlas['transition_counts']
    valid = (counts >= summary['edge_count_threshold']) & ~np.eye(len(counts), dtype=bool)
    rows, cols = np.where(valid)
    weights = atlas['center_distance'][rows, cols]/np.sqrt(counts[rows, cols])
    dist, pred = dijkstra(csr_matrix((weights, (rows, cols)), shape=counts.shape), directed=True, return_predecessors=True)
    routes = []
    for t in range(320):
        src, dst = int(a['chart'][t]), int(a['goal_chart'][t])
        route = [dst]
        while route[-1] != src:
            parent = int(pred[src, route[-1]])
            assert parent >= 0 and len(route) <= 48
            route.append(parent)
        route.reverse()
        routes.append(route)
        np.testing.assert_allclose(dist[src, dst], a['goal_route_cost'][t])
        assert len(route) == a['path_length'][t]
        if not terminal[t]:
            assert route[1] == a['next_chart'][t]
            np.testing.assert_allclose(a['ref'][t, 6:8], atlas['medoid_states'][route[1], 6:8])
    initial_route = routes[0]
    font = font_manager.findfont('Times New Roman', fallback_to_default=False)
    plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman'],
        'mathtext.fontset':'stix','font.size':9,'axes.labelsize':9,'axes.titlesize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.edgecolor':'#616161','axes.linewidth':.6,'text.color':INK,
        'axes.labelcolor':INK,'pdf.fonttype':42,'ps.fonttype':42,
        'savefig.facecolor':'white','lines.solid_capstyle':'round','lines.solid_joinstyle':'round'})

    xy = states[:stop+1, 6:8]*100
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    curve = CubicSpline(np.arange(stop+1), xy, bc_type='natural')
    curve_points = curve(np.linspace(0, stop, 16*stop+1))
    if '--combined-only' in sys.argv:
        combined_figure(a, atlas, states, target, chart, error, stop, terminal_start,
            initial_route, curve, curve_points, arc)
        print(DEST/'reacher_waypoint_combined.pdf')
        return
    # Same 6 x 1.5-inch canvas and typography as the direct-tracking figure.
    fig = plt.figure(figsize=(6, 1.5))
    bottom, height = .30, .82
    def panel(left, width):
        ax = fig.add_axes([left/6, bottom/1.5, width/6, height/1.5])
        style(ax)
        return ax
    motion = panel(.56, 1.79)
    selection = panel(2.81, 1.11)
    err = panel(4.47, 1.43)
    snapshots = np.unique(np.searchsorted(arc, np.linspace(0, arc[-1], 9)[1:-1]))
    for t in snapshots:
        motion.plot(*arm(states[t]).T, color=ORANGE, alpha=.32, lw=.8)
    motion.plot(*curve_points.T, color=ORANGE, lw=1.2, zorder=4)
    for fraction in [.22, .62]:
        t = np.searchsorted(arc, fraction*arc[-1])
        motion.annotate('', xy=curve(t+.3), xytext=curve(t-.3),
            arrowprops={'arrowstyle':'-|>','color':ORANGE,'lw':.8,'mutation_scale':6})
    for state, color, ls in [(states[0], BLUE, '--'), (states[stop], INK, '-')]:
        motion.plot(*arm(state).T, color=color, ls=ls, lw=1.2, marker='o', ms=2.5,
            markerfacecolor='white', markeredgewidth=.7, zorder=5)
    motion.scatter(*target*100, marker='x', s=20, color=RED, lw=1, zorder=7)
    motion.scatter(0,0,s=10,color=INK,zorder=8)
    xrange = 18*1.79/height
    motion.set(xlim=(-11, -11+xrange), ylim=(-1,17), xticks=[0,10,20], yticks=[0,10], ylabel=r'$p_{\mathrm{f}}^y$ (cm)')
    motion.set_aspect('equal', adjustable='box')
    selection.stairs(chart[:stop], np.arange(stop+1), baseline=None, color=ORANGE, lw=1.2)
    selection.set(xlim=(0,stop), ylim=(0,54), xticks=[0,40,stop], yticks=[1,24,48], ylabel='Chart ID')
    # Annotate short stages with alternating offsets, preserving their true IDs.
    offsets = {43:(1,-9,'left'),16:(0,-9,'center'),44:(0,3,'center'),5:(0,-1,'center'),26:(0,3,'center'),48:(0,3,'center'),30:(0,3,'center')}
    for lo, hi in zip(changes, np.r_[changes[1:],stop]):
        if lo >= stop: break
        c = int(chart[lo]); dx, dy, ha = offsets[c]
        selection.text((lo+min(hi,stop))/2+dx,c+dy,str(c),fontsize=7,ha=ha,va='bottom')
    err.semilogy(np.arange(stop+1), error[:stop+1], color=ORANGE, lw=1.2)
    err.axhline(10,color='#8e969c',lw=.7,ls=(0,(4,3)))
    err.text(stop-2,12,'10 mm',fontsize=8,ha='right',color='#777777')
    err.scatter(stop,error[stop],s=10,color=ORANGE,clip_on=False)
    err.annotate(f'{error[stop]:.2f} mm',xy=(stop,error[stop]),xytext=(-4,-13),textcoords='offset points',ha='right',fontsize=8)
    err.set(xlim=(0,stop),ylim=(.1,300),xticks=[0,40,stop],yticks=[.1,1,10,100],ylabel='Error (mm)')
    err.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f'{v:g}'))
    err.yaxis.set_minor_locator(ticker.NullLocator())
    handles = [Line2D([],[],color=ORANGE,lw=1.2,label='Path (interp.)'),
        Line2D([],[],color=BLUE,lw=1.2,ls='--',label='Initial pose'),
        Line2D([],[],color=INK,lw=1.2,label='Final pose'),
        Line2D([],[],color=ORANGE,lw=1.2,alpha=.32,label='Intermediate'),
        Line2D([],[],color=RED,marker='x',ms=3.5,ls='',label='Target')]
    fig.legend(handles=handles,loc='upper center',frameon=False,fontsize=8,ncol=5,
        handlelength=2.2,handletextpad=.45,columnspacing=1.2,bbox_to_anchor=(.52,.98),borderaxespad=0)
    for left, center, title, xlabel in [(.56,1.455,'(a) Fingertip trajectory',r'$p_{\mathrm{f}}^x$ (cm)'),
        (2.81,3.365,'(b) Chart selection','Control step'),(4.47,5.185,'(c) Tracking error','Control step')]:
        fig.text(left/6,1.19/1.5,title,fontsize=9)
        fig.text(center/6,.03/1.5,xlabel,ha='center',fontsize=9)
    save(fig,'reacher_waypoint_three_panel')

    # Graph geometry is a projection through representative fingertip positions,
    # not an assertion that straight graph edges are feasible physical motion.
    fig = plt.figure(figsize=(6,2.65))
    graph = fig.add_axes([.085,.19,.40,.64])
    timeline = fig.add_axes([.615,.19,.36,.64])
    for ax in [graph,timeline]: style(ax)
    points = atlas['medoid_states'][:,6:8]*100
    graph.scatter(*points.T,s=10,color='#c9ced2',alpha=.65,zorder=2)
    graph.plot(*curve_points.T,color=ORANGE,lw=1.2,alpha=.85,zorder=3)
    label_offsets = {43:(7,-9),16:(8,0),44:(7,6),5:(0,9),26:(-10,7),48:(-12,0),30:(5,-12)}
    for u,v in zip(initial_route[:-1],initial_route[1:]):
        graph.annotate('',xy=points[v],xytext=points[u],arrowprops={'arrowstyle':'-|>',
            'color':BLUE,'lw':1,'linestyle':(0,(3,2)),'shrinkA':4,'shrinkB':4,'mutation_scale':8},zorder=4)
    for k in initial_route:
        graph.scatter(*points[k],s=24,facecolor='white',edgecolor=BLUE,lw=1,zorder=5)
        graph.annotate(str(k+1),points[k],xytext=label_offsets[k+1],textcoords='offset points',
            ha='center',va='center',fontsize=8,color=BLUE,zorder=7,
            bbox={'facecolor':'white','edgecolor':'none','pad':.25,'alpha':.85})
    graph.scatter(*target*100,marker='x',s=28,lw=1.1,color=RED,zorder=6)
    graph.set(xlim=(-22,24),ylim=(-21,21),xticks=[-20,0,20],yticks=[-20,0,20],
        xlabel=r'$p_{\mathrm{f}}^x$ (cm)',ylabel=r'$p_{\mathrm{f}}^y$ (cm)')
    graph.set_aspect('equal',adjustable='box')
    graph.set_title('(a) Initial graph route',loc='left',pad=8)
    timeline.axvspan(terminal_start,40,color='#f3f4f5',zorder=0)
    timeline.stairs(chart[:40],np.arange(41),baseline=None,color=ORANGE,lw=1.4,zorder=4)
    timeline.stairs(a['next_chart'][:terminal_start]+1,np.arange(terminal_start+1),baseline=None,color=BLUE,ls='--',lw=1.1,zorder=5)
    timeline.axvline(terminal_start,color='#8e969c',ls=':',lw=.8)
    timeline.text(32,53,'Terminal reference',ha='center',fontsize=8)
    timeline.text(32,49,'active',ha='center',fontsize=8)
    timeline.set(xlim=(0,40),ylim=(0,58),xticks=[0,10,24,40],yticks=[1,16,30,48],xlabel='Control step',ylabel='Chart ID')
    timeline.set_title('(b) Online reference updates',loc='left',pad=8)
    timeline.legend(handles=[Line2D([],[],color=ORANGE,lw=1.4,label='Current chart'),
        Line2D([],[],color=BLUE,lw=1.1,ls='--',label='Waypoint chart')],loc='lower right',frameon=True,
        facecolor='white',edgecolor='none',framealpha=.92,fontsize=8,handlelength=2.4)
    fig.legend(handles=[Line2D([],[],color=ORANGE,lw=1.2,label='Fingertip path (interp.)'),
        Line2D([],[],color=BLUE,lw=1,ls='--',marker='o',mfc='white',ms=3,label='Directed graph route'),
        Line2D([],[],color=RED,marker='x',ls='',ms=4,label='Target')],
        loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,fontsize=8,handlelength=2.2)
    fig.text(.52,.025,'Graph nodes are shown at their representative fingertip positions.',ha='center',fontsize=8,color='#616161')
    save(fig,'reacher_graph_reference_generation')
    metadata = {'seed':573,'source':str(SOURCE),'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'atlas':str(ATLAS),'font':font,'display_steps':stop,'settling_step':settled,
        'terminal_reference_first_step':terminal_start,'initial_route_one_based':[int(k)+1 for k in initial_route],
        'chart_changes':[[int(t),int(chart[t])] for t in changes],
        'success_30':summary['success_30'],'success_100':summary['success_100'],
        'final_100_mean_error_mm':summary['mean_last_100']*1000,
        'max_replay_error':float(np.max(np.abs(np.array(replay)-a['y']))),
        'all_routes_and_active_waypoint_positions_verified':True,
        'interpolation':'Natural cubic spline through recorded output positions, visualization only',
        'graph_projection':'Representative fingertip positions; graph edges are not physical trajectories'}
    (DEST/'plot_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
