"""Paper plots from the saved Clarabel, lambda_g=.05 terminal-only rollout.

Replay recorded actions only to recover and verify the pre-control pose.
No optimization, experiment rerun, or alteration of the saved data is performed.
"""
from pathlib import Path
import os
import json
import hashlib
import argparse

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/atlas_deepc_paper_mpl")
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/routed_deepc_iid/clarabel_lamg005_waypoints_v1"
ORANGE = "#df7b20"
INK = "#273441"
BLUE = "#45677e"
RED = "#b94336"


def output(obs):
    return np.r_[obs[:4], obs[6:8], obs[8:10] + obs[4:6],
                 obs[0]*obs[1]-obs[2]*obs[3], obs[2]*obs[1]+obs[0]*obs[3]]


def arm(y):
    # Units in cm; measured fingertip, elbow from the 10-cm first link.
    return np.vstack([np.zeros(2), 10*np.array([y[0], y[2]]), 100*y[6:8]])


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#eeeeee", linewidth=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=.6)


def combined_figure(states, target, chart, error_mm, destination, failure=False, trajectory_only=False):
    """Render at final print size, with exactly aligned axes, not resized images."""
    width, height = 6.0, 1.50
    bottom, panel_height = .30, .82
    fig = plt.figure(figsize=(width,height))
    def panel(left, panel_width):
        ax = fig.add_axes([left/width,bottom/height,panel_width/width,panel_height/height])
        style(ax)
        return ax
    motion = panel(.56,1.79)
    selection = panel(2.81,1.11 if failure else .84)
    tail = None if failure else panel(3.76,.16)
    error = panel(4.47,1.43)
    steps = len(states)-1
    xy = 100*states[:,6:8]
    distance = np.r_[0,np.cumsum(np.linalg.norm(np.diff(xy,axis=0),axis=1))]
    samples = np.unique(np.searchsorted(distance,np.linspace(0,distance[-1],10)[1:-1]))
    curve = CubicSpline(np.arange(steps+1),xy,bc_type="natural")
    for i in samples:
        motion.plot(*arm(states[i]).T,color=ORANGE,alpha=.32,lw=.8,zorder=1)
    motion.plot(*curve(np.linspace(0,steps,16*steps+1)).T,color=ORANGE,lw=1.2,zorder=3)
    for fraction in (.24,.63):
        i = np.searchsorted(distance,fraction*distance[-1])
        motion.annotate("",xy=curve(min(i+.3,steps)),xytext=curve(max(i-.3,0)),
                        arrowprops={"arrowstyle":"-|>","color":ORANGE,"lw":.8,"mutation_scale":6})
    for state,color,ls in ((states[0],BLUE,"--"),(states[-1],INK,"-")):
        motion.plot(*arm(state).T,color=color,ls=ls,lw=1.2,marker="o",ms=2.5,
                    markerfacecolor="white",markeredgewidth=.7,zorder=5)
    motion.scatter(*target*100,marker="x",s=20,color=RED,lw=1,zorder=7)
    motion.scatter(0,0,s=10,color=INK,zorder=8)
    xlim=(-4,28)
    # Match the data-range ratio to the physical axes box to preserve equal scale
    # without allowing Matplotlib to move or shrink this panel independently.
    ymax=3.5
    yrange=(xlim[1]-xlim[0])*panel_height/1.79
    motion.set(xlim=xlim,ylim=(ymax-yrange,ymax),xticks=[0,10,20],yticks=[-10,-5,0],
               ylabel=r"$p_{\mathrm{f}}^y$ (cm)")
    motion.set_aspect("equal",adjustable="box")
    handles=[Line2D([],[],color=ORANGE,lw=1.2,label="Path (interp.)"),
             Line2D([],[],color=BLUE,lw=1.2,ls="--",label="Initial pose"),
             Line2D([],[],color=INK,lw=1.2,label="Final pose"),
             Line2D([],[],color=ORANGE,lw=1.2,alpha=.32,label="Intermediate"),
             Line2D([],[],color=RED,marker="x",ms=3.5,ls="",label="Target")]
    if trajectory_only:
        for other in (selection,tail,error):
            if other is not None:
                other.remove()
        width,height=1.50,.95
        fig.set_size_inches(width,height)
        plot_left, plot_width, plot_height = .43, .99, .55
        motion.set_position([plot_left/width,.34/height,plot_width/width,plot_height/height])
        xlim=(-4,21.8)
        yrange=(xlim[1]-xlim[0])*plot_height/plot_width
        motion.set_xlim(xlim)
        motion.set_ylim(-1-yrange/2,-1+yrange/2)
        motion.set_yticks([-5,0])
        fig.text((plot_left+plot_width/2)/width,.035/height,r"$p_{\mathrm{f}}^x$ (cm)",fontsize=9,ha="center")
        stem="reacher_failure_trajectory" if failure else "reacher_success_trajectory"
        for extension in ("pdf","png"):
            fig.savefig(destination/f"{stem}.{extension}",dpi=300)
        plt.close(fig)
        print(destination/f"{stem}.pdf")
        return
    legend=fig.legend(handles=handles,loc="upper center",frameon=False,fontsize=8,ncol=5,
                      handlelength=2.2,handletextpad=.45,labelspacing=0,borderpad=0,
                      columnspacing=1.2,bbox_to_anchor=(.52,1.47/height),borderaxespad=0)
    if not failure:
        assert np.all(chart[40:steps]==chart[40])
    for ax in ([selection] if failure else [selection,tail]):
        ax.stairs(chart[:steps],np.arange(steps+1),baseline=None,color=ORANGE,lw=1.2)
        ax.set_ylim(0,53)
    selection.set(yticks=[1,24,48],ylabel="Chart ID")
    if failure:
        selection.set(xlim=(0,steps),xticks=[0,160,320])
    else:
        selection.set(xlim=(0,41),xticks=[0,20,40])
        tail.set(xlim=(steps-2,steps),xticks=[steps],yticks=[1,24,48])
        tail.spines["left"].set_visible(False)
        tail.tick_params(axis="y",left=False,labelleft=False)
        center=(3.65+3.76)/2/width
        for dx in (-.024,0,.024):
            for ypos,color in ((bottom,"#616161"),(bottom+panel_height*chart[40]/53,ORANGE)):
                fig.add_artist(Line2D([center+dx/width],[ypos/height],transform=fig.transFigure,
                                     marker="o",ms=.85,markeredgewidth=0,color=color,ls=""))
    change=np.r_[0,np.flatnonzero(np.diff(chart))+1]
    for start,end,c in zip(change,np.r_[change[1:],steps],chart[change]):
        if start<(steps if failure else 40):
            xpos=(start+min(end,steps if failure else 40))/2
            selection.text(xpos if c!=42 else xpos-3,c+(3 if c!=42 else -7),str(c),
                           ha="center" if c!=42 else "right",fontsize=8)
    error.semilogy(np.arange(steps+1),error_mm,color=ORANGE,lw=1.2)
    error.axhline(10,color="#8e969c",lw=.7,ls=(0,(4,3)))
    error.text(steps-2,12,"10 mm",fontsize=8,ha="right",color="#777777")
    error_label=f"{error_mm[-1]:.1f} mm" if failure else f"{error_mm[-1]:.3f} mm"
    error.annotate(error_label,xy=(steps,error_mm[-1]),xytext=(-3,-15 if failure else 9),
                   textcoords="offset points",ha="right",fontsize=8)
    error.scatter(steps,error_mm[-1],s=10,color=ORANGE,clip_on=False)
    error.set(xlim=(0,steps),ylim=(.1,300),xticks=[0,160,320] if failure else [0,50,100],yticks=[.1,1,10,100],
              ylabel="Error (mm)")
    error.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f"{v:g}"))
    error.yaxis.set_minor_locator(ticker.NullLocator())
    for left,center,title,xlabel in (
        (.56,.56+1.79/2,"(a) Fingertip trajectory",r"$p_{\mathrm{f}}^x$ (cm)"),
        (2.81,(2.81+3.92)/2,"(b) Chart selection","Control step"),
        (4.47,4.47+1.43/2,"(c) Tracking error","Control step")):
        fig.text(left/width,1.19/height,title,fontsize=9,ha="left",va="baseline")
        fig.text(center/width,.03/height,xlabel,fontsize=9,ha="center",va="baseline")
    fig.canvas.draw()
    boxes=[ax.get_position() for ax in (motion,selection,tail,error) if ax is not None]
    np.testing.assert_allclose([b.y0 for b in boxes],bottom/height,atol=1e-10)
    np.testing.assert_allclose([b.y1 for b in boxes],(bottom+panel_height)/height,atol=1e-10)
    legend_box=legend.get_window_extent(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    assert legend_box.y0 > boxes[0].y1, "Legend must clear the trajectory axes."
    stem="reacher_failure_three_panel" if failure else "reacher_success_three_panel"
    for extension in ("pdf","png"):
        fig.savefig(destination/f"{stem}.{extension}",dpi=300)
    plt.close(fig)
    print(destination/f"{stem}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=222)
    parser.add_argument("--trajectory-only", action="store_true")
    parser.add_argument("--chart-only", action="store_true")
    parser.add_argument("--combined-only", action="store_true")
    parser.add_argument("--failure-example", action="store_true")
    args = parser.parse_args()
    if args.failure_example and not args.combined_only:
        parser.error("--failure-example requires --combined-only")
    source = RESULTS / f"terminal/seed{args.seed}_control_results.npz"
    with np.load(source, allow_pickle=True) as f:
        a = {k: f[k] for k in f.files}
    summary = json.loads(a["summary"].item())
    manifest = json.loads((RESULTS / "manifest.json").read_text())
    assert manifest["solver"] == "CLARABEL"
    assert summary["lam_g"] == .05 and summary["reference_mode"] == "terminal"
    assert not summary["graph_waypoints_enabled"]
    if args.failure_example:
        assert not summary["success_30"] and not summary["success_100"]
    else:
        assert summary["success_30"] and summary["success_100"]
    assert np.all(a["path_length"] == 1)
    y = a["y"]
    steps = len(y)
    target = a["final_goal"].reshape(-1)[6:8]
    np.testing.assert_allclose(a["ref"][:, 6:8], np.tile(target, (steps, 1)))
    np.testing.assert_allclose(a["distance"], np.linalg.norm(y[:, 6:8]-target, axis=1))
    env = gym.make("Reacher-v5", max_episode_steps=steps+7)
    obs, _ = env.reset(seed=args.seed)
    np.testing.assert_allclose(obs[4:6], target)
    for _ in range(2):
        obs, *_ = env.step(np.zeros(2))
    initial = output(obs)
    replay = []
    for action in a["u"]:
        obs, *_ = env.step(action)
        replay.append(output(obs))
    env.close()
    replay_error = float(np.max(np.abs(np.asarray(replay)-y)))
    np.testing.assert_allclose(replay, y, rtol=1e-8, atol=1e-9)
    states = np.vstack([initial, y])
    # Display-only settling rule, not a change to the experiment's success test.
    # Verify both conditions through the entire recorded run, then keep a tail.
    unsettled = np.flatnonzero((1000*a["distance"] > 1) |
                              (np.linalg.norm(y[:, 4:6], axis=1) > .01))
    settled_step = int(unsettled[-1]+2) if len(unsettled) else 1
    display_steps = min(steps, int(np.ceil((settled_step+20)/10)*10))
    if args.failure_example:
        display_steps=steps  # Do not hide a failure by truncating the evaluation.
    states = states[:display_steps+1]
    xy = 100*states[:, 6:8]
    error_mm = 1000*np.linalg.norm(states[:, 6:8]-target, axis=1)
    chart = a["chart"].astype(int)+1
    change = np.r_[0, np.flatnonzero(np.diff(chart))+1]
    font_path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.edgecolor": "#616161", "axes.linewidth": .6,
        "text.color": INK, "axes.labelcolor": INK, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white",
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round"})
    destination = ROOT / "paper/figures" / f"reacher_no_waypoint_seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    if args.combined_only:
        combined_figure(states,target,chart,error_mm,destination,failure=args.failure_example,
                        trajectory_only=args.trajectory_only)
        if args.failure_example:
            provenance={"seed":args.seed,"source":str(source),
                "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                "selection_reason":"Failed rollout nearest median failed final-100 mean error among the 1000 terminal-only runs",
                "solver":manifest["solver"],"lambda_g":summary["lam_g"],
                "reference_mode":summary["reference_mode"],"graph_waypoints_enabled":False,
                "terminal_reference_selection_uses_graph":True,
                "success_30":summary["success_30"],"success_100":summary["success_100"],
                "mean_last_100_error_mm":summary["mean_last_100"]*1000,
                "final_error_mm":float(error_mm[-1]),"displayed_steps":display_steps,
                "chart_ids_one_based":chart[change].tolist(),
                "all_qps_optimal":bool(np.all(a["status"]=="optimal")),
                "max_replay_output_error":replay_error,
                "trajectory_rendering":"Natural cubic spline interpolating recorded positions, visualization only",
                "error_scale":None if args.trajectory_only else "logarithmic",
                "figure_inches":[1.50,.95] if args.trajectory_only else [6,1.5],
                "legend_included":not args.trajectory_only,
                "title_included":not args.trajectory_only,
                "chart_axis_break":False}
            metadata_name="failure_trajectory_metadata.json" if args.trajectory_only else "failure_plot_metadata.json"
            (destination/metadata_name).write_text(json.dumps(provenance,indent=2)+"\n")
            print(json.dumps(provenance,indent=2))
        return
    def save(fig, name):
        if args.chart_only and name != "selected_chart_ids":
            plt.close(fig)
            return
        if args.trajectory_only and name != "fingertip_trajectory":
            plt.close(fig)
            return
        # Keep the physical canvas width: tight cropping would alter the font
        # sizes when LaTeX scales the resulting PDF to the intended print width.
        fig.savefig(destination / f"{name}.pdf")
        fig.savefig(destination / f"{name}.png", dpi=300)
        plt.close(fig)

    # Motion: draw snapshots at equal traveled-distance intervals, not after
    # equal times, to avoid stacking nearly identical steady-state postures.
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    snapshot_indices = np.unique(np.searchsorted(cumulative, np.linspace(0, cumulative[-1], 10)[1:-1]))
    # Visualization only: interpolate through every recorded position in time.
    # Do not change the recorded states, snapshots, or error calculations.
    curve = CubicSpline(np.arange(display_steps+1), xy, axis=0, bc_type="natural")
    np.testing.assert_allclose(curve(np.arange(display_steps+1)), xy, atol=1e-12)
    curve_xy = curve(np.linspace(0, display_steps, display_steps*16+1))
    fig, ax = plt.subplots(figsize=(3.15, 2.65), layout="constrained")
    style(ax)
    for i in snapshot_indices:
        points = arm(states[i])
        ax.plot(*points.T, color=ORANGE, alpha=.32, lw=.9, zorder=1)
    ax.plot(*curve_xy.T, color=ORANGE, lw=1.35, zorder=3)
    # Direction arrows follow the displayed interpolant.
    for frac in [.24, .63]:
        i = int(np.searchsorted(cumulative, frac*cumulative[-1]))
        ax.annotate("", xy=curve(min(i+.3,display_steps)), xytext=curve(max(0,i-.3)),
                    arrowprops={"arrowstyle": "-|>", "color": ORANGE, "lw": .9,"mutation_scale":7}, zorder=4)
    for state, color, linestyle in [(initial, BLUE, "--"), (states[-1], INK, "-")]:
        points = arm(state)
        ax.plot(*points.T, color=color, linestyle=linestyle, lw=1.4,
                marker="o", ms=3, markerfacecolor="white", markeredgewidth=.8, zorder=5)
    ax.scatter(*target*100, marker="x", s=26, color=RED, linewidths=1.1, zorder=7)
    ax.scatter(0, 0, s=14, color=INK, zorder=8)
    all_points = np.concatenate([arm(s) for s in states])
    low, high = all_points.min(0)-1.5, all_points.max(0)+1.5
    low[1] -= 2.0  # Clear lower-right space for the compact in-panel legend.
    ax.set(xlim=(low[0],high[0]), ylim=(low[1],high[1]), aspect="equal",
           xlabel=r"$p_{\mathrm{f}}^x$ (cm)",
           ylabel=r"$p_{\mathrm{f}}^y$ (cm)")
    ax.set_title("Fingertip trajectory", loc="left", pad=8)
    handles = [Line2D([],[],color=ORANGE,lw=1.35,label="Interpolated path"),
               Line2D([],[],color=BLUE,lw=1.4,ls="--",label="Initial pose"),
               Line2D([],[],color=INK,lw=1.4,label="Final pose"),
               Line2D([],[],color=ORANGE,lw=1.4,alpha=.32,label="Intermediate poses"),
               Line2D([],[],color=RED,marker="x",ls="",ms=4,label="Target")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(.98,.005),
              frameon=False, handlelength=1.5, labelspacing=.2, fontsize=8,
              handletextpad=.5,borderpad=.1)
    save(fig,"fingertip_trajectory")

    # Compress only the constant tail, explicitly marking the broken time axis.
    assert display_steps > 42 and np.all(chart[40:display_steps] == chart[40]), \
        "A broken constant-tail axis is unsuitable for this rollout."
    fig, (ax, tail) = plt.subplots(1, 2, figsize=(2.05,1.95), sharey=True,
                                  gridspec_kw={"width_ratios":[5,1]})
    fig.subplots_adjust(left=.23,right=.94,bottom=.24,top=.81,wspace=.24)
    for panel in (ax,tail):
        style(panel)
        panel.stairs(chart[:display_steps], np.arange(display_steps+1), baseline=None,
                     color=ORANGE, lw=1.3)
        panel.set_ylim(0,53)
    ax.set(xlim=(0,41),ylabel="Chart ID",yticks=[1,24,48],xticks=[0,20,40])
    tail.set(xlim=(display_steps-2,display_steps),xticks=[display_steps])
    tail.spines["left"].set_visible(False)
    tail.tick_params(axis="y",left=False,labelleft=False)
    fig.text(.23,.89,"Chart selection",fontsize=9,ha="left")
    fig.text(.585,.035,"Control step",fontsize=9,ha="center")
    # One compact break centered in the gap avoids oversized marks at all
    # four endpoints. Orange dots show the unchanged, omitted plateau.
    left_box, right_box = ax.get_position(), tail.get_position()
    gap_center = (left_box.x1+right_box.x0)/2
    plateau_y = left_box.y0+left_box.height*chart[40]/53
    for offset in (-.012,0,.012):
        for dot_y,dot_color in ((plateau_y,ORANGE),(left_box.y0,"#616161")):
            fig.add_artist(Line2D([gap_center+offset],[dot_y],
                                 transform=fig.transFigure,marker="o",ms=.9,
                                 color=dot_color,linestyle="none",markeredgewidth=0))
    for start,end,c in zip(change,np.r_[change[1:],steps],chart[change]):
        if start < 40:
            label_x = (start+min(end,40))/2
            ax.text(label_x if c!=42 else label_x-3,c+(2.8 if c!=42 else -6),str(c),
                    ha="center" if c!=42 else "right",fontsize=8,color=INK)
    save(fig,"selected_chart_ids")

    fig, ax = plt.subplots(figsize=(3.15,1.95),layout="constrained")
    style(ax)
    ax.semilogy(np.arange(display_steps+1),error_mm,color=ORANGE,lw=1.3)
    ax.axhline(10,color="#8e969c",lw=.9,ls=(0,(4,3)))
    ax.text(display_steps-2,12,"10 mm",fontsize=8,color="#777777",ha="right")
    ax.scatter(display_steps,error_mm[-1],s=17,color=ORANGE,zorder=4,clip_on=False)
    ax.annotate(f"{error_mm[-1]:.3f} mm",xy=(display_steps,error_mm[-1]),
                xytext=(-7,10),textcoords="offset points",ha="right",fontsize=8,color=INK)
    ax.set(xlim=(0,display_steps),ylim=(.1,300),xlabel="Control step",ylabel="Fingertip error (mm)",
           xticks=np.arange(0,display_steps+1,20),yticks=[.1,1,10,100])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v,_:f"{v:g}"))
    ax.yaxis.set_minor_locator(ticker.NullLocator())
    ax.set_title("Target tracking error",loc="left",pad=8)
    save(fig,"fingertip_error")

    metadata={"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed":args.seed,"solver":manifest["solver"],"lambda_g":summary["lam_g"],
        "reference_mode":summary["reference_mode"],"graph_waypoints_enabled":False,
        "terminal_reference_selection_uses_graph":True,"strict_success":True,
        "font":font_path,"final100_mean_error_mm":summary["mean_last_100"]*1000,
        "all_qps_optimal":bool(np.all(a["status"]=="optimal")),
        "chart_ids_one_based":chart[change].tolist(),"chart_change_decision_steps":change.tolist(),
        "intermediate_configuration_steps":snapshot_indices.tolist(),
        "pose_step_convention":"0 is after two zero-input history steps and before the first optimized input; saved y rows are steps 1..320",
        "chart_step_convention":"stairs on [t,t+1) show chart used to compute input at decision t",
        "initial_configuration_source":"seeded simulator initialization, verified by replaying every recorded action",
        "max_replay_output_error":replay_error,"error_axis_scale":"logarithmic"}
    metadata.update({"display_control_steps":display_steps,
        "settled_from_step":settled_step,
        "display_settling_rule":"error <= 1 mm and joint-speed norm <= 0.01 rad/s for all remaining saved steps, plus >=20 steps of tail (rounded up to tens)",
        "display_final_error_mm":float(error_mm[-1]),
        "trajectory_rendering":"natural cubic spline in control-step time, interpolating every recorded fingertip position; visualization only, not measured intersample motion",
        "intermediate_configuration_opacity":.32,
        "print_layout":{"width_inches":3.15,"axis_and_title_font_pt":9,
                        "chart_panel_width_inches":2.05,
                        "tick_legend_annotation_font_pt":8,"template_width_verified":False,
                        "note":"Compact panel proportions based on user-provided Select-DPC screenshot. Insert at 3.15in to retain font sizes; actual template text width unverified."},
        "final_configuration_step":display_steps})
    metadata["chart_axis_break"]={"visible_intervals":[[0,41],[display_steps-2,display_steps]],
                                  "omitted_interval_has_no_chart_changes":True}
    (destination/"plot_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps(metadata,indent=2))
    print(destination)


if __name__=="__main__":
    main()
