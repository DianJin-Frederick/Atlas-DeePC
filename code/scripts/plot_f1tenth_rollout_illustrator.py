"""Plot a saved F1TENTH paper rollout and export editable Adobe Illustrator art.

No simulation or result files are modified. Default: the median-error successful
Atlas lap at 0.1% NSR. Run with a Matplotlib-enabled Python and --draw.
"""
import argparse
import io
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/f1tenth_rollout_nsr0p1'
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/atlas_f1tenth_plot_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image

EXPORT_JSX = r'''
(function(){
 var base=__BASE__, old=app.userInteractionLevel, doc=null;
 app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
 try {
  doc=app.open(new File(base+'.svg'));
  if(doc.textFrames.length<15)throw new Error('Editable text import failed');
  var ai=new IllustratorSaveOptions();ai.pdfCompatible=true;
  doc.saveAs(new File(base+'.ai'),ai);
  var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;
  png.horizontalScale=300;png.verticalScale=300;
  doc.exportFile(new File(base+'.png'),ExportType.PNG24,png);
  var pdf=new PDFSaveOptions();pdf.preserveEditability=false;
  doc.saveAs(new File(base+'.pdf'),pdf);
  var message='Editable text frames: '+doc.textFrames.length+'; vector paths: '+doc.pathItems.length;
  doc.close(SaveOptions.DONOTSAVECHANGES);doc=null;
  return message;
 } finally {
  if(doc)doc.close(SaveOptions.DONOTSAVECHANGES);
  app.userInteractionLevel=old;
 }
})();
'''


def editable_svg(fig, destination):
    """Live ordinary text, exact Computer Modern vector glyphs for math labels."""
    buffers=[]
    for mode in ('none','path'):
        with matplotlib.rc_context({'svg.fonttype':mode}):
            b=io.StringIO();fig.savefig(b,format='svg');buffers.append(b.getvalue())
    live,outlined=map(ET.fromstring,buffers)
    ids={e.get('id'):e for e in outlined.iter() if e.get('id')}
    import copy
    # Expand glyph references for reliable Illustrator SVG import.
    for parent in list(outlined.iter()):
        for child in list(parent):
            if child.tag.split('}')[-1]!='use':continue
            ref=child.get('{http://www.w3.org/1999/xlink}href','').lstrip('#')
            if ref not in ids:continue
            glyph=copy.deepcopy(ids[ref]);glyph.attrib.pop('id',None)
            group=ET.Element('{http://www.w3.org/2000/svg}g')
            group.set('transform',child.get('transform','')+f" translate({child.get('x','0')},{child.get('y','0')})")
            group.append(glyph);parent.insert(list(parent).index(child),group);parent.remove(child)
    for parent in live.iter():
        for child in list(parent):
            if child.get('id') in ('math_xlabel','math_ylabel','math_case','math_error','math_noise'):
                parent.insert(list(parent).index(child),copy.deepcopy(ids[child.get('id')]))
                parent.remove(child)
    ET.register_namespace('','http://www.w3.org/2000/svg')
    ET.register_namespace('xlink','http://www.w3.org/1999/xlink')
    ET.ElementTree(live).write(destination,encoding='utf-8',xml_declaration=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draw',action='store_true')
    parser.add_argument('--seed',type=int);args=parser.parse_args()
    source=ROOT/'experiments/f1tenth/paper_nsr_20261027/sigma_0p001/atlas_k12_d40'
    runs=[]
    for p in sorted(source.glob('seed*/summary.json')):
        s=json.loads(p.read_text())
        replay=json.loads((p.parent/'collision_replay.json').read_text())
        if s.get('verified_success') and replay.get('collision_free'):
            runs.append((s,p.parent))
    median=float(np.median([s['rmse_cm'] for s,_ in runs]))
    summary,folder=(next(v for v in runs if v[0]['seed']==args.seed) if args.seed else
                    min(runs,key=lambda v:(abs(v[0]['rmse_cm']-median),v[0]['seed'])))
    trace=np.load(folder/'rollout.npz');calls=json.loads((folder/'calls.json').read_text())
    err=trace['cross_track_error']*100; steps=np.arange(len(err))
    charts=np.array([v['chart'] for v in calls[:len(err)]])+1
    xy=np.vstack([trace['X'][:,:2],trace['final_state'][:2]])
    rmse=float(np.sqrt(np.mean(err**2)))
    assert np.isclose(rmse,summary['rmse_cm'])
    assert len(charts)==len(err)==len(trace['X'])
    K=summary['config']['clusters'];switches=int(np.count_nonzero(np.diff(charts)))
    waypoints=np.loadtxt(ROOT/'data/f1tenth/example_waypoints.csv',delimiter=';',skiprows=3)
    orange,blue='#e68122','#005099'
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
        'font.size':9,'axes.titlesize':10,'axes.labelsize':9.5,'mathtext.fontset':'cm',
        'axes.edgecolor':'#555555','axes.linewidth':.75,'xtick.major.size':3,
        'ytick.major.size':3,'xtick.major.width':.65,'ytick.major.width':.65,
        'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'atlas-f1tenth-rollout'})
    fig=plt.figure(figsize=(7.2,3.65),facecolor='white')
    lap=fig.add_axes([.08,.16,.45,.64])
    error=fig.add_axes([.65,.16,.33,.64])
    for ax in (lap,error):ax.grid(color='#e2e6eb',lw=.5);ax.set_axisbelow(True)
    # Contour actual occupancy-map boundaries, not an assumed-width raceline.
    bitmap=np.flipud(np.asarray(Image.open(ROOT/'data/f1tenth/maps/example_map.png').convert('L')))
    ox,oy=-78.21853769831466,-44.37590462453829;res=.0625
    cs=lap.contour(ox+(np.arange(bitmap.shape[1])+.5)*res,
                   oy+(np.arange(bitmap.shape[0])+.5)*res,bitmap,
                   levels=[128],colors=['#c3c8cc'],linewidths=.8,zorder=1)
    cs.set_gid('Track_boundary_from_occupancy_map')
    actual,=lap.plot(xy[:,0],xy[:,1],color=orange,lw=1.9,label='Closed loop',zorder=3)
    reference,=lap.plot(waypoints[:,1],waypoints[:,2],color=blue,lw=.8,
                       linestyle=(0,(4,3)),label='Raceline',zorder=4,alpha=.9)
    start=lap.scatter(*xy[0],marker='x',s=25,color='#be382b',linewidths=1.4,label='Start',zorder=5)
    lap.set(xlim=(-55,4),ylim=(-18,29),xticks=[-50,-40,-30,-20,-10,0],yticks=[-10,0,10,20])
    lap.set_aspect('equal',adjustable='box')
    lap_box=lap.get_position()
    error.set_position([.65,lap_box.y0,.33,lap_box.height])
    lap.set_title('(a) Closed-loop lap',loc='left',pad=5)
    lap.set_xlabel(r'$p^x$ (m)',labelpad=3).set_gid('math_xlabel')
    lap.set_ylabel(r'$p^y$ (m)',labelpad=3).set_gid('math_ylabel')
    error.axhspan(-rmse,rmse,color=blue,alpha=.10,lw=0)
    error.axhline(0,color='#a9a9a9',lw=.5)
    error.plot(steps,err,color=orange,lw=1.05)
    error.set_title('(b) Cross-track error',loc='left',pad=5)
    error.set_ylabel(r'$e_t^\perp$ (cm)',labelpad=3).set_gid('math_error')
    bound=max(3,float(np.ceil(np.max(abs(err))*1.15)))
    error.set_ylim(-bound,bound);error.yaxis.set_major_locator(MaxNLocator(nbins=4))
    error.text(.98,.95,f'RMSE {rmse:.2f} cm',transform=error.transAxes,
               ha='right',va='top',color=blue,fontsize=9,
               bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    error.set(xlim=(0,len(err)-1),xticks=[0,50,100,150,200],xlabel='Time step')
    case=(r'Atlas-DeePC: NSR $=0.1\%$ ($\sigma=0.001$), '
          rf'$K={K}$, $d={summary["config"]["rank"]}$; seed {summary["seed"]}')
    fig.text(.52,.98,case,ha='center',va='top',fontsize=10).set_gid('math_case')
    fig.legend([actual,reference,start],['Closed loop','Raceline','Start'],
               loc='upper center',bbox_to_anchor=(.52,.925),ncol=3,frameon=False,
               handlelength=2.3,columnspacing=2.5,fontsize=9)
    OUT.mkdir(parents=True,exist_ok=True);base=OUT/'f1tenth_rollout_nsr0p1'
    editable_svg(fig,base.with_suffix('.svg'))
    fig.savefig(base.with_name(base.name+'_python_preview.png'),dpi=250)
    plt.close(fig)
    meta=dict(source=str(folder),seed=summary['seed'],NSR_percent=100*summary['noise_sigma'],
              K=K,d=summary['config']['rank'],steps=len(err),control_interval_s=.1,
              rmse_cm=rmse,group_median_rmse_cm=median,selection='Closest successful-lap RMSE to group median',
              chart_ID_convention='Stored zero-based IDs plus one',chart_ids=np.unique(charts).tolist(),
              chart_switches=switches,collision_verified=True,
              note='All chart past images span the measured history space; residual ties select chart 1.',
              warmup='Excluded from the displayed trajectory, step count and error; included in saved collision replay.')
    base.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    caption=(r'\caption{Atlas-DeePC rollout on F1TENTH at $0.1\%$ NSR (seed '+str(summary['seed'])+
        r'), with $K=12$ and $d=40$. (a) Collision-verified closed-loop lap and reference raceline; '
        r'the marker denotes the start of Atlas-DeePC control after initialization. '
        r'(b) Cross-track error evaluated using the true vehicle position; the shaded band marks '
        r'$\pm$ the rollout RMSE. The control interval is '
        r'$\Delta t=0.1\,\mathrm{s}$. Projection residuals tie in this configuration, '
        r'so the smallest-index chart is selected throughout.}'+'\n')
    base.with_name(base.name+'_caption.tex').write_text(caption)
    jsx=base.with_suffix('.jsx');jsx.write_text(EXPORT_JSX.replace('__BASE__',json.dumps(str(base))))
    print(json.dumps(meta,indent=2))
    if args.draw:
        command=f'tell application "/Applications/Adobe Illustrator 2024/Adobe Illustrator.app" to do javascript file (POSIX file "{jsx}")'
        result=subprocess.run(['osascript','-e',command],capture_output=True,text=True)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        print(result.stdout)
        base.with_name(base.name+'_illustrator_report.txt').write_text(result.stdout)


if __name__=='__main__':main()
