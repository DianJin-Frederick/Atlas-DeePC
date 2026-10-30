"""Five-panel F1TENTH results figure, with editable Illustrator export."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import copy
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_f1tenth_rollout_illustrator import editable_svg, EXPORT_JSX

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/f1tenth_combined_results'
OUT.mkdir(parents=True,exist_ok=True)
base=OUT/'f1tenth_combined_results'
source=ROOT/'experiments/f1tenth/paper_nsr_20261027'
folder=source/'sigma_0p001/atlas_k12_d40/seed542'
summary=json.loads((folder/'summary.json').read_text())
replay=json.loads((folder/'collision_replay.json').read_text())
assert summary['verified_success'] and replay['collision_free']
trace=np.load(folder/'rollout.npz')
xy=np.vstack([trace['X'][:,:2],trace['final_state'][:2]])
err=100*trace['cross_track_error']
rmse=float(np.sqrt(np.mean(err**2)))
assert np.isclose(rmse,summary['rmse_cm'])
rows=json.loads((source/'report.json').read_text())
plt.rcParams.update({
    'font.family':'Times New Roman','font.size':8.4,'axes.titlesize':9,
    'axes.labelsize':8.4,'mathtext.fontset':'cm','svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,
    'axes.edgecolor':'#888888','axes.linewidth':.6,
    'xtick.major.size':3,'ytick.major.size':3,
    'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.pad':3,'ytick.major.pad':3,
})
orange,blue='#e68122','#005099'
fig=plt.figure(figsize=(7.5,5.1),facecolor='white')
lap=fig.add_axes([.075,.375,.405,.445])
error=fig.add_axes([.075,.11,.405,.16])
axes=[fig.add_axes([.625,y,.355,.21]) for y in [.66,.385,.11]]
for ax in [lap,error,*axes]:
    ax.grid(color='#e2e6eb',lw=.5);ax.set_axisbelow(True)

# Left column: map plus a shallow error trace, preserving equal map scales.
bitmap=np.flipud(np.asarray(Image.open(ROOT/'data/f1tenth/maps/example_map.png').convert('L')))
ox,oy,res=-78.21853769831466,-44.37590462453829,.0625
lap.contour(ox+(np.arange(bitmap.shape[1])+.5)*res,
            oy+(np.arange(bitmap.shape[0])+.5)*res,bitmap,
            levels=[128],colors=['#c3c8cc'],linewidths=.65,zorder=1)
waypoints=np.loadtxt(ROOT/'data/f1tenth/example_waypoints.csv',delimiter=';',skiprows=3)
actual,=lap.plot(*xy.T,color=orange,lw=1.7,label='Closed loop',zorder=3)
reference,=lap.plot(waypoints[:,1],waypoints[:,2],color=blue,lw=.75,
                   ls=(0,(4,3)),label='Raceline',zorder=4)
start=lap.scatter(*xy[0],marker='x',s=25,color='#be382b',linewidths=1.3,zorder=5)
lap.set(xlim=(-55,4),ylim=(-18,29),xticks=[-50,-40,-30,-20,-10,0],yticks=[-10,0,10,20])
lap.set_aspect('equal',adjustable='box')
lap.set_title('(a) Closed-loop lap',loc='left',pad=6)
lap.set_xlabel(r'$p^x$ (m)',labelpad=3).set_gid('math_xlabel')
lap.set_ylabel(r'$p^y$ (m)',labelpad=3).set_gid('math_ylabel')
error.axhspan(-rmse,rmse,color=blue,alpha=.10,lw=0)
error.axhline(0,color='#a9a9a9',lw=.5)
error.plot(np.arange(len(err)),err,color=orange,lw=.85)
error.set_title('(b) Cross-track error',loc='left',pad=6)
error.set_ylabel(r'$e_t^\perp$ (cm)',labelpad=3).set_gid('math_error')
error.set(xlim=(0,len(err)-1),ylim=(-10.5,5),xticks=[0,50,100,150,200],
          yticks=[-10,-5,0,5],xlabel='Time step')
error.text(.99,1.08,f'RMSE {rmse:.2f} cm',transform=error.transAxes,
           ha='right',va='bottom',fontsize=8,color=blue)
fig.text(.2775,.985,'Atlas-DeePC: '+r'$\sigma=0.001$'+'\n'+r'$K=12$, $d=40$; seed 542',
         ha='center',va='top',fontsize=8.5,linespacing=1.5).set_gid('math_case')
fig.legend([actual,reference,start],['Closed loop','Raceline','Start'],
           loc='upper center',bbox_to_anchor=(.2775,.919),ncol=3,frameon=False,
           handlelength=1.8,columnspacing=1.1,fontsize=8.0)

# Right column: three compact panels sharing the noise-level axis.
levels=sorted({r['sigma'] for r in rows});x=np.arange(len(levels))
comparison_palette=['#365C91','#CC6647','#268879','#8268A6']
styles=[('Atlas-DeePC',None,comparison_palette[0],'o','-'),
        ('Contextual-DPC (30)',30,comparison_palette[1],'s','--'),
        ('Contextual-DPC (60)',60,comparison_palette[2],'^','-.'),
        ('Contextual-DPC (90)',90,comparison_palette[3],'D',':')]
for name,columns,color,marker,linestyle in styles:
    selected=[r for r in rows if (r['method'].startswith('atlas') if columns is None
                                else r['config'].get('columns')==columns)]
    lookup={r['sigma']:r for r in selected};ordered=[lookup[s] for s in levels]
    values=[[100*r['collision_free_laps']/r['evaluated'] for r in ordered],
            [np.nan if r['median_rmse_cm'] is None else r['median_rmse_cm'] for r in ordered],
            [r['timing']['median_ms'] for r in ordered]]
    for ax,y in zip(axes,values):
        ax.plot(x,y,label=name,color=color,marker=marker,ls=linestyle,lw=1.3,
                markersize=4.5,markerfacecolor='white',markeredgewidth=1,clip_on=False)
for ax,title,ylabel in zip(axes,
        ['(c) Collision-free lap success','(d) Tracking accuracy','(e) Online computation'],
        ['Success rate (%)','Median RMSE (cm)','Median time (ms/step)']):
    ax.set_title(title,loc='left',pad=6)
    ax.set_ylabel(ylabel,labelpad=5)
    ax.set_xticks(x,[f'{100*s:g}%' for s in levels]);ax.set_xlim(-.2,len(levels)-.8)
for ax in axes[:2]:ax.tick_params(labelbottom=False)
axes[2].set_xlabel(r'Noise level $\sigma$',labelpad=4).set_gid('math_noise')
axes[0].set(ylim=(-4,105),yticks=[0,25,50,75,100])
axes[1].set(ylim=(0,12),yticks=[0,4,8,12])
axes[2].set(ylim=(0,16),yticks=[0,4,8,12,16])
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.8025,.985),
           ncol=2,frameon=False,handlelength=1.8,columnspacing=.7,handletextpad=.45,fontsize=7.8)
editable_svg(fig,base.with_suffix('.svg'))
fig.savefig(base.with_name(base.name+'_python_preview.png'),dpi=200)
plt.close(fig)

# Expand remaining marker references, preserving their per-instance styles.
tree=ET.parse(base.with_suffix('.svg'));root=tree.getroot()
definitions={e.get('id'):e for e in root.iter() if e.get('id')}
for parent in list(root.iter()):
    for child in list(parent):
        if child.tag.split('}')[-1]!='use':continue
        ref=child.get('{http://www.w3.org/1999/xlink}href','').lstrip('#')
        glyph=copy.deepcopy(definitions[ref]);glyph.attrib.pop('id',None)
        group=ET.Element('{http://www.w3.org/2000/svg}g')
        group.set('transform',child.get('transform','')+f' translate({child.get("x","0")},{child.get("y","0")})')
        if child.get('style'):group.set('style',child.get('style'))
        group.append(glyph);parent.insert(list(parent).index(child),group);parent.remove(child)
tree.write(base.with_suffix('.svg'),encoding='utf-8',xml_declaration=True)
base.with_suffix('.jsx').write_text(EXPORT_JSX.replace('__BASE__',json.dumps(str(base))))
base.with_suffix('.json').write_text(json.dumps({'rollout_source':str(folder),
    'comparison_source':str(source/'report.json'),'seed':542,'sigma':.001,'K':12,'d':40,
    'rmse_cm':rmse,'panels':['lap','cross-track error','success','RMSE','time'],
    'comparison_palette':dict(zip([s[0] for s in styles],comparison_palette))},indent=2))
print(base.with_suffix('.jsx'))
