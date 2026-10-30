"""Editable vector schematic of F1TENTH outputs and cross-track error."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Arc, PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import matplotlib.patheffects as pe
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/figures/f1tenth_tracking_schematic'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,
                     'mathtext.fontset':'cm','svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(8, 4.8))
fig.subplots_adjust(left=.025,right=.99,bottom=.025,top=.97)
ax.set(xlim=(-.7,9.6),ylim=(-.65,5.45),aspect='equal')
ax.axis('off')
fig.canvas.draw()
blue,orange,teal,gray = '#003f78','#dc7516','#006653','#7c858b'
orange_label = '#9a4200'
halo = [pe.withStroke(linewidth=2.5, foreground='white'), pe.Normal()]

def math(text,x,y,size=14,color='#222222',ha='center',va='center'):
    # Computer Modern paths ONLY for math; ordinary labels remain editable text.
    # At a 3.2-inch wrap width, this yields roughly 8--10 pt math labels.
    path=TextPath((0,0),text,size=size*1.55)
    b=path.get_extents()
    scale=(ax.get_xlim()[1]-ax.get_xlim()[0])/(ax.get_window_extent().width*72/fig.dpi)
    dx=-(b.x0+b.width/2) if ha=='center' else -b.x0
    dy=-(b.y0+b.height/2) if va=='center' else -b.y0
    transform=Affine2D().translate(dx,dy).scale(scale).translate(x,y)+ax.transData
    ax.add_patch(PathPatch(path,transform=transform,facecolor=color,edgecolor='none',
                          path_effects=halo,zorder=20))

def arrow(a,b,color=gray,lw=1.2,style='-|>',scale=12,z=8):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle=style,color=color,lw=lw,
                mutation_scale=scale,shrinkA=0,shrinkB=0),zorder=z)

# Track boundaries are schematic offsets from the illustrated raceline.
x=np.linspace(-.15,9.15,250)
y=1.5+.045*(x-3)**2
slope=.09*(x-3)
norm=np.stack([-slope,np.ones_like(x)],axis=1)/np.sqrt(1+slope**2)[:,None]
center=np.stack([x,y],axis=1)
upper=center+1.25*norm; lower=center-1.25*norm
ax.add_patch(Polygon(np.vstack([upper,lower[::-1]]),facecolor='#f5f6f7',edgecolor='none',zorder=0))
for edge in [upper,lower]:
    ax.plot(*edge.T,color='#a1a8ad',lw=1.3,zorder=1)
ax.plot(x,y,color=orange,lw=2,zorder=2)

# Waypoint, tangent, normal and an offset vehicle reference point.
q=np.array([4.6,1.5+.045*(4.6-3)**2])
psi_rl=np.arctan(.09*(q[0]-3))
tangent=np.array([np.cos(psi_rl),np.sin(psi_rl)])
normal=np.array([-tangent[1],tangent[0]])
p=q+.88*normal
psi=np.deg2rad(34)
rotation=np.array([[np.cos(psi),-np.sin(psi)],[np.sin(psi),np.cos(psi)]])

# Global position components, read from the same vehicle reference point.
arrow((0,0),(9.35,0),color='#41474b',lw=1.0)
arrow((0,0),(0,5.15),color='#41474b',lw=1.0)
math('$x$',9.35,-.23);math('$y$',-.23,5.1)
math('$0$',-.18,-.19,size=11)
ax.plot([p[0],p[0]],[0,p[1]],ls=(0,(3,3)),color='#a1acb6',lw=.9,zorder=2)
ax.plot([0,p[0]],[p[1],p[1]],ls=(0,(3,3)),color='#a1acb6',lw=.9,zorder=2)
math(r'$p_t^x$',p[0],-.29,color=blue)
math(r'$p_t^y$',-.32,p[1],color=blue)

# Car body, four tires and windshield; heading is its longitudinal axis.
def car_poly(coords,face,edge='none',lw=.7):
    ax.add_patch(Polygon(np.asarray(coords)@rotation.T+p,facecolor=face,edgecolor=edge,lw=lw,zorder=7))
for longitudinal in [-.36,.36]:
    for lateral in [-.24,.24]:
        car_poly([(longitudinal-.13,lateral-.045),(longitudinal+.13,lateral-.045),
                  (longitudinal+.13,lateral+.045),(longitudinal-.13,lateral+.045)],'#343f48')
car_poly([(-.55,-.21),(.42,-.21),(.57,-.13),(.57,.13),(.42,.21),(-.55,.21)],'#d6e7f4',blue,1.2)
car_poly([(.07,-.16),(.3,-.14),(.3,.14),(.07,.16)],'#7aa4c8',blue,.5)
ax.plot(*p,'o',ms=3.3,color=blue,zorder=15)

# Speed and heading outputs, measured relative to the global x direction.
direction=np.array([np.cos(psi),np.sin(psi)])
arrow(p+.59*direction,p+2*direction,color=blue,lw=1.8,scale=14)
math('$v_t$',*(p+1.6*direction+np.array([-.1,.26])),color=blue,size=16)
ax.plot([p[0]+.48,p[0]+1.65],[p[1],p[1]],ls=(0,(3,3)),color='#9aa6b1',lw=.9,zorder=5)
ax.add_patch(Arc(p,2.1,2.1,theta1=0,theta2=np.rad2deg(psi),color=blue,lw=1.0,zorder=8))
math(r'$\psi_t$',p[0]+1.58,p[1]+.40,color=blue)
math('$p_t$',p[0]-.65,p[1]+.53,color=blue,size=14)

# Cross-track error is the normal projection at the selected waypoint.
ax.plot(*q,'o',ms=4,color=orange,zorder=10)
ax.plot(*np.stack([q,p]).T,color=teal,lw=1.4,zorder=11)
dimension_offset=-.82*tangent
start=q+dimension_offset; end=p+dimension_offset
ax.plot(*np.stack([q,start-.12*tangent]).T,color=teal,lw=.7,zorder=4)
ax.plot(*np.stack([p,end-.12*tangent]).T,color=teal,lw=.7,zorder=4)
arrow(start,end,color=teal,lw=1.2,style='<->',scale=11,z=12)
math(r'$e_t^\perp$',*((start+end)/2-.5*tangent),color=teal,size=16)
corner=np.stack([q+.14*tangent,q+.14*tangent+.14*normal,q+.14*normal])
ax.plot(*corner.T,color=teal,lw=.8,zorder=12)
arrow(q+.2*tangent,q+2.15*tangent,color='#707b83',lw=1.0,scale=10,z=5)
ax.plot([q[0]+.3,q[0]+2.2],[q[1],q[1]],ls=(0,(3,3)),color='#9aa6b1',lw=.7,zorder=3)
ax.add_patch(Arc(q,3.4,3.4,theta1=0,theta2=np.rad2deg(psi_rl),color='#707b83',lw=.9,zorder=8))
math(r'$p_t^{\mathrm{rl}}$',q[0]+.15,q[1]-.42,color=orange_label,size=14)
math(r'$\psi_t^{\mathrm{rl}}$',q[0]+1.85,q[1]-.31,color='#303a42',size=14)

# Short explanatory callouts, in Times New Roman.
ax.text(1.1,4.65,'Vehicle outputs',color='#222222',fontsize=18)
math(r'$y_t=(p_t^x,p_t^y,v_t,\psi_t)$',3.15,4.12,size=15)
ax.annotate('Track boundary',xy=upper[215],xytext=(7.45,4.8),ha='center',
            fontsize=18,color='#343c42',path_effects=halo,
            arrowprops=dict(arrowstyle='-',color='#899298',lw=.8))
ax.annotate('Raceline',xy=(7.5,1.5+.045*4.5**2),xytext=(7.55,1.35),
            fontsize=18,color=orange_label,path_effects=halo,
            arrowprops=dict(arrowstyle='-',color=orange,lw=.8))
ax.text(.5,-.43,'Global position coordinates',fontsize=14,color='#41474b')
fig.savefig(OUT/'f1tenth_tracking_schematic.png',dpi=200,facecolor='white')
fig.savefig(OUT/'f1tenth_tracking_schematic.svg',facecolor='white')
print(OUT/'f1tenth_tracking_schematic.png')
