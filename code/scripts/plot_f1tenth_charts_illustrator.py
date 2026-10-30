"""Editable native Illustrator chart clouds, from saved noiseless atlas.

Run with a NumPy-enabled Python; --draw invokes Adobe Illustrator.
Edit K, MEMBERS, SEED or the JSX layout below to revise the figure.
All paths and text remain editable; math uses Computer Modern Unicode.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
K, MEMBERS, SEED = 5, 40, 20261027
STEM = 'f1tenth_chart_clouds_sigma0'
OUT = ROOT / 'output/pdf/f1tenth_chart_clouds_sigma0'
APP = '/Applications/Adobe Illustrator 2024/Adobe Illustrator.app'

JSX = r'''
var s=__SPEC__;
(function(){
 var old=app.userInteractionLevel;app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
 var doc=null;
 try{
 var W=432,H=308,FS=9*72/72.27;
 doc=app.documents.add(DocumentColorSpace.RGB,W,H);
 doc.artboards[0].artboardRect=[0,H,W,0];
 var roman=app.textFonts.getByName('TimesNewRomanPSMT');
 var cm=app.textFonts.getByName('CMUSerif-Roman'),mi=app.textFonts.getByName('CMUSerif-Italic');
 var g=doc.layers[0];g.name='Background';
 function col(h){var c=new RGBColor();c.red=parseInt(h.substr(1,2),16);c.green=parseInt(h.substr(3,2),16);c.blue=parseInt(h.substr(5,2),16);return c;}
 function layer(n){g=doc.layers.add();g.name=n;}
 function path(pts,c,w,alpha,dash,name){var p=g.pathItems.add(),q=[];for(var i=0;i<pts.length;i++)q.push([pts[i][0],H-pts[i][1]]);p.setEntirePath(q);p.filled=false;p.stroked=true;p.strokeColor=col(c);p.strokeWidth=w;p.opacity=alpha||100;if(dash)p.strokeDashes=[2,2];if(name)p.name=name;return p;}
 function text(x,y,str,align,math,rotate){var t=doc.textFrames.pointText([x,H-y-FS*.82]);t.move(g,ElementPlacement.PLACEATEND);t.contents=str;t.textRange.paragraphAttributes.justification=Justification.LEFT;t.textRange.characterAttributes.textFont=math?cm:roman;t.textRange.characterAttributes.size=FS;t.textRange.characterAttributes.fillColor=col('#151515');if(rotate)t.rotate(rotate);app.redraw();var b=t.visibleBounds;var dx=x-(align=='center'?b[0]+.25*(b[2]-b[0]):align=='right'?b[2]:b[0]);t.translate(dx,H-y-b[1]);return t;}
 var bg=g.pathItems.rectangle(H,0,W,H);bg.stroked=false;bg.fillColor=col('#ffffff');
 var blue='#247eb5',orange='#e57a12';
 function axis(x,y,w,h,lim,xt,yt,xlabel,ylabel,showY){
  var a={X:function(v){return x+w*(v-lim[0])/(lim[1]-lim[0]);},Y:function(v){return y+h-h*(v-lim[2])/(lim[3]-lim[2]);}};
  for(var j=0;j<yt.length;j++){var yy=a.Y(yt[j]);path([[x,yy],[x+w,yy]],'#e7e7e7',.35);path([[x-3,yy],[x,yy]],'#111111',.65);if(showY)text(x-5,yy-4,String(yt[j]),'right',true);}
  if(lim[0]<=0&&lim[1]>=0)path([[a.X(0),y],[a.X(0),y+h]],'#aaaaaa',.55,100,true);
  for(var j=0;j<xt.length;j++){var xx=a.X(xt[j]);path([[xx,y+h],[xx,y+h+3]],'#111111',.65);text(xx,y+h+5,String(xt[j]),'center',true);}
  path([[x,y],[x,y+h],[x+w,y+h]],'#111111',.7);
  text(x+w/2,y+h+20,xlabel,'center',false);
  if(showY){var t=text(8,y+h/2,ylabel,'center',false,90);t.left=6;t.top=H-(y+h/2-t.height/2);}
  return a;
 }
 function curve(a,ys,kind,c,w,alpha,name){var p=[];for(var t=0;t<ys.length;t++)p.push(kind=='xy'?[a.X(ys[t][0]),a.Y(ys[t][1])]:[a.X(ys[t][0]),a.Y(ys[t][1])]);path(p,c,w,alpha,false,name);}
 function dot(x,y,c,r){var p=g.pathItems.ellipse(H-y+r,x-r,2*r,2*r);p.filled=true;p.fillColor=col(c);p.stroked=false;}
 function triangle(a,rows,c){var n=rows.length,p=[a.X(rows[n-1][0]),a.Y(rows[n-1][1])],q=[a.X(rows[n-2][0]),a.Y(rows[n-2][1])];var theta=Math.atan2(p[1]-q[1],p[0]-q[0]),r=3.5;var pts=[[p[0]+r*Math.cos(theta),p[1]+r*Math.sin(theta)],[p[0]+r*Math.cos(theta+2.5),p[1]+r*Math.sin(theta+2.5)],[p[0]+r*Math.cos(theta-2.5),p[1]+r*Math.sin(theta-2.5)]];var z=path(pts,c,.4);z.closed=true;z.filled=true;z.fillColor=col(c);}
 for(var k=0;k<s.panels.length;k++){
  var panel=s.panels[k],x=35+k*79;
  layer('Chart '+(k+1)+' - axes and labels');
  text(x+34,6,'Chart '+(k+1),'center',false);
  var title=text(x+34,19,'n = '+panel.count,'center',true);
  title.characters[0].characterAttributes.textFont=mi;
  var top=axis(x,36,68,110,s.path_limits,s.path_xticks,s.path_yticks,'Forward (m)','Lateral (m)',k==0);
  var bottom=axis(x,192,68,56,[-.4,1,s.speed_limits[0],s.speed_limits[1]],[0,1],s.speed_ticks,'Relative time (s)','Speed (m/s)',k==0);
  layer('Chart '+(k+1)+' - 40 sampled members');
  for(var j=0;j<panel.members.length;j++){
   var v=panel.members[j];curve(top,v.xy.slice(0,5),'xy',blue,.55,27);curve(top,v.xy.slice(4),'xy',orange,.55,27);
   curve(bottom,v.speed.slice(0,5),'speed',blue,.55,27);curve(bottom,v.speed.slice(4),'speed',orange,.55,27);
  }
  layer('Chart '+(k+1)+' - representative column '+panel.representative_index);
  var v=panel.representative;
  curve(top,v.xy.slice(0,5),'xy',blue,1.5,100,'Representative past');curve(top,v.xy.slice(4),'xy',orange,1.5,100,'Representative future');
  triangle(top,v.xy.slice(4),orange);dot(top.X(0),top.Y(0),blue,2);
  curve(bottom,v.speed.slice(0,5),'speed',blue,1.5,100);curve(bottom,v.speed.slice(4),'speed',orange,1.5,100);
 }
 layer('Legend');
 // Two compact rows retain full-sized editable legend labels.
 var entries=[[55,286,'Member past',blue,.65,40],[55,298,'Member future',orange,.65,40],[235,286,'Representative past',blue,1.7,100],[235,298,'Representative future',orange,1.7,100]];
 for(var i=0;i<entries.length;i++){var e=entries[i];path([[e[0],e[1]+4],[e[0]+18,e[1]+4]],e[3],e[4],e[5]);text(e[0]+23,e[1],e[2],'left',false);}
 // Add enough bottom margin to keep the legend inside the artboard.
 doc.artboards[0].artboardRect=[0,H,W,-10];
 var opts=new IllustratorSaveOptions();opts.pdfCompatible=true;doc.saveAs(new File(s.base+'.ai'),opts);
 var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;png.horizontalScale=300;png.verticalScale=300;doc.exportFile(new File(s.base+'.png'),ExportType.PNG24,png);
 var pdf=new PDFSaveOptions();pdf.preserveEditability=false;doc.saveAs(new File(s.base+'.pdf'),pdf);
 var report='Text frames: '+doc.textFrames.length+'; paths: '+doc.pathItems.length+'; math font: '+cm.name;
 doc.close(SaveOptions.DONOTSAVECHANGES);doc=null;return report;
 }finally{if(doc)doc.close(SaveOptions.DONOTSAVECHANGES);app.userInteractionLevel=old;}
})();
'''

# Figure 3 style: square equal-scale cloud panels, light frames, one top legend.
# SVG keeps explicit anchors and live text; Illustrator imports the named groups.
EXPORT_JSX = r'''
var s=__SPEC__;
(function(){
 var old=app.userInteractionLevel;app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
 var doc=null;
 try{
  doc=app.open(new File(s.base+'.svg'));
  if(doc.textFrames.length<20)throw new Error('Expected editable text.');
  var ai=new IllustratorSaveOptions();ai.pdfCompatible=true;
  doc.saveAs(new File(s.base+'.ai'),ai);
  var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;
  png.horizontalScale=400;png.verticalScale=400;
  doc.exportFile(new File(s.base+'.png'),ExportType.PNG24,png);
  var pdf=new PDFSaveOptions();pdf.preserveEditability=false;
  doc.saveAs(new File(s.base+'.pdf'),pdf);
  var result='Editable text frames: '+doc.textFrames.length+'; groups: '+doc.groupItems.length;
  doc.close(SaveOptions.DONOTSAVECHANGES);doc=null;return result;
 }finally{if(doc)doc.close(SaveOptions.DONOTSAVECHANGES);app.userInteractionLevel=old;}
})();
'''

def make_svg(spec):
    import matplotlib as mpl
    from matplotlib.mathtext import math_to_image, MathTextParser
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextToPath
    mpl.rcParams['mathtext.fontset']='cm'
    mpl.rcParams['svg.fonttype']='path'
    width,height=432,229
    fs=9*72/72.27
    blue,orange,dark='#679bd0','#ee8222','#005099'
    root=ET.Element('svg',xmlns='http://www.w3.org/2000/svg',width=f'{width}pt',height=f'{height}pt',viewBox=f'0 0 {width} {height}')
    def el(parent,tag,**attrs):
        return ET.SubElement(parent,tag,{k.replace('_','-'):str(v) for k,v in attrs.items()})
    def label(parent,x,y,value,anchor='middle',size=fs,math=False):
        t=el(parent,'text',x=x,y=y,text_anchor=anchor,font_family='CMU Serif' if math else 'Times New Roman',font_size=size,fill='#111111')
        t.text=value
        return t
    math_count=0
    def equation(parent,x,top,value,size=8.1,anchor='middle'):
        # Exact Computer Modern math glyphs, imported as editable vector shapes.
        import copy
        nonlocal math_count
        math_count+=1
        buffer=io.BytesIO()
        math_to_image(value,buffer,format='svg',prop=FontProperties(size=size),dpi=72)
        svg=ET.fromstring(buffer.getvalue())
        glyphs={e.get('id'):e for e in svg.iter() if e.get('id')}
        # Illustrator does not reliably import nested SVG glyph references.
        # Expand each glyph into its own path before importing.
        for container in list(svg.iter()):
            for child in list(container):
                if child.tag.split('}')[-1]!='use':continue
                ref=child.get('{http://www.w3.org/1999/xlink}href',child.get('href',''))
                glyph=copy.deepcopy(glyphs[ref.lstrip('#')])
                glyph.attrib.pop('id',None)
                wrapper=ET.Element('g')
                transform=child.get('transform','')
                wrapper.set('transform',f'{transform} translate({child.get("x","0")},{child.get("y","0")})')
                wrapper.append(glyph)
                container.insert(list(container).index(child),wrapper)
                container.remove(child)
        w=float(svg.get('width').removesuffix('pt'))
        prefix=f'math{math_count}_'
        ids={e.get('id'):prefix+e.get('id') for e in svg.iter() if e.get('id')}
        group=el(parent,'g',id=f'ComputerModern_math_{math_count}',transform=f'translate({x-w/2 if anchor=="middle" else x},{top})')
        for child in svg:
            if child.tag.split('}')[-1] not in ('defs','g'):continue
            for elem in child.iter():
                elem.tag=elem.tag.split('}')[-1]
                for name,val in list(elem.attrib.items()):
                    if name=='id':elem.set(name,ids[val])
                    else:
                        for old,new in ids.items():
                            val=val.replace('#'+old,'#'+new)
                        elem.set(name,val)
                # Matplotlib includes a white background patch; omit it.
            for candidate in list(child):
                if candidate.get('id')==prefix+'patch_1':child.remove(candidate)
            group.append(child)
        return w
    def mixed_label(parent,x,baseline,words,math,size,anchor='start'):
        # Ordinary wording stays live Times New Roman; only formulas use CM.
        text_prop=FontProperties(family='Times New Roman',size=size)
        tw,_,_=TextToPath().get_text_width_height_descent(words,text_prop,False)
        mw,mh,md,_,_=MathTextParser('path').parse(math,dpi=72,prop=FontProperties(size=size))
        start=x-(tw+mw)/2 if anchor=='middle' else x
        label(parent,start,baseline,words,'start',size=size)
        equation(parent,start+tw,baseline-(mh-md),math,size=size,anchor='start')
    def line(parent,pts,color,stroke=.6,opacity=1):
        return el(parent,'polyline',points=' '.join(f'{x:.5f},{y:.5f}' for x,y in pts),fill='none',stroke=color,stroke_width=stroke,opacity=opacity,stroke_linejoin='round',stroke_linecap='round')
    def circle(parent,x,y,r,color,fill=None,stroke=.6,opacity=1):
        return el(parent,'circle',cx=x,cy=y,r=r,fill=fill or color,stroke=color,stroke_width=stroke,opacity=opacity)
    def arrow(parent,previous,tip,color,size=2.7):
        delta=np.asarray(tip)-np.asarray(previous)
        length=np.linalg.norm(delta)
        if length<1e-10:return
        direction=delta/length;normal=np.array([-direction[1],direction[0]])
        p=np.asarray(tip)
        vertices=[p+direction*size*.45,p-direction*size*.8+normal*size*.55,p-direction*size*.8-normal*size*.55]
        return el(parent,'polygon',points=' '.join(f'{x:.5f},{y:.5f}' for x,y in vertices),fill=color,stroke=color,stroke_width=.2)
    el(root,'rect',x=0,y=0,width=width,height=height,fill='white')
    legend=el(root,'g',id='Legend')
    for x,words,value,c in [(28,'Past samples ',r'$(T_{\mathrm{ini}}=5)$',blue),(162,'Future samples ',r'$(T_{\mathrm{f}}=10)$',orange),(299,'Nearest-centroid trajectory',None,dark)]:
        line(legend,[(x-21,11),(x-6,11)],c,1.4)
        if c==dark:arrow(legend,(x-21,11),(x-6,11),dark,2.4)
        if value:mixed_label(legend,x,14,words,value,size=8.4)
        else:label(legend,x,14,words,'start',size=8.4)
    side=74;left=31;pitch=79;top=43
    lim=spec['path_limits'];bound=max(abs(v) for v in lim)
    # Common symmetric range puts the aligned reference pose at each centre.
    bound=float(np.ceil(bound))
    for k,panel in enumerate(spec['panels']):
        x=left+k*pitch
        group=el(root,'g',id=f'Chart_{k+1}')
        mixed_label(group,x+side/2,36,f'Chart {k+1}: ',rf'$|\mathcal{{I}}_{{{k+1}}}|={panel["count"]}$',size=8.1,anchor='middle')
        xy=lambda row:(x+side*(row[0]+bound)/(2*bound),top+side*(bound-row[1])/(2*bound))
        frame=el(group,'g',id=f'Chart_{k+1}_axes')
        el(frame,'rect',x=x,y=top,width=side,height=side,fill='none',stroke='#999999',stroke_width=.5)
        line(frame,[(x+side/2,top),(x+side/2,top+side)],'#dddddd',.4)
        line(frame,[(x,top+side/2),(x+side,top+side/2)],'#dddddd',.4)
        members=el(group,'g',id=f'Chart_{k+1}_members')
        for v in panel['members']:
            pts=[xy(row) for row in v['xy']]
            line(members,pts[:5],blue,.42,.40);line(members,pts[4:],orange,.42,.34)
        rep=el(group,'g',id=f'Chart_{k+1}_representative')
        pts=[xy(row) for row in panel['representative']['xy']]
        line(rep,pts,'white',2.2);line(rep,pts,dark,1.25)
        for j in (4,14):arrow(rep,pts[j-1],pts[j],dark)
        if k==0:
            scale=side*2/(2*bound)
            line(frame,[(x+4,top+side-6),(x+4+scale,top+side-6)],'#777777',.7)
            label(frame,x+5+scale,top+side-4,'2 m','start',size=7)
        # Compact speed profiles retain the same representative styling.
        sy,sh=146,47
        slo,shi=spec['speed_limits']
        sp=lambda row:(x+side*(row[0]+.4)/1.4,sy+sh*(shi-row[1])/(shi-slo))
        speed=el(group,'g',id=f'Chart_{k+1}_speed')
        for tick in (0,4,8):
            yy=sp((0,tick))[1]
            line(speed,[(x,yy),(x+side,yy)],'#e5e5e5',.35)
            if k==0:label(speed,x-4,yy+2.8,str(tick),'end',size=8,math=True)
        for tick in ((-.4,0,1) if k==0 else (0,1)):
            xx=sp((tick,0))[0]
            line(speed,[(xx,sy+sh),(xx,sy+sh+2)],'#999999',.4)
            label(speed,xx,sy+sh+10,str(tick).replace('1.0','1'),'middle',size=7.7,math=True)
        line(speed,[(x,sy),(x,sy+sh),(x+side,sy+sh)],'#999999',.5)
        zero=sp((0,0))[0]
        line(speed,[(zero,sy),(zero,sy+sh)],'#dddddd',.4)
        for v in panel['members']:
            pts=[sp(row) for row in v['speed']]
            line(speed,pts[:5],blue,.42,.40);line(speed,pts[4:],orange,.42,.34)
        pts=[sp(row) for row in panel['representative']['speed']]
        line(speed,pts,'white',2.2);line(speed,pts,dark,1.25)
    label(root,229,131,'Forward displacement (m)')
    t=label(root,10,80,'Lateral displacement (m)',size=8.4);t.set('transform','rotate(-90 10 80)')
    label(root,229,223,'Relative time (s)')
    t=label(root,10,170,'Speed (m/s)');t.set('transform','rotate(-90 10 170)')
    ET.ElementTree(root).write(spec['base']+'.svg',encoding='utf-8',xml_declaration=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--draw',action='store_true');args=p.parse_args()
    source=ROOT/'experiments/f1tenth/noise_ratio_sweep_20261027/sigma_0'/f'atlas_k{K}.npz'
    a=np.load(source);d=np.load(ROOT/'data/f1tenth/offline_300_seconds_sigma_0_seed3408.npz')
    assert float(d['noise_sigma'])==0
    rng=np.random.default_rng(SEED);panels=[];all_xy=[];all_speed=[]
    def values(idx):
        e,start=a['metadata'][idx];y=d['Y'][e,start:start+15].copy();anchor=y[4].copy()
        delta=y[:,:2]-anchor[:2];c,s=np.cos(anchor[3]),np.sin(anchor[3])
        xy=delta@np.array([[c,-s],[s,c]])
        speed=np.c_[np.arange(-4,11)*.1,y[:,2]]
        all_xy.extend(xy.tolist());all_speed.extend(y[:,2].tolist())
        return dict(xy=xy.tolist(),speed=speed.tolist())
    for k in range(K):
        ids=np.flatnonzero(a['labels']==k)
        speeds=np.array([d['Y'][e,start+4,2] for e,start in a['metadata'][ids]])
        rep=int(ids[np.argmin(abs(speeds-a['centers'][k,0]))])
        chosen=rng.choice(ids,min(MEMBERS,len(ids)),replace=False)
        panels.append(dict(count=len(ids),member_indices=chosen.tolist(),members=[values(i) for i in chosen],representative_index=rep,representative=values(rep)))
    # Common limits across all charts; different horizontal/vertical scales.
    xy=np.array(all_xy);low=np.floor(xy.min(0)-.3);high=np.ceil(xy.max(0)+.3)
    limits=[float(low[0]),float(high[0]),float(low[1]),float(high[1])]
    ticks=lambda lo,hi,step:np.arange(np.ceil(lo/step)*step,hi+.001,step).tolist()
    sl=[float(np.floor(min(all_speed))),float(np.ceil(max(all_speed)))];sl[1]=max(sl[1],sl[0]+1)
    OUT.mkdir(parents=True,exist_ok=True)
    spec=dict(base=str(OUT/STEM),panels=panels,path_limits=limits,path_xticks=ticks(*limits[:2],4),path_yticks=ticks(*limits[2:],4),speed_limits=sl,speed_ticks=ticks(*sl,2),source=str(source),sigma=0,K=K,rank=40)
    (OUT/(STEM+'.json')).write_text(json.dumps(spec,indent=2))
    make_svg(spec)
    jsx=OUT/(STEM+'.jsx');jsx.write_text(EXPORT_JSX.replace('__SPEC__',json.dumps(spec)))
    print('Counts:',[x['count'] for x in panels],'limits',limits,'speed',sl)
    if args.draw:
        command=f'tell application "{APP}" to do javascript file "{jsx}"'
        result=subprocess.run(['osascript','-e',command],capture_output=True,text=True,check=True)
        print(result.stdout)

if __name__=='__main__':main()
