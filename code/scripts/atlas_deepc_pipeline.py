"""Create an editable Atlas-DeePC pipeline SVG and native Illustrator/PDF exports.

Usage: python3 atlas_deepc_pipeline.py --out-dir /path/to/figures
Requires only standard-library Python; AI/PDF/PNG export uses Adobe Illustrator.
Edit STYLE, BOXES, ROUTES and LABELS below. Coordinates are points, top-left origin.
SVG labels are live text, not outlines. Named module groups survive AI import.
Use --svg-only without Illustrator. Existing files require explicit --overwrite.
"""
import argparse
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

STEM = "atlas_deepc_pipeline"
APP = "/Applications/Adobe Illustrator 2024/Adobe Illustrator.app"
STYLE = dict(width=432, height=292, font_size=9 * 72 / 72.27,
             font="Times New Roman", ink="#243341", muted="#61717c",
             blue="#377eb4", teal="#258774", orange="#d87a19")
# name, x, y, width, height, title, subtitle, color, dashed outline
BOXES = [
    ("01_Data", 12,36,82,43,"Recorded data","Input-output trajectories","blue",False),
    ("02_Clustering",116,36,82,43,"Clustering","Selected features","blue",False),
    ("03_Subspace_fitting",220,36,88,43,"Subspace fitting","Local low-rank bases","blue",False),
    ("04_Behavioral_atlas",330,36,90,43,"Behavioral atlas","Precomputed charts","teal",False),
    ("05_Measurements",12,135,82,43,"Measurements","Recent input-output data","blue",False),
    ("06_Chart_selection",116,135,82,43,"Chart selection","Projection residual","blue",False),
    ("07_Predictive_control",220,135,110,43,"Predictive control","Low-dimensional coordinates","blue",False),
    ("08_Robot",352,135,68,43,"Robot","Apply control","teal",False),
    ("09_Optional_planner",220,219,110,43,"Reference planner","Precomputed transition graph","orange",True),
    ("10_Task_goal",352,219,68,43,"Task goal","Target / reference","orange",False),
]
# name, polyline points, color, dashed
ROUTES = [
    ("Data_to_clustering",[(94,57.5),(116,57.5)],"blue",False),
    ("Clusters_to_fitting",[(198,57.5),(220,57.5)],"blue",False),
    ("Bases_to_atlas",[(308,57.5),(330,57.5)],"teal",False),
    ("Stored_bases",[(375,79),(375,100),(157,100),(157,135)],"teal",False),
    ("History_to_selection",[(94,156.5),(116,156.5)],"blue",False),
    ("Selected_chart",[(198,156.5),(220,156.5)],"blue",False),
    ("Control_input",[(330,156.5),(352,156.5)],"blue",False),
    ("Planner_chart_index",[(157,178),(157,240.5),(220,240.5)],"orange",True),
    ("Planner_task_goal",[(352,240.5),(330,240.5)],"orange",True),
    ("Tracking_reference",[(275,219),(275,178)],"orange",True),
    ("Measurement_feedback",[(386,178),(386,192),(426,192),(426,282),(6,282),(6,156.5),(12,156.5)],"muted",False),
]
# x, top, text, size, color, bold, alignment
LABELS = [
    (12,10,"OFFLINE",9,"blue",True,"start"),
    (66,10,"Construct the behavioral atlas",9,"ink",False,"start"),
    (12,111,"ONLINE",9,"blue",True,"start"),
    (220,111,"Select a chart and compute control",9,"ink",False,"start"),
    (265,86,"Stored chart bases",8,"teal",False,"middle"),
    (12,224,"OPTIONAL GUIDANCE",7.4,"orange",True,"start"),
    (162,225,"Chart index",7.8,"orange",False,"start"),
    (281,189,"Tracking reference",7.8,"orange",False,"start"),
    (104,269,"New measurements",7.8,"muted",False,"start"),
]

JSX = r'''
var spec=__SPEC__;
(function(){
    var old=app.userInteractionLevel;
    app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
    var doc=null;
    try {
        doc=app.open(new File(spec.base+".svg"));
        doc.layers[0].name="Atlas-DeePC - editable vector modules";
        if(doc.textFrames.length<25) throw new Error("Expected live editable text frames.");
        var ai=new IllustratorSaveOptions();ai.pdfCompatible=true;
        doc.saveAs(new File(spec.base+".ai"),ai);
        var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;
        png.horizontalScale=300;png.verticalScale=300;
        doc.exportFile(new File(spec.base+".png"),ExportType.PNG24,png);
        var pdf=new PDFSaveOptions();pdf.preserveEditability=false;
        pdf.compatibility=PDFCompatibility.ACROBAT7;
        doc.saveAs(new File(spec.base+".pdf"),pdf);
        var report="Live text frames: "+doc.textFrames.length+"; module groups: "+doc.groupItems.length;
        doc.close(SaveOptions.DONOTSAVECHANGES);doc=null;
        return report;
    } finally {
        if(doc) doc.close(SaveOptions.DONOTSAVECHANGES);
        app.userInteractionLevel=old;
    }
})();
'''


def node(parent, tag, **attrs):
    return ET.SubElement(parent, tag, {k.replace("_", "-"): str(v) for k,v in attrs.items()})


def text(parent, x, top, value, size, color, bold=False, align="middle"):
    t=node(parent,"text",x=x,y=top+size*.8,fill=STYLE[color],
           font_family=STYLE["font"],font_size=size,font_weight="bold" if bold else "normal",
           text_anchor=align)
    t.text=value


def make_svg(path):
    w,h=STYLE["width"],STYLE["height"]
    root=ET.Element("svg",dict(xmlns="http://www.w3.org/2000/svg",width=f"{w}pt",height=f"{h}pt",viewBox=f"0 0 {w} {h}"))
    node(root,"title").text="Atlas-DeePC: offline atlas construction and modular online control"
    node(root,"rect",x=0,y=0,width=w,height=h,fill="#ffffff")
    for name,x,y,bw,bh,title,detail,col,dashed in BOXES:
        g=node(root,"g",id=name)
        fill={"blue":"#f5f8fc","teal":"#f1f8f6","orange":"#fff8ef"}[col]
        rect=node(g,"rect",x=x,y=y,width=bw,height=bh,rx=3,fill=fill,stroke=STYLE[col],stroke_width=.7)
        if dashed: rect.set("stroke-dasharray","3 2")
        text(g,x+bw/2,y+9,title,STYLE["font_size"],"ink",True)
        text(g,x+bw/2,y+25,detail,7.2,"muted")
    for name,pts,col,dashed in ROUTES:
        g=node(root,"g",id=name)
        line=node(g,"polyline",points=" ".join(f"{x},{y}" for x,y in pts),fill="none",stroke=STYLE[col],stroke_width=.8)
        if dashed: line.set("stroke-dasharray","3 2")
        (ax,ay),(bx,by)=pts[-2:]
        dx,dy=bx-ax,by-ay
        norm=(dx*dx+dy*dy)**.5
        dx,dy=dx/norm,dy/norm
        arrow=[(bx,by),(bx-4*dx+1.7*dy,by-4*dy-1.7*dx),(bx-4*dx-1.7*dy,by-4*dy+1.7*dx)]
        node(g,"polygon",points=" ".join(f"{x},{y}" for x,y in arrow),fill=STYLE[col])
    labels=node(root,"g",id="11_Headings_and_connection_labels")
    for x,y,value,size,col,bold,align in LABELS:
        text(labels,x,y,value,size,col,bold,align)
    ET.indent(root)
    ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--overwrite",action="store_true")
    p.add_argument("--svg-only",action="store_true")
    args=p.parse_args();out=args.out_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    base=out/STEM
    for ext in ("svg","ai","pdf","png","jsx"):
        if base.with_suffix("."+ext).exists() and not args.overwrite:
            raise SystemExit("Output exists. Save manual edits separately before using --overwrite.")
    make_svg(base.with_suffix(".svg"))
    if args.svg_only: return
    jsx=base.with_suffix(".jsx")
    jsx.write_text(JSX.replace("__SPEC__",json.dumps(dict(base=str(base)))),encoding="utf-8")
    apple=f'tell application "{APP}" to do javascript file (POSIX file "{jsx}")'
    r=subprocess.run(["osascript","-e",apple],capture_output=True,text=True)
    if r.returncode: raise RuntimeError(r.stderr)
    print(r.stdout.strip())
    for ext in ("svg","ai","pdf","png"):
        f=base.with_suffix("."+ext)
        assert f.is_file() and f.stat().st_size>0, f
        print(f)


if __name__=="__main__": main()
