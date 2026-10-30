"""Build an editable Illustrator Figure 6 from verified saved Reacher sweeps.

Python verifies the experiment data and supplies the drawing specification;
Illustrator draws native paths/text and exports the publication PDF and PNG.
Run with the gym-robotics-mujoco Python environment and PYTHONPATH=src.
"""
import json
from pathlib import Path
import subprocess

from atlas_deepc.reporting.rank_sweep_k48_k64 import load_saved, load_chart_counts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/reacher_rank_sweep_k48_k64_k80_k96/illustrator_final"
STEM = "rank_success_k48_k64_k80_k96"
APP = "/Applications/Adobe Illustrator 2024/Adobe Illustrator.app"

JSX = r'''
var spec = __SPEC__;
(function () {
    var previousInteraction = app.userInteractionLevel;
    app.userInteractionLevel = UserInteractionLevel.DONTDISPLAYALERTS;
    try {
        var W = 432, H = 548;
        for(var openIndex=app.documents.length-1;openIndex>=0;openIndex--) {
            var existing=app.documents[openIndex];
            if(existing.saved && (existing.fullName.fsName===spec.base+".ai" ||
                                  existing.fullName.fsName===spec.base+".pdf"))
                existing.close(SaveOptions.DONOTSAVECHANGES);
        }
        var doc = app.documents.add(DocumentColorSpace.RGB, W, H);
        doc.artboards[0].artboardRect = [0,H,W,0];
        var normal = app.textFonts.getByName("TimesNewRomanPSMT");
        var cm = app.textFonts.getByName("CMUSerif-Roman");
        var cmi = app.textFonts.getByName("CMUSerif-Italic");
        // 9 TeX points = 8.966 Illustrator/PDF points. The paper is 11pt,
        // with footnotesize=9pt and a 6in text block.
        var FS = 9 * 72 / 72.27;
        var blue = "#377eb4", orange = "#e67e17";
        var green = "#258774", purple = "#9467ad";
        var grid = "#e4e8ed", gray = "#777777", black = "#111111";
        var group = doc.layers[0]; group.name = "Background";

        function color(hex) {
            var c = new RGBColor();
            c.red = parseInt(hex.substr(1,2),16);
            c.green = parseInt(hex.substr(3,2),16);
            c.blue = parseInt(hex.substr(5,2),16);
            return c;
        }
        function xy(x,y) { return [x,H-y]; }
        function line(points,hex,width,dashed,name) {
            var p = group.pathItems.add(), coords=[];
            for (var i=0;i<points.length;i++) coords.push(xy(points[i][0],points[i][1]));
            p.setEntirePath(coords); p.filled=false; p.stroked=true;
            p.strokeColor=color(hex); p.strokeWidth=width;
            if(dashed) p.strokeDashes=[3,2];
            if(name) p.name=name;
            return p;
        }
        function rect(x,y,w,h,fill) {
            var p=group.pathItems.rectangle(H-y,x,w,h);
            p.stroked=false;p.filled=true;p.fillColor=color(fill);return p;
        }
        function marker(x,y,hex,r,hollow,square,width) {
            var p;
            if(square==="diamond" || square==="triangle") {
                p=group.pathItems.add();
                var rr=1.25*r;
                var vertices=square==="diamond"
                    ? [xy(x,y-rr),xy(x+rr,y),xy(x,y+rr),xy(x-rr,y)]
                    : [xy(x,y-rr),xy(x+rr,y+rr*.75),xy(x-rr,y+rr*.75)];
                p.setEntirePath(vertices);p.closed=true;
            } else {
                p=square ? group.pathItems.rectangle(H-y+r,x-r,2*r,2*r)
                         : group.pathItems.ellipse(H-y+r,x-r,2*r,2*r);
            }
            p.stroked=true;p.strokeWidth=width || .65;p.strokeColor=color(hex);
            p.filled=true;p.fillColor=color(hollow?"#ffffff":hex);
            return p;
        }
        function text(x,y,s,align,hex,mathStart,mathEnd,rotation) {
            var t=group.textFrames.add();t.contents=s;
            var attr=t.textRange.characterAttributes;
            attr.textFont=normal;attr.size=FS;attr.fillColor=color(hex||black);
            if(mathStart !== undefined && mathStart >= 0) {
                for(var j=mathStart;j<mathEnd;j++) {
                    t.characters[j].characterAttributes.textFont =
                        /[Kpd]/.test(s.charAt(j)) ? cmi : cm;
                }
            }
            if(rotation) t.rotate(rotation);
            t.left=x-(align==="center"?t.width/2:(align==="right"?t.width:0));
            t.top=H-y;
            return t;
        }
        function layer(name) { group=doc.layers.add();group.name=name; }
        function axis(x,y,w,h,xmin,xmax,ymin,ymax,xticks,yticks,logx,showX,showY,categories) {
            function X(v) {
                if(categories) {
                    for(var c=0;c<categories.length;c++)
                        if(categories[c]===v) return x+w*(c+.35)/(categories.length-1+.7);
                    throw new Error("Unknown categorical chart count: "+v);
                }
                return x+w*(logx ? (Math.log(v)-Math.log(xmin))/(Math.log(xmax)-Math.log(xmin))
                                 : (v-xmin)/(xmax-xmin));
            }
            function Y(v) { return y+h-h*(v-ymin)/(ymax-ymin); }
            for(var i=0;i<xticks.length;i++) {
                var xx=X(xticks[i]);
                line([[xx,y],[xx,y+h]],grid,.45,false);
                line([[xx,y+h],[xx,y+h+2.5]],black,.5,false);
                if(showX) text(xx,y+h+4,String(xticks[i]),"center");
            }
            for(var j=0;j<yticks.length;j++) {
                var yy=Y(yticks[j]);
                line([[x,yy],[x+w,yy]],grid,.45,false);
                line([[x-2.5,yy],[x,yy]],black,.5,false);
                if(showY) text(x-5,yy-4.5,String(yticks[j]),"right");
            }
            line([[x,y],[x,y+h],[x+w,y+h]],gray,.6,false);
            return {X:X,Y:Y,x:x,y:y,w:w,h:h};
        }
        function curve(ax,rows,xkey,hex,dashed,square,hollow,radius) {
            var pts=[];
            for(var i=0;i<rows.length;i++) pts.push([ax.X(rows[i][xkey]),ax.Y(rows[i].success_rate)]);
            line(pts,hex,.85,dashed);
            for(var j=0;j<pts.length;j++) marker(pts[j][0],pts[j][1],hex,radius,hollow,square,.6);
        }
        function legendItem(x,y,s,hex,dashed,square,hollow,mathEnd) {
            line([[x,y+4],[x+20,y+4]],hex,.9,dashed);
            marker(x+10,y+4,hex,1.6,hollow,square,.7);
            text(x+26,y,s,"left",black,mathEnd?0:-1,mathEnd||0);
        }
        rect(0,0,W,H,"#ffffff");
        layer("Top legend");
        legendItem(112,8,"Without guidance",blue,true,false,false);
        legendItem(268,8,"With guidance",orange,false,false,false);
        layer("Shared axis labels");
        text(5,154,"Success rate (%)","left",black,-1,0,90);
        text(127,351,"Chart rank d","center",black,11,12);
        text(337,351,"Chart rank d","center",black,11,12);

        var counts=[48,64,80,96];
        var annotations={
            "48,8,terminal":[59,42], "48,8,graph":[52,110],
            "48,10,terminal":[61,42], "48,10,graph":[60,110],
            "64,8,terminal":[55,42], "64,8,graph":[53,110],
            "64,10,terminal":[61,42], "64,10,graph":[61,110],
            "80,8,terminal":[54,42], "80,8,graph":[49,110],
            "80,10,terminal":[61,42], "80,10,graph":[47,110],
            "96,8,terminal":[59,42], "96,8,graph":[55,110],
            "96,10,terminal":[57,42], "96,10,graph":[45,110]
        };
        for(var row=0;row<4;row++) for(var col=0;col<2;col++) {
            var k=counts[row],p=col?10:8,x=col?250:40,y=43+78*row;
            var letter="abcdefgh".charAt(2*row+col);
            layer("("+letter+") K="+k+", p="+p);
            var title="("+letter+") K = "+k+", p = "+p;
            text(x,y-16,title,"left",black,4,title.length);
            var ax=axis(x,y,174,53,27.5,70.5,-2,120,[28,35,42,49,56,63,70],[0,50,100],false,row===3,col===0);
            var entries=[];
            for(var m=0;m<2;m++) {
                var mode=m?"graph":"terminal",hex=m?orange:blue;
                var entry=spec.panels[k+","+p+","+mode];
                curve(ax,entry.curve,"rank",hex,!m,false,false,1.25);
                entries.push({entry:entry,hex:hex,mode:mode});
            }
            for(var j=0;j<entries.length;j++) {
                var e=entries[j],b=e.entry.best,pos=annotations[k+","+p+","+e.mode];
                var bx=ax.X(b.rank),by=ax.Y(b.success_rate),tx=ax.X(pos[0]);
                var ty=e.mode==="graph" ? y-1 : ax.Y(pos[1]);
                marker(bx,by,e.hex,3.2,true,false,1.1);
                var dx=tx-bx,dy=ty-by,length=Math.sqrt(dx*dx+dy*dy);
                if(length>8) line([[bx+dx*4/length,by+dy*4/length],[tx,ty+(e.mode==="graph"?4:-2)]],e.hex,.5,false);
                var label=b.success_rate.toFixed(0)+"% (d = "+b.rank+")";
                var t=text(tx,ty-5,label,"center",e.hex,label.indexOf("d"),label.length-1);
                // White backing keeps annotation text distinct from lines/grid.
                var back=rect(t.left-1,H-t.top-.5,t.width+2,t.height+1,"#ffffff");
                back.move(t,ElementPlacement.PLACEAFTER);
            }
        }
        layer("(i) Chart-count sweep");
        var label="(i) Success rate at the best tested rank";
        text(40,375,label,"left");
        var chartCounts=[1,2,4,8,16,24,32,48,64,80,96];
        var ax=axis(40,425,384,90,0,1,-2,105,chartCounts,[0,20,40,60,80,100],false,true,true,chartCounts);
        var styles=[
            {p:8,mode:"terminal",hex:green,dash:true,square:true,hollow:true},
            {p:8,mode:"graph",hex:purple,dash:false,square:"diamond",hollow:true},
            {p:10,mode:"terminal",hex:blue,dash:true,square:false,hollow:false},
            {p:10,mode:"graph",hex:orange,dash:false,square:"triangle",hollow:false}
        ];
        for(var s=0;s<styles.length;s++) {
            var style=styles[s],rows=[];
            for(var t=0;t<spec.counts.length;t++)
                if(spec.counts[t].output_dimension===style.p && spec.counts[t].reference_mode===style.mode)
                    rows.push(spec.counts[t]);
            rows.sort(function(a,b){return a.charts-b.charts;});
            curve(ax,rows,"charts",style.hex,style.dash,style.square,style.hollow,1.9);
        }
        // Put the four-curve legend outside the data region to avoid occlusion.
        legendItem(100,391,"p = 8, without guidance",green,true,true,true,5);
        legendItem(278,391,"p = 8, with guidance",purple,false,"diamond",true,5);
        legendItem(100,405,"p = 10, without guidance",blue,true,false,false,6);
        legendItem(278,405,"p = 10, with guidance",orange,false,"triangle",false,6);
        text(5,438,"Success rate (%)","left",black,-1,0,90);
        text(232,534,"Chart count K","center",black,12,13);
        app.redraw();

        var report={width_pt:W,height_pt:H,font_size_pt:FS,text_frames:doc.textFrames.length,
                    path_items:doc.pathItems.length,fonts:{},outside_artboard:[]};
        for(var t=0;t<doc.textFrames.length;t++) {
            var tf=doc.textFrames[t],b=tf.visibleBounds;
            if(b[0]<-.1 || b[1]>H+.1 || b[2]>W+.1 || b[3]<-.1)
                report.outside_artboard.push(tf.contents);
            for(var c=0;c<tf.characters.length;c++)
                report.fonts[tf.characters[c].characterAttributes.textFont.name]=true;
        }
        var aiOpt=new IllustratorSaveOptions();aiOpt.pdfCompatible=true;
        doc.saveAs(new File(spec.base+".ai"),aiOpt);
        var pngOpt=new ExportOptionsPNG24();pngOpt.artBoardClipping=true;
        pngOpt.transparency=false;pngOpt.horizontalScale=300;pngOpt.verticalScale=300;
        doc.exportFile(new File(spec.base+".png"),ExportType.PNG24,pngOpt);
        var pdfOpt=new PDFSaveOptions();pdfOpt.preserveEditability=false;
        pdfOpt.compatibility=PDFCompatibility.ACROBAT7;
        doc.saveAs(new File(spec.base+".pdf"),pdfOpt);
        function stringify(o) {
            if(o===null) return "null";
            if(typeof o==="string") return '"'+o.replace(/\\/g,'\\\\').replace(/"/g,'\\"').replace(/\n/g,'\\n')+'"';
            if(typeof o!=="object") return String(o);
            var a=[];
            if(o instanceof Array) { for(var i=0;i<o.length;i++) a.push(stringify(o[i]));return '['+a.join(',')+']'; }
            for(var key in o) if(o.hasOwnProperty(key)) a.push(stringify(key)+':'+stringify(o[key]));
            return '{'+a.join(',')+'}';
        }
        var f=new File(spec.report);f.open("w");f.write(stringify(report));f.close();
        doc.close(SaveOptions.DONOTSAVECHANGES);
        app.open(new File(spec.base+".ai"));
        return "Figure 6 saved: "+spec.base;
    } finally { app.userInteractionLevel=previousInteraction; }
})();
'''


def main(prepare_only=False):
    panels, provenance = load_saved()
    counts, count_provenance = load_chart_counts(panels)
    expected = [1, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96]
    for p in (8, 10):
        for mode in ("terminal", "graph"):
            assert sorted(r["charts"] for r in counts if
                          (r["output_dimension"], r["reference_mode"]) == (p, mode)) == expected
    OUT.mkdir(parents=True, exist_ok=True)
    spec = dict(
        base=str(OUT / STEM), report=str(OUT / "illustrator_report.json"),
        panels={f"{k},{p},{mode}": dict(
            curve=[{key:r[key] for key in ("rank", "success_rate")} for r in curve],
            best={key:best[key] for key in ("rank", "success_rate")})
            for (k,p,mode),(curve,best) in panels.items()},
        counts=counts,
    )
    (OUT / "figure_data.json").write_text(json.dumps(dict(
        **spec, provenance=provenance, count_provenance=count_provenance,
        textwidth_inches=6, footnotesize_tex_pt=9,
        math_font="Computer Modern Unicode Serif",
    ), indent=2))
    jsx_path = OUT / "draw_figure6.jsx"
    jsx_path.write_text(JSX.replace("__SPEC__", json.dumps(spec)))
    caption = (
        r"Success rates on 100 development seeds (211--310). "
        r"(a)--(h) Success rate versus chart rank $d$ for $K=48,64,80,96$, "
        r"with $p=8$ (left) and $p=10$ (right); highlighted markers indicate selected ranks. "
        r"(i) Success rate at the best tested rank versus chart count $K$ for all four "
        r"configurations; chart counts are displayed at equally spaced positions. "
        r"The new $K\leq32$ runs for $p=8$ (both configurations) and unguided $p=10$ "
        r"use 300 control steps; the remaining results use 320. "
        r"Rank-search ranges and tie-breaking rules differ across studies."
    )
    (OUT / "figure6_inclusion.tex").write_text(
        "\\begin{figure*}[t]\n    \\centering\n"
        f"    \\includegraphics[width=\\textwidth]{{{STEM}.pdf}}\n"
        f"    \\caption{{{caption}}}\n"
        "    \\label{fig:reacher-rank-ablation}\n\\end{figure*}\n"
    )
    command = f'tell application "{APP}" to do javascript file (POSIX file "{jsx_path}")'
    if not prepare_only:
        subprocess.run(["osascript", "-e", command], check=True)
    print(OUT / f"{STEM}.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    main(prepare_only=parser.parse_args().prepare_only)
