"""Add verified terminal chart IDs to the existing, manually styled Illustrator figure."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/pdf/reacher_guidance_recovery_pairs/terminal_chart_labels"
SOURCE = Path("/Users/kumiko/Library/CloudStorage/OneDrive-UW-Madison/EECS/Research_Projects/atlas_DeePC/figures/reacher_p8_guidance_six_seed_pairs.ai")
META = ROOT / "output/pdf/reacher_guidance_recovery_pairs/reacher_p8_guidance_matched_k64_d53_six_pairs.json"
APP = "/Applications/Adobe Illustrator 2024/Adobe Illustrator.app"

COMMON = r'''
var spec=__SPEC__;
function stringify(o) {
    if(o===null) return "null";
    if(typeof o==="string") return '"'+o.replace(/\\/g,'\\\\').replace(/"/g,'\\"').replace(/\n/g,'\\n')+'"';
    if(typeof o!=="object") return String(o);
    var a=[];
    if(o instanceof Array) {for(var i=0;i<o.length;i++) a.push(stringify(o[i]));return '['+a.join(',')+']';}
    for(var key in o) if(o.hasOwnProperty(key)) a.push(stringify(key)+':'+stringify(o[key]));
    return '{'+a.join(',')+'}';
}
function writeJSON(path,o) {var f=new File(path);f.open('w');f.write(stringify(o));f.close();}
function target() {
    for(var i=0;i<app.documents.length;i++) {
        var d=app.documents[i];
        if(d.fullName.fsName===spec.source) return d;
    }
    return app.open(new File(spec.source));
}
function redArms(doc) {
    var a=[];
    for(var i=0;i<doc.pathItems.length;i++) {
        var p=doc.pathItems[i];
        if(!p.stroked || p.pathPoints.length!==3 || Math.abs(p.strokeWidth-1.45)>.01) continue;
        var c=p.strokeColor;
        if(c.typename!=="RGBColor" || c.red<180 || c.green>100 || c.blue>100) continue;
        var pts=[];for(var j=0;j<3;j++)pts.push([p.pathPoints[j].anchor[0],p.pathPoints[j].anchor[1]]);
        a.push({index:i,points:pts,width:p.strokeWidth,color:[c.red,c.green,c.blue]});
    }
    a.sort(function(a,b){
        var ay=(a.points[0][1]+a.points[1][1]+a.points[2][1])/3;
        var by=(b.points[0][1]+b.points[1][1]+b.points[2][1])/3;
        if(Math.abs(ay-by)>65) return by-ay;
        return a.points[0][0]-b.points[0][0];
    });
    return a;
}
'''

INSPECT = r'''
(function(){
    var doc=target(), ab=doc.artboards[0].artboardRect;
    var result={source:doc.fullName.fsName,saved:doc.saved,artboard:ab,arms:redArms(doc),texts:[]};
    for(var i=0;i<doc.textFrames.length;i++) {
        var t=doc.textFrames[i];
        try {result.texts.push({text:t.contents,font:t.textRange.characterAttributes.textFont.name,size:t.textRange.characterAttributes.size,bounds:t.visibleBounds});}
        catch(e) {result.texts.push({text:t.contents,error:String(e)});}
    }
    writeJSON(spec.out+'/live_inspection.json',result);
    var opt=new ExportOptionsPNG24();opt.artBoardClipping=true;opt.transparency=false;opt.horizontalScale=250;opt.verticalScale=250;
    doc.exportFile(new File(spec.out+'/live_before.png'),ExportType.PNG24,opt);
    return 'Inspected live figure; no document changes.';
})();
'''

ANNOTATE = r'''
(function(){
    var previousInteraction=app.userInteractionLevel;
    app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
    try {
    var doc=target(), arms=redArms(doc),ab=doc.artboards[0].artboardRect;
    if(arms.length!==12) throw new Error('Expected 12 terminal poses.');
    var font=app.textFonts.getByName('TimesNewRomanPSMT');
    // Verify the terminal pose in every panel against the recorded joint configuration.
    for(var i=0;i<12;i++) {
        var dx=arms[i].points[1][0]-arms[i].points[0][0];
        var dy=arms[i].points[1][1]-arms[i].points[0][1];
        var length=Math.sqrt(dx*dx+dy*dy),q=spec.labels[i].final_goal;
        if(Math.abs(dx/length-q[0])>.0001 || Math.abs(dy/length-q[2])>.0001)
            throw new Error('Terminal pose/data mismatch in panel '+i);
    }
    var opts=new IllustratorSaveOptions();opts.pdfCompatible=true;
    var backup=new File(spec.out+'/before_terminal_chart_labels.live.ai');
    if(backup.exists) throw new Error('Live backup already exists; refusing to replace it.');
    doc.saveAs(backup,opts);
    var layer=doc.layers.add();layer.name='Terminal reference chart IDs';
    var report=[];
    for(var i=0;i<12;i++) {
        var label=spec.labels[i],pos=spec.positions[i];
        var c=new RGBColor();c.red=206;c.green=59;c.blue=49;
        var t=layer.textFrames.add();t.contents=String(label.chart_id);
        t.name='Terminal chart ID - panel '+String.fromCharCode(97+i)+' - seed '+label.seed;
        t.textRange.characterAttributes.textFont=font;
        t.textRange.characterAttributes.size=7.5;
        t.textRange.characterAttributes.fillColor=c;
        t.left=ab[0]+pos[0];t.top=ab[1]-pos[1];
        // Same subtle white backing as the existing chart-switch IDs.
        var b=layer.pathItems.rectangle(t.top+.6,t.left-.8,t.width+1.6,t.height+1.2);
        var white=new RGBColor();white.red=white.green=white.blue=255;
        b.filled=true;b.fillColor=white;b.stroked=false;b.opacity=70;
        b.move(t,ElementPlacement.PLACEAFTER);
        report.push({panel:String.fromCharCode(97+i),seed:label.seed,chart_id:label.chart_id,bounds:t.visibleBounds});
    }
    doc.saveAs(new File(spec.out+'/reacher_p8_guidance_six_seed_pairs.ai'),opts);
    var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;png.horizontalScale=300;png.verticalScale=300;
    doc.exportFile(new File(spec.out+'/reacher_p8_guidance_six_seed_pairs.png'),ExportType.PNG24,png);
    var pdf=new PDFSaveOptions();pdf.preserveEditability=false;pdf.compatibility=PDFCompatibility.ACROBAT7;
    doc.saveAs(new File(spec.out+'/reacher_p8_guidance_six_seed_pairs.pdf'),pdf);
    writeJSON(spec.out+'/annotation_report.json',report);
    doc.close(SaveOptions.DONOTSAVECHANGES);
    app.open(new File(spec.out+'/reacher_p8_guidance_six_seed_pairs.ai'));
    return 'Added 12 verified terminal chart IDs; original file awaits visual QA.';
    } finally {app.userInteractionLevel=previousInteraction;}
})();
'''

PUBLISH = r'''
(function(){
    var staged=spec.out+'/reacher_p8_guidance_six_seed_pairs.ai';
    for(var i=app.documents.length-1;i>=0;i--) {
        var d=app.documents[i];
        if(d.fullName.fsName===staged) {
            if(!d.saved) throw new Error('Working copy has new unsaved edits; refusing to close.');
            d.close(SaveOptions.DONOTSAVECHANGES);
        }
    }
    app.open(new File(spec.source));
    return 'Updated original research figure opened.';
})();
'''


def prepare(mode):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [row for pair in json.loads(META.read_text())["rows"] for row in pair]
    labels=[]
    for row in rows:
        p=Path(row["source"])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row["sha256"]
        with np.load(p,allow_pickle=True) as d:
            labels.append(dict(seed=row['seed'],mode=row['mode'],chart_id=int(d['goal_chart'][-1])+1,
                               final_goal=d['final_goal'].tolist(),source=str(p)))
    positions=[(60,145),(208,111),(340,116),(478,116),(47,227),(230,288),
               (328,284),(467,286),(91,398),(230,396),(365,386),(502,386)]
    spec=dict(source=str(SOURCE),out=str(OUT),labels=labels,positions=positions)
    (OUT/'verified_terminal_charts.json').write_text(json.dumps(spec,indent=2))
    if mode=='annotate':
        backup=OUT/'before_terminal_chart_labels.disk.ai'
        if not backup.exists():
            shutil.copy2(SOURCE,backup)
    if mode=='publish':
        saved_original=OUT/'before_terminal_chart_labels.disk.ai'
        assert hashlib.sha256(SOURCE.read_bytes()).digest()==hashlib.sha256(saved_original.read_bytes()).digest(), 'Original file changed during editing.'
        original_pdf=SOURCE.with_suffix('.pdf')
        backup_pdf=OUT/'before_terminal_chart_labels.disk.pdf'
        if original_pdf.exists() and not backup_pdf.exists():
            shutil.copy2(original_pdf,backup_pdf)
        for suffix in ('.ai','.pdf'):
            shutil.copy2(OUT/('reacher_p8_guidance_six_seed_pairs'+suffix),SOURCE.with_suffix(suffix))
    script=COMMON+{'inspect':INSPECT,'annotate':ANNOTATE,'publish':PUBLISH}[mode]
    path=OUT/f'{mode}.jsx'
    path.write_text(script.replace('__SPEC__',json.dumps(spec)))
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['inspect','annotate','publish'])
    prepare(parser.parse_args().mode)
