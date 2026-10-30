
var spec={"base": "/Users/kumiko/Projects/atlas_deepc_reacher/output/pdf/atlas_deepc_pipeline/atlas_deepc_pipeline"};
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
