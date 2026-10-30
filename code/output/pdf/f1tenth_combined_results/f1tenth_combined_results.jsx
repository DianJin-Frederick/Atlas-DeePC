
(function(){
 var base="/Users/kumiko/Projects/atlas_deepc_reacher/output/pdf/f1tenth_combined_results/f1tenth_combined_results", old=app.userInteractionLevel, doc=null;
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
