
var base="/Users/kumiko/Projects/atlas_deepc_reacher/output/pdf/f1tenth_noise_comparison/f1tenth_noise_comparison";
(function(){
 var old=app.userInteractionLevel;
 app.userInteractionLevel=UserInteractionLevel.DONTDISPLAYALERTS;
 var doc=null;
 try{
  doc=app.open(new File(base+'.svg'));
  if(doc.textFrames.length<40)throw new Error('Live labels did not import.');
  var font=app.textFonts.getByName('TimesNewRomanPSMT');
  for(var i=0;i<doc.textFrames.length;i++){
   doc.textFrames[i].textRange.characterAttributes.textFont=font;
  }
  var ai=new IllustratorSaveOptions();ai.pdfCompatible=true;
  doc.saveAs(new File(base+'.ai'),ai);
  var png=new ExportOptionsPNG24();png.artBoardClipping=true;png.transparency=false;
  png.horizontalScale=400;png.verticalScale=400;
  doc.exportFile(new File(base+'.png'),ExportType.PNG24,png);
  var pdf=new PDFSaveOptions();pdf.preserveEditability=false;
  doc.saveAs(new File(base+'.pdf'),pdf);
  var result='Live text frames: '+doc.textFrames.length+'; vector paths: '+doc.pathItems.length;
  doc.close(SaveOptions.DONOTSAVECHANGES);doc=null;return result;
 }finally{if(doc)doc.close(SaveOptions.DONOTSAVECHANGES);app.userInteractionLevel=old;}
})();
