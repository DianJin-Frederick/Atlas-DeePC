"""Editable paper figure: Python SVG -> Adobe Illustrator AI/PDF/PNG.

Run this script, then execute the generated JSX in Illustrator.
Text remains live Times New Roman; mathtext uses Computer Modern.
"""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/pdf/f1tenth_noise_comparison'
STEM = 'f1tenth_noise_comparison'
SOURCE = ROOT / 'experiments/f1tenth/paper_nsr_20261027/report.json'
OUT.mkdir(parents=True, exist_ok=True)
rows = json.loads(SOURCE.read_text())
plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 8.4,
    'mathtext.fontset': 'cm', 'axes.titlesize': 8.7, 'axes.labelsize': 8.4,
    'svg.fonttype': 'none', 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#888888', 'axes.linewidth': .6,
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .6, 'ytick.major.width': .6,
    'xtick.major.pad': 3, 'ytick.major.pad': 3,
})
levels = sorted({r['sigma'] for r in rows})
x = np.arange(len(levels))
fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
styles = [
    ('Atlas-DeePC', None, '#005099', 'o', '-'),
    ('Contextual-DPC (30)', 30, '#e68122', 's', '--'),
    ('Contextual-DPC (60)', 60, '#48958a', '^', '-.'),
    ('Contextual-DPC (90)', 90, '#aa6688', 'D', ':'),
]
for name, columns, color, marker, linestyle in styles:
    selected = [r for r in rows if
                (r['method'].startswith('atlas') if columns is None
                 else r['config'].get('columns') == columns)]
    lookup = {r['sigma']: r for r in selected}
    ordered = [lookup[s] for s in levels]
    values = [
        [100*r['collision_free_laps']/r['evaluated'] for r in ordered],
        [np.nan if r['median_rmse_cm'] is None else r['median_rmse_cm'] for r in ordered],
        [r['timing']['median_ms'] for r in ordered],
    ]
    for i, (ax, y) in enumerate(zip(axes, values)):
        curve, = ax.plot(x, y, label=name, color=color, marker=marker, linestyle=linestyle,
                        linewidth=1.3, markersize=4.6, markerfacecolor='white',
                        markeredgewidth=1.05, clip_on=False)
        curve.set_gid(f'panel_{i+1}_{"atlas" if columns is None else "contextual_"+str(columns)}')

for ax, title, ylabel in zip(axes,
        ['(a) Collision-free lap success', '(b) Tracking accuracy', '(c) Online computation'],
        ['Success rate (%)', 'Median cross-track RMSE (cm)', 'Median controller time (ms/step)']):
    ax.set_title(title, loc='left', pad=8)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_xlabel('NSR (%)', labelpad=4)
    ax.set_xticks(x, [f'{100*s:g}' for s in levels])
    ax.set_xlim(-.2, len(levels)-.8)
    ax.grid(True, color='#e2e6eb', linewidth=.5)
    ax.set_axisbelow(True)
axes[0].set_ylim(-4, 105)
axes[0].set_yticks([0, 25, 50, 75, 100])
axes[1].set_ylim(0, 12)
axes[1].set_yticks(np.arange(0, 13, 2))
axes[2].set_ylim(0, 16)
axes[2].set_yticks([0, 4, 8, 12, 16])
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.52, 1.005), ncol=4, frameon=False, handlelength=2.5,
           columnspacing=1.4, handletextpad=.55, fontsize=8.2)
fig.subplots_adjust(left=.075, right=.99, bottom=.19, top=.745, wspace=.40)
base = OUT / STEM
fig.savefig(base.with_suffix('.svg'), facecolor='white')
plt.close(fig)

# Expand SVG marker references for consistent Illustrator import; text stays live.
tree = ET.parse(base.with_suffix('.svg'))
root = tree.getroot()
definitions = {e.get('id'): e for e in root.iter() if e.get('id')}
for parent in list(root.iter()):
    for child in list(parent):
        if child.tag.split('}')[-1] != 'use':
            continue
        href = child.get('{http://www.w3.org/1999/xlink}href', child.get('href', ''))
        glyph = copy.deepcopy(definitions[href.lstrip('#')])
        glyph.attrib.pop('id', None)
        group = ET.Element('{http://www.w3.org/2000/svg}g')
        group.set('transform', f'{child.get("transform", "")} translate({child.get("x", "0")},{child.get("y", "0")})')
        if child.get('style'):
            group.set('style', child.get('style'))
        group.append(glyph)
        parent.insert(list(parent).index(child), group)
        parent.remove(child)
ET.register_namespace('', 'http://www.w3.org/2000/svg')
ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
tree.write(base.with_suffix('.svg'), encoding='utf-8', xml_declaration=True)

jsx = r'''
var base=__BASE__;
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
'''
base.with_suffix('.jsx').write_text(jsx.replace('__BASE__', json.dumps(str(base))))
base.with_suffix('.json').write_text(json.dumps({
    'source':str(SOURCE), 'rows':rows, 'noise_axis':'equally spaced labeled categories',
    'text_font':'Times New Roman', 'math_font':'Computer Modern',
    'rmse_scope':'successful rollouts only; no successes -> omitted',
    'caption_embedded':False,
}, indent=2))
print(base.with_suffix('.jsx'))
