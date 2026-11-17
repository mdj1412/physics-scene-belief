"""Exploratory case-mix reweighting of already-frozen actions, without refitting."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
s=json.loads((root/'score57.json').read_text())
a,b=s['mean_by_regime']
names=s['methods']
out={}
for action in ['4x2_map','4x2_risk','8x1_map']:
    i=names.index(action)
    values={}
    for baseline in ['v54_gate','fixed_I']:
        j=names.index(baseline)
        da,db=a[i]-a[j],b[i]-b[j]
        values[baseline]={'delta_A':da,'delta_B':db,'cross':None if da*db>=0 else db/(db-da)}
    out[action]=values
(root/'prevalence57.json').write_text(json.dumps(out,indent=2)+'\n')
