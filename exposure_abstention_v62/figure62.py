"""Compare paired optical coverage and the three confirmatory downstream contrasts."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
v=json.loads((root/'vision62.json').read_text())
s=json.loads((root/'score62.json').read_text())['uniform']['independent']
fig,(a,b)=plt.subplots(1,2,figsize=(10.3,3.9),layout='constrained')
x=np.arange(4)
for i,label,col in [(1,'v61 reader','#26737a'),(2,'Selective reader','#be5b33')]:
    y=np.array([row[i]['coverage'] for row in v['per_cell']])*100
    pos=x+(i-1.5)*.32
    a.bar(pos,y,width=.28,color=col,label=label)
    for xx,yy in zip(pos,y):
        a.text(xx,yy+.13,f'{yy:.2f}' if yy>.01 else f'{yy:.3f}',ha='center',fontsize=8)
a.set(xticks=x,xticklabels=['0.20 / 0.25','0.20 / 0.80','0.80 / 0.25','0.80 / 0.80'],
      ylabel='Readable active R/I pairs (%)',xlabel='LED pulse width / shutter exposure (ms)',ylim=(0,9.6))
a.legend(loc='upper left',fontsize=8,frameon=False)
a.set_title('A   What was still readable?',loc='left',fontsize=12,weight='bold')
rows=np.array(s['vs_v61reset_v61sham_gate']['mean_Brier'])*1.e6
for y,(pt,lo,hi) in zip([2,1,0],rows):
    b.errorbar(pt,y,xerr=np.array([[pt-lo],[hi-pt]]),color='#563964',fmt='o',ms=6,capsize=4,lw=1.6)
b.axvline(0,color='black',linestyle=':',linewidth=.9)
b.set(yticks=[2,1,0],yticklabels=['v61 reset','v61 sham','v54 gate'],
      ylim=(-.65,2.65),xlabel='Selective reset minus comparator Brier (×10⁻⁶)')
b.set_title('B   Did abstaining improve the decision?',loc='left',fontsize=12,weight='bold')
for ax in [a,b]:
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y' if ax is a else 'x',lw=.5,color='#dddddd')
fig.savefig(root/'exposure_abstention62.png',dpi=180)
