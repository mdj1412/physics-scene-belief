"""Show a measured rendered contact and the independent cause/policy endpoints."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from frame59 import read_signs

root=Path(__file__).resolve().parent
with np.load(root/'frame_panels/panel_00.npz') as f:
    frames=f['frames'][0,0,0].reshape(8,2,6,13,27)
    reads=read_signs(frames)
    ix=np.flatnonzero((reads[:,0]*reads[:,1])<0)
    if not len(ix):
        ix=np.flatnonzero(np.all(reads!=0,axis=1))
    sample=ix[0]
with open(root/'score59.json') as f: score=json.load(f)
fig=plt.figure(figsize=(10,5.1),layout='constrained')
grid=fig.add_gridspec(3,6,height_ratios=[1.,1.,2.2])
for contact in range(2):
    for time in range(6):
        ax=fig.add_subplot(grid[contact,time])
        ax.imshow(frames[sample,contact,time],cmap='gray',vmin=0,vmax=210,interpolation='nearest')
        if contact==0: ax.set_title(f'{time} ms',size=10)
        if time==0: ax.set_ylabel(['Reference','Impact'][contact])
        ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(grid[2,:])
p=np.asarray(score['primary']['accuracy'][:4])[::-1]*100
# Independent mechanism result directly from the scoring file when available.
d=json.loads((root/'diagnostics59.json').read_text())
def get_order(d):
    return [d['cause_accuracy'][a][0] for a in ('r0c0','r1c0','r0c1','r1c1')]
try: ca=np.asarray(get_order(d))*100
except (KeyError,TypeError):
    # Use exact population summaries calculated from saved pilot probabilities.
    from route59 import log_odds
    allc=[]
    for i in range(24):
        with np.load(root/'pilots'/f'panel_{i:02}.npz') as inp:
            codes=inp['codes']
        a=[]
        for r,c in ((0,0),(1,0),(0,1),(1,1)):
            _,o=log_odds(codes[r,c],r,c)
            a.append(.5*((o[0]>1e-12).mean()+1-(o[1]>1e-12).mean()))
        allc.append(a)
    ca=np.mean(allc,axis=0)*100
x=np.arange(4)
for i,(v,color,label) in enumerate([(ca,'#206a70','Cause identification'),(p,'#9b5d2d','Policy identification')]):
    marks=ax.bar(x+(i-.5)*.32,v-48,.31,bottom=48,color=color,label=label)
    ax.bar_label(marks,labels=[f'{val:.2f}%' for val in v],fontsize=8,padding=2)
ax.axhline(50,color='#555555',ls='--',linewidth=.8)
ax.set(ylim=(48,58),xticks=x,xticklabels=['Sham','Reset','Sham + camera','Reset + camera'],ylabel='Balanced accuracy (%)')
ax.legend(frameon=False,loc='upper left',ncol=2,fontsize=8)
ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'sensor_reset59.png',dpi=170)
