"""Plot measured paired-sign information and the resulting policy selection."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

root=Path(__file__).resolve().parent
s=json.loads((root/'score57.json').read_text())
pairs=100*np.asarray(s['complete_pair_double_one_zero'])[:,::-1]
route=100*np.asarray(s['route_by_regime'])[:,[1,0,3]]
fig,axes=plt.subplots(1,2,figsize=(9.4,3.6),layout='constrained')
for ax,data,title,ticks,ylabel in zip(axes,[pairs,route],
    ['A   Two readable specimens','B   Choice after eight impacts'],
    [['0 flips','1 flip','2 flips'],['8×1 MAP','4×2 MAP','4×2 risk']],
    ['Fraction of complete pairs (%)','Selected fixed 3R+I (%)']):
    x=np.arange(3)
    for j,(row,name,color) in enumerate(zip(data,['Environment A','Environment B'],['#2b677c','#bc713e'])):
        marks=ax.bar(x+.36*(j-.5),row,width=.34,color=color,label=name)
        ax.bar_label(marks,labels=[f'{v:.2f}' if 0<v<1 else f'{v:.1f}' for v in row],fontsize=8,padding=3)
    ax.set(title=title,xticks=x,xticklabels=ticks,ylabel=ylabel,ylim=(0,68))
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.15)
axes[1].legend(frameon=False,fontsize=8,loc='upper right')
fig.savefig(root/'grouped_pilot57.png',dpi=185)
