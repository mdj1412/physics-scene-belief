"""Per-rubber figure with macro/micro aggregation and the distinct same-rubber reference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root=Path(__file__).resolve().parents[1]
d=json.loads((root/'analysis'/'audit.json').read_text())
groups=['hexer_duro','power_3','grass_dtecs','diamant','blowfish','abs_ii_soft']
names=['Hexer Duro','Power 3','Grass D.TecS','Diamant','Blowfish','A-B-S II soft']
rows=[d['fold'][g] for g in groups]+[d['macro'],d['micro']]
labels=[f'{n}  (n={d["support"][g]["n"]})' for n,g in zip(names,groups)]+['Product mean','Impact mean']
metrics=[('inclined','Inclined rigid','#245687','o'),('two_regime','Two-regime affine','#bc4d28','D'),
         ('single_affine','Single affine','#7767a2','^'),('known_rubber_rbf','Same-rubber RBF*','#45836c','s')]
fig,(ax,bx)=plt.subplots(1,2,figsize=(9.6,5.1),sharey=True)
for a,k,title in [(ax,0,'Velocity vector error (m/s)'),(bx,1,'Spin error (rad/s)')]:
    for j,(m,label,color,marker) in enumerate(metrics):
        a.scatter([v[m][k] for v in rows],np.arange(8)+(j-1.5)*.09,label=label,color=color,marker=marker,s=33,zorder=4)
    a.axhline(5.5,color='#aaa',lw=.8);a.set_xlim(left=0);a.grid(axis='x',alpha=.2);a.set_title(title,loc='left')
    a.spines[['left','top','right']].set_visible(False);a.tick_params(axis='y',length=0)
ax.set(yticks=np.arange(8),yticklabels=labels,ylim=(7.6,-.6))
fig.legend(handles=[Line2D([],[],marker=m,color=c,ls='none',label=l) for _,l,c,m in metrics],loc='lower center',bbox_to_anchor=(.5,.06),ncol=4,frameon=False)
fig.text(.5,.018,'*Diagnostic with target-rubber training labels. Other methods use zero target impacts.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.14,1,1))
for ext in ['pdf','png']:fig.savefig(root/'analysis'/f'rubber_holdout_v70.{ext}',dpi=170,bbox_inches='tight')
