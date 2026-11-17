"""Plot the two distinct endpoint denominators for the previously measured specimen."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parents[1]
fast=json.loads((root/'analysis/paired.json').read_text())['groups']['fast']
methods=['majority','continuous_loss_gate','single_affine_eb','elastic_impulse']
labels=['Majority vote','Continuous-loss gate','Single affine EB','Elastic impulse']
colors=['#496c91','#167d76','#935182','#d47432']
fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),sharey=True)
for ax,key,title,scale in zip(axes,['velocity','spin','accuracy'],['Velocity error (m/s)\n37 impacts','Spin error (rad/s)\n37 impacts','Release sign accuracy (%)\n22 resolved impacts'],[1,1,100]):
    for i,(m,c) in enumerate(zip(methods,colors)):
        a,b=[fast['performance'][m+'_k'+str(k)][key]*scale for k in (4,8)]
        ax.plot([a,b],[i,i],color=c,zorder=1,lw=2)
        ax.scatter([a],[i],s=76,facecolors='white',edgecolors=c,zorder=2,lw=1.5)
        ax.scatter([b],[i],s=26,color=c,zorder=3)
    ax.set_xlim(left=0)
    ax.set_title(title,fontsize=10)
    ax.grid(axis='x',color='#dddddd',lw=.6)
    ax.set_axisbelow(True)
    ax.spines[['right','top','left']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.margins(y=.22)
axes[0].set_yticks(np.arange(4),labels)
axes[0].invert_yaxis()
axes[-1].set_xlim(0,100)
fig.legend(handles=[Line2D([],[],marker='o',mec='black',mfc='white',ms=8,ls='none',label='4 calibration impacts'),Line2D([],[],marker='o',ms=5,color='black',ls='none',label='8 calibration impacts')],loc='lower center',ncol=2,frameon=False)
fig.suptitle('Fast incidence on A-B-S II soft  |  previously measured specimen',fontsize=13,fontweight='bold')
fig.tight_layout(rect=(0,.10,1,.91))
fig.savefig(root/'analysis/fast_endpoints.png',dpi=180,bbox_inches='tight')
