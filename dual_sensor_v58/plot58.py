"""Covariance across two objects for same- and cross-instrument readouts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
rows=json.loads((ROOT/'pilot_results58.json').read_text())['rows']
fig,axes=plt.subplots(1,2,figsize=(9.6,3.65),sharey=True,layout='constrained')
for env,ax in enumerate(axes):
    for measure,color,marker,offset,name in [
        ('same_cov','#905122','o',-.10,'Same sensor'),
        ('cross_cov','#176a78','s',.10,'Cross sensors')
    ]:
        for i,(rho,h) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
            estimate,low,high=[100*x for x in rows[f'1,1,{rho},{h},{env}'][measure]]
            ax.errorbar(i+offset,estimate,yerr=[[estimate-low],[high-estimate]],
                        marker=marker,color=color,capsize=3,linestyle='none',
                        label=name if i==0 else None)
    ax.axhline(0,color='#777777',lw=.7,ls='--')
    ax.set(xticks=range(4),xticklabels=['Neither','Sensor\nlatch','Physical\npersistence','Both'],
           title='Environment '+('A' if env==0 else 'B'),xlim=(-.5,3.5),ylim=(-1,9.5))
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='y',alpha=.16)
axes[0].set_ylabel('Double mismatch above marginal product (pp)')
axes[1].legend(frameon=False,loc='upper left')
fig.savefig(ROOT/'dual_sensor58.png',dpi=185)
