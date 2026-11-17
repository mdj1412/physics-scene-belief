"""Publication graphic, drawn from the benchmark output files."""
from pathlib import Path
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

root = Path(__file__).resolve().parents[1]
src = json.loads((root/'solver'/'metrics.json').read_text())
inc = json.loads((root/'analysis'/'inclined.json').read_text())
emp = json.loads((root/'analysis'/'empirical_summary.json').read_text())
marg = json.loads((root/'solver'/'marginal_baselines.json').read_text())
colors=['#22577a','#c45120','#447d56','#77629d','#79838b']
fig, ax=plt.subplots(1,3,figsize=(12.8,4.2),gridspec_kw={'width_ratios':[1.25,1.25,1]})
methods=['Flat PyBullet / Coulomb','Tangential spring','Inclined rigid','Two-regime affine','Source frequency']
for i,s in enumerate(src['studies']):
    source=s['source']; target=s['target']
    lab=f'{source} → {target}'
    met=s['metrics']; alt=next(r for r in emp if r['target']==target and r['model']=='two_regime')
    incline=next(r for r in inc if r['target']==target)
    metrics=[[met['pybullet']['mae_vx'],met['compliant_memory']['mae_vx'],incline['vx_mae'],alt['com_mae']],
             [met['pybullet']['brier_050'],met['compliant_memory']['brier_050'],incline['brier'],alt['brier'],marg[source]['050']['brier_source_marginal']]]
    for a,values in zip(ax[:2],metrics):
        for k,val in enumerate(values):a.scatter(val,k+(-.12 if i==0 else .12),s=42,color=colors[k],marker='o' if i==0 else 'D',edgecolors='white',linewidth=.5,zorder=3)
for a,n,title,xlabel in zip(ax,[4,5],['Turf · translation','Turf · near-stick proxy'],['Tangential COM MAE (m/s)','Brier loss (threshold 0.5 m/s)']):
    a.set(yticks=range(n),yticklabels=methods[:n],xlabel=xlabel,title=title)
    a.set_xlim(left=0);a.set_ylim(n-.5,-.5);a.grid(axis='x',alpha=.2);a.tick_params(axis='y',length=0)
rr=list(csv.DictReader((root/'data69'/'cross_2014_examples.csv').open()))
pred=list(csv.DictReader((root/'tennis'/'predictions.csv').open()))
for i,r in enumerate(rr):
    center=float(r['vx_out_digitized_ms']); lo=float(r['vx_out_digitization_low_ms']);hi=float(r['vx_out_digitization_high_ms'])
    ax[2].errorbar(center,i,xerr=[[center-lo],[hi-center]],fmt='s',color='#222222',capsize=4)
    for model,offset,color,ratio in [('pybullet',-.12,colors[0],0),('compliant_memory',.12,colors[1],.5)]:
        v=next(float(z['com_vx_m_s']) for z in pred if z['case']==r['case'].upper() and z['model']==model and float(z['e'])==.75 and float(z['mu'])==.5 and float(z['kt_ratio'])==ratio)
        ax[2].scatter(v,i+offset,color=color)
ax[2].set(yticks=[0,1],yticklabels=['42 mm','99 mm'],ylim=(1.65,-.65),xlabel='Rebound COM velocity (m/s)',title='Tennis · measured footprint')
ax[2].grid(axis='x',alpha=.2)
for a in ax:
    a.spines[['top','right','left']].set_visible(False); a.set_axisbelow(True)
fig.legend(handles=[Line2D([],[],marker='o',color='k',linestyle='none',label='Turf A → B'),Line2D([],[],marker='D',color='k',linestyle='none',label='Turf B → A'),Line2D([],[],marker='s',color='#222',linestyle='none',label='Tennis video ± digitization envelope')],loc='lower center',ncol=3,frameon=False)
fig.tight_layout(rect=(0,.10,1,1),w_pad=1.5)
for ext in ['png','pdf']:
    fig.savefig(root/'analysis'/f'real_impacts_v69.{ext}',dpi=185,bbox_inches='tight')
