from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
require();s=read(R/'summary.json');names=['adaptive','adaptive_global','fixed32','fixed128','global128','global32','native','qmc64_native','nested128','nested512','query','probe']
fig,axs=plt.subplots(2,3,figsize=(16,9))
for k,task in enumerate(['contact_event','slip_event']):
    for col in range(3):
        ax=axs[k,col]
        if col==0:vals=[s['means']['matched_nominal'][task][m] for m in names];label='Mean Brier loss'
        elif col==1:vals=[s['timing_gpu'][m]['task_medians_s'][task]*1000 for m in names];label='GPU online median (ms)'
        else:vals=[100*s['ranking']['matched_nominal'][m].get('seed2_agreement',[np.nan]*2)[k] for m in names];label='Top-1 agreement across acquisition seeds (%)'
        ax.barh(range(len(names)),vals,color=['#b9582b' if m=='adaptive' else '#447fa0' for m in names]);ax.set_yticks(range(len(names)),names if col==0 else []);ax.set_ylim(len(names)-.5,-.5);ax.set_title(task);ax.set_xlabel(label);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        hi=np.nanmax(vals);ax.set_xlim(0,110 if col==2 else hi*1.22)
        for j,v in enumerate(vals):
            if np.isfinite(v):ax.text(v+hi*.015,j,f'{v:.4f}' if col==0 else f'{v:.1f}',va='center',fontsize=8)
            else:ax.text(3,j,'deterministic' if names[j] in ['native','qmc64_native','query','probe'] else 'not audited',va='center',fontsize=8,color='#777777')
fig.tight_layout();fig.savefig(R/'tradeoff.png',dpi=160);plt.close(fig);print('Plot generated with aligned rows')
