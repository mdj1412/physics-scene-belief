import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
require();s=read(R/'summary.json');o=read(R/'timing_optimized_gpu.json')
names=['native','pair','nested128','nested512','qmc64_native','qmc64_pair','query','probe']
labels=['LMMSE16','Disagreement16','Nested128','Nested512','LMMSE64','Disagreement64','Always-query','Always-probe']
colors=['#4778ac','#d87932','#609d86','#38755f','#6e9ec7','#e5a166','#9a83b5','#aa9b88']
fig,axs=plt.subplots(2,2,figsize=(11,8))
for row,task in enumerate(['contact_event','slip_event']):
    for col in [0,1]:
        ax=axs[row,col];values=[s['means']['matched_nominal'][task][m] if col==0 else o[m]['task_medians_s'][task]*1000 for m in names]
        ax.barh(range(8),values,color=colors);ax.set_yticks(range(8),labels if col==0 else []);ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_xlabel('Brier loss (lower is better)' if col==0 else 'GPU latency, ms (lower is better)');ax.set_title(('Contact event' if row==0 else 'Stick event')+(' — loss' if col==0 else ' — optimized runtime'));ax.set_xlim(0,max(values)*1.22)
        for i,v in enumerate(values):ax.text(v+max(values)*.015,i,f'{v:.4f}' if col==0 else f'{v:.2f}',va='center',fontsize=9)
fig.suptitle('Matched expanded-prior control: decision quality and compute',fontsize=13);fig.tight_layout(rect=(0,0,1,.96));fig.savefig(R/'optimized_tradeoff.png',dpi=180);plt.close(fig)
