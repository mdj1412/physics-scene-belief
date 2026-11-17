"""Plot the primary same-count comparison directly from server summary files."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

assert os.getenv("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES")==""
root=Path(__file__).resolve().parent
score=json.loads((root/'score63.json').read_text())
optical=json.loads((root/'optical_results63.json').read_text())
labels=['Neither','Transient only','Coherent only','Both']
y=np.arange(4)
fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
diff=np.array(optical['confidence_minus_random'])[:,1]*100
ci=np.array(optical['confidence_minus_random_ci'])[:,1,:]*100
risk=np.array([x['mean_Brier_confidence_minus'][0] for x in score['per_cell']])*1e6
for ax,val,lo,hi,title,name in zip(axes,[diff,risk[:,0]],[ci[:,0],risk[:,1]],[ci[:,1],risk[:,2]],
        ['Error among accepted sign pairs','Held-out deployment Brier loss'],['Difference (percentage points)','Difference (×10⁻⁶)']):
    ax.axvline(0,color='#808080',ls='--',lw=.9)
    ax.errorbar(val,y,xerr=[val-lo,hi-val],fmt='o',color='#1D6375',capsize=4)
    ax.set_yticks(y,labels)
    ax.set_ylim(3.6,-.6)
    ax.set_xlabel(name);ax.set_title(title,loc='left',fontsize=11,weight='bold')
    ax.grid(axis='y',color='#e6e6e6',lw=.6)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Confidence minus random abstention at the same read count',fontsize=14,weight='bold')
fig.savefig(root/'tracking_abstention63.png',dpi=175,facecolor='white')
