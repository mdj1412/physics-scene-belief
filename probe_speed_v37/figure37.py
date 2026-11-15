"""Plot the paired speed experiment from audited summaries."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
a=json.loads((ROOT/'stats37.json').read_text())
speed=np.array(a['speeds']); groups=a['scopes']; blue='#185773';orange='#af4e22'; grey='#77858d'
fig,axes=plt.subplots(1,3,figsize=(11.2,3.7))
ax=axes[0]
for tag,l,c in [('local_equal','All local regimes',blue),('weak','Weak',orange),('null','Local zero',grey)]:
    p=np.array(groups[tag]['extras'])[:,0]
    ax.plot(speed,p,'o-',color=c,label=l,ms=4)
ax.set_ylim(0,1.04);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
ax.set_ylabel('Slipping fourth probes')
ax.set_title('A    Contact state',loc='left',fontweight='bold',fontsize=11)
ax=axes[1]
for tag,l,c,off in [('strong_vs_low','Strong vs. weak / zero',blue,-.008),('weak_vs_null','Weak vs. zero',orange,.008)]:
    row=a['discrimination'][tag];vals=np.asarray(row['auc']);ci=np.asarray(row['ci95'])
    ax.errorbar(speed+off,vals,yerr=np.abs(ci.T-vals),color=c,label=l,fmt='o-',ms=4,capsize=3)
ax.axhline(.50,lw=.8,ls='--',color='.4')
ax.set_ylabel('Posterior strength AUROC')
ax.set_title('B    Surface discrimination',loc='left',fontweight='bold',fontsize=11)
ax=axes[2]
for tag,l,c,off in [('local_equal','All local regimes',blue,-.014),('weak_null_equal','Weak / zero',orange,0),('strong','Strong',grey,.014)]:
    vals=np.array(groups[tag]['brier_delta'][:3]);ci=np.array(groups[tag]['brier_simul95'][:3])
    ax.errorbar(speed+off,vals*1000,yerr=np.abs(ci.T-vals)*1000,fmt='o-',ms=4,color=c,label=l,capsize=3)
ax.axhline(0,color='.4',lw=.8,ls='--')
ax.set_ylabel('Gate − fixed Brier (×0.001)')
ax.set_title('C    Held-out prediction',loc='left',fontweight='bold',fontsize=11)
for ax in axes:
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,fontsize=8)
    ax.set_xlim(.45,1.15);ax.set_xticks(speed,['+0.50','+0.75','+1.10']);ax.set_xlabel('Final probe speed')
    ax.spines[['right','top']].set_visible(False);ax.tick_params(labelsize=8)
fig.subplots_adjust(left=.07,right=.99,wspace=.43,bottom=.31,top=.89)
fig.savefig(ROOT/'v37.png',dpi=185,bbox_inches='tight')
