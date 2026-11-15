"""Paired policy effects, correction rates, and one-measurement discrimination ceiling."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'stats36.json').read_text())
regimes=['strong','weak','null','global_null']; names=['Strong','Weak','Local zero','Global zero']
x=np.arange(len(names))
fig,axes=plt.subplots(1,3,figsize=(11.4,3.95),gridspec_kw={'width_ratios':[1.18,1,.98]})
ax=axes[0]
for j,color,shift,title in [(1,'#26647e',-.13,'vs. v35 correction'),(2,'#b34828',.13,'vs. three-reading gate')]:
    p=np.array([s['scopes'][r]['delta'][j] for r in regimes]);ci=np.array([s['scopes'][r]['ci_simultaneous'][j] for r in regimes])
    ax.errorbar(p,x+shift,xerr=np.abs(ci.T-p),fmt='o',color=color,label=title,ms=4,capsize=2)
ax.axvline(0,lw=.8,color='.4')
ax.set_xticks([-.010,-.005,0,.005],['−0.010','−0.005','0','+0.005'])
ax.set_yticks(x,names);ax.set_ylim(3.65,-.65);ax.set_xlabel('Paired Brier loss difference');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21),frameon=False,fontsize=8)
ax.set_title('A    Four-reading gate',loc='left',fontsize=11,fontweight='bold')
ax=axes[1]
for j,color,shift,title in [(0,'#b6c9d0',-.18,'First three'),(1,'#26647e',.18,'With near probe')]:
    ax.bar(x+shift,[s['scopes'][r]['extras'][j] for r in regimes],width=.34,label=title,color=color)
ax.set_xticks(x,['Strong','Weak','Local\nzero','Global\nzero']);ax.set_ylim(0,1);ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_ylabel('Apply correction');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=2,frameon=False,fontsize=8)
ax.set_title('B    Gate decisions',loc='left',fontsize=11,fontweight='bold')
ax=axes[2]
for height,color,style,label in [(0.,'#888','--','0 mm'),(.023,'#26647e','-','23 mm')]:
    row=[v for v in s['sensor'] if v['height']==height]
    ax.plot([v['speed'] for v in row],[v['values'][0][0] for v in row],style+'o',c=color,label=label,ms=4)
ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0));ax.set_ylim(.49,1);ax.set_xlabel('Tangential speed');ax.set_ylabel('Privileged Bayes accuracy');ax.legend(frameon=False,fontsize=8)
ax.set_title('C    Strong vs. local zero',loc='left',fontsize=11,fontweight='bold')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
fig.subplots_adjust(wspace=.50,bottom=.28,left=.08,right=.98,top=.87)
fig.savefig(ROOT/'v36.png',dpi=185,bbox_inches='tight')
