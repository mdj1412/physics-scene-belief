"""Publication plot from completed auditable summaries; no model refit."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
root=Path(__file__).resolve().parent
au=json.loads((root/'audit35.json').read_text())
ss=json.loads((root/'transfer_stress35.json').read_text())
schedules=('same','alternating')
names=('Single direction','Alternating')
colors=('#145b79','#b14a1c')
fig,ax=plt.subplots(1,2,figsize=(9.7,4.55),gridspec_kw={'width_ratios':[1.7,1]})
for j,(sch,lab,c) in enumerate(zip(schedules,names,colors)):
    y=np.arange(5)+(.10 if j else -.10)
    vals=[ss[reg][sch]['delta'] for reg in ('full','half','S1_abs_oriented','zero','reverse')]
    ci=np.array([ss[reg][sch]['pointwise95'] for reg in ('full','half','S1_abs_oriented','zero','reverse')])
    ax[0].errorbar(vals,y,xerr=np.abs(ci.T-vals),fmt='o',label=lab,color=c,capsize=3,ms=5)
    r=au['timing']['ratio'][j]; lo,hi=au['timing']['ratio_simultaneous_ci95'][j]
    ax[1].errorbar(r-1,j,xerr=[[r-lo],[hi-r]],fmt='o',label=lab,capsize=4,ms=7,color=c)
ax[0].axvline(0,color='.4',lw=.9)
ax[0].set_yticks(np.arange(5),['Matched','Half amplitude','Measured magnitudes*','Zero','Reversed'])
ax[0].set_ylim(4.45,-.6)
ax[0].set_xlabel('Corrected − original Brier (lower is better)')
ax[0].set_title('A    Does the cue transfer to the query?',loc='left',fontsize=11,fontweight='bold')
ax[0].legend(loc='lower center',bbox_to_anchor=(.55,-.31),ncol=2,frameon=False,fontsize=9)
ax[1].set_yticks([0,1],names)
ax[1].set_ylim(1.55,-.55)
ax[1].axvline(0,color='.4',lw=.9)
ax[1].axvline(.02,color='#888',lw=.8,ls='--')
ax[1].xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x*100:.1f}%'))
ax[1].tick_params(axis='x',labelsize=9)
ax[1].set_xlabel('Change in full GPU p90')
ax[1].set_title('B    Inference latency',loc='left',fontsize=11,fontweight='bold')
for a in ax:
    a.spines[['right','top']].set_visible(False);a.grid(axis='x',alpha=.12)
fig.subplots_adjust(left=.17,right=.97,top=.86,bottom=.30,wspace=.48)
fig.text(.02,.015,'*Lab magnitudes substituted at query only; all event labels are simulated. Bars: paired block bootstrap 95% intervals (A: pointwise; B: simultaneous).',fontsize=8,color='.3')
fig.savefig(root/'v35.png',dpi=200,bbox_inches='tight')
fig.savefig(root/'v35.pdf',bbox_inches='tight')
