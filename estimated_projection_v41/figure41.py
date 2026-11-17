"""Render the frozen primary comparisons from panel-level statistics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent

def main():
    s=json.loads((ROOT/'stats41.json').read_text())
    d=json.loads((ROOT/'direction_audit41.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.3),gridspec_kw={'width_ratios':[1.0,1.02,1.08]})
    plt.rcParams.update({'font.size':9})
    colors=['#236483','#b55027','#37765d','#7f5191']
    ax=axes[0]
    for col,(name,v) in zip(colors,d['cells'].items()):
        n=np.array([4,8,16,32])
        p=v['prefix']
        ax.plot(n,[p[str(k)]['angle_deg']['mean'] for k in n],marker='o',ms=4,lw=1.6,color=col,label=name.replace('_',' / ')+' mm')
    ax.set(xscale='log',xticks=[4,8,16,32],xticklabels=['4','8','16','32'],xlabel='Matched calibration pairs',ylabel='Mean principal angle (degrees)',title='A. Recovering the direction')
    ax.minorticks_off()
    ax.legend(loc='best',frameon=False,fontsize=8)
    ax=axes[1]
    for idx,entry in enumerate(s['by_family_height'].items()):
        name,record=entry
        part=record['est_vs_original']; m,l,u=[part[k]*1000 for k in ('mean','lo','hi')]
        ax.errorbar(m,3-idx,xerr=[[m-l],[u-m]],fmt='o',color=colors[idx],capsize=3)
    ax.axvline(0,ls='--',lw=1,color='#6c7375')
    ax.set(yticks=list(range(4)),yticklabels=[k.replace('_h',' / ') for k in reversed(s['by_family_height'])],
           xlabel='Estimated − original Brier (×1,000)',title='B. Projection at fixed shrinkage',ylim=(-.5,3.5))
    ax=axes[2]
    zero=json.loads((ROOT/'no_correction41.json').read_text())['paired_correction_minus_zero']
    pairs=[s['selector_minus_fixed'],s['selector_minus_optimizer'],{k:np.asarray(v)[:,2] for k,v in zero.items()}]
    for j,val in enumerate(pairs):
        for i in range(3):
            m,l,u=[val[k][i]*1000 for k in ('mean','lo','hi')]
            ax.errorbar(m,2-i+(.13,0,-.13)[j],xerr=[[m-l],[u-m]],fmt=('o','s','^')[j],capsize=3,
                        color=colors[j],label=(('vs fixed','vs optimizer','vs no correction')[j] if i==0 else None))
    ax.axvline(0,ls='--',lw=1,color='#6c7375')
    ax.set(yticks=[2,1,0],yticklabels=['Original','Oracle','Estimated'],ylim=(-.5,2.5),
           xlabel='Selector Brier difference (×1,000)',title='C. Gain from choosing a policy')
    ax.legend(frameon=False,fontsize=8,loc='best')
    for ax in axes:
        ax.spines[['right','top']].set_visible(False)
        ax.grid(axis='y',alpha=.14)
        ax.title.set_fontsize(11)
    fig.tight_layout(pad=1.2,w_pad=1.7)
    fig.savefig(ROOT/'v41.png',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__': main()
