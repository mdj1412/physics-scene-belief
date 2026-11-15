from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    s=json.loads((ROOT/'policy_stats33.json').read_text());g=json.loads((ROOT/'results_prefix/prefix_gain33.json').read_text());t=json.loads((ROOT/'transfer33.json').read_text())
    fig,(a,b)=plt.subplots(1,2,figsize=(11.2,4),layout='constrained')
    labels=['2×2 vs 1×1','1×1 vs fixed 4','2×2 vs fixed 4','1×1 vs uncertainty','2×2 vs uncertainty']
    y=np.arange(5);a.axvline(0,color='.25',lw=.8)
    c=1000*np.asarray(s['ci']);v=1000*np.asarray(s['delta'])
    a.errorbar(v,y,xerr=(v-c[:,0],c[:,1]-v),fmt='o',capsize=3,color='#25587a')
    a.set(yticks=y,yticklabels=labels,xlabel='Difference in Brier loss × 1,000',title='Full stopping policy');a.invert_yaxis()
    b.axvline(0,color='.25',lw=.8);y=np.arange(4)
    labels=['High SD: gain','High SD: excess','Selected: gain','Selected: excess']
    for k,dd,cc,col,label in ((0,g['B']['paired_means'],g['B']['simultaneous95'],'#bd692c','Independent confirmation (2×2)'),(1,t['values'],t['ci'],'#4b7b55','Separate policy worlds (1×1)')):
        v=1000*np.asarray(dd);c=1000*np.asarray(cc);b.errorbar(v,y+(k-.5)*.26,xerr=(v-c[:,0],c[:,1]-v),fmt='o' if k==0 else 's',ms=4,capsize=2.5,color=col,label=label)
    b.set(yticks=y,yticklabels=labels,xlabel='Brier gain or excess gain × 1,000',title='When is an extra probe useful?');b.invert_yaxis()
    b.legend(loc='upper center',bbox_to_anchor=(.5,-.18),fontsize=8,frameon=False)
    fig.savefig(ROOT/'v33.png',dpi=175);plt.close(fig)
if __name__=='__main__':main()
