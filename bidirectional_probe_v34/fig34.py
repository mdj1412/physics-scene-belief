from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
def read(n):return json.loads((ROOT/n).read_text())
def main():
    r=read('results_diag/raw_analysis.json');m=read('decision_stats34.json');t=read('terminal_stats34.json');s=read('selector_stats34.json')
    fig,(a,b)=plt.subplots(1,2,figsize=(11.3,4.5),layout='constrained')
    for i,(key,clr) in enumerate([('Single direction','#21677d'),('Alternating','#ba6829')]):
        x=np.array([2,4,8]);y=np.array(r['mean'][i]);c=np.array(r['ci'][i]);a.errorbar(x+i*.06,y,yerr=(y-c[:,0],c[:,1]-y),color=clr,marker='o',capsize=3,label=key)
    a.axhline(.25,ls='--',lw=.9,c='0.4',label='Factual only');a.legend(frameon=False,fontsize=9);a.set(xticks=[2,4,8],ylabel='Brier loss for identifying surface orientation',xlabel='Paid calibration probes per environment',title='What the prefix identifies')
    b.axvline(0,ls='--',lw=.9,color='.4');y=np.arange(3)
    for i,(clr,lab) in enumerate([('#21677d','Single direction'),('#ba6829','Alternating')]):
        es=np.array([m['aug'][5*i],s['delta'][4*i],t['delta'][3*i]])*1000
        cs=np.array([m['refit_band'][5*i],s['ci'][4*i],t['ci'][3*i]])*1000
        b.errorbar(es,y+i*.2,xerr=(es-cs[:,0],cs[:,1]-es),fmt='o' if not i else 'D',color=clr,capsize=3,label=lab)
    b.set(yticks=y+.1,yticklabels=['Stop / continue','Choose sensor','Decode outcome'],xlabel='Estimated cue − no cue: outcome Brier × 1,000',title='Where the information changes prediction');b.invert_yaxis();b.margins(y=.24);b.legend(frameon=False,fontsize=9)
    fig.savefig(ROOT/'v34.png',dpi=175);plt.close(fig)
if __name__=='__main__':main()
