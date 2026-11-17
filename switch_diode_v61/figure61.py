"""Real paid traces and the timing-reader ablation in the blind deployment population."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import vision61 as v
root=Path(__file__).resolve().parent
with np.load(root/'fixtures/panel_00.npz') as f:
    wave=f['wave'].reshape(4,16,2,2,128)[:,:,1]
    frames=f['frames'][:,0,0].reshape(4,16,2,8,15,27)[:,:,1]
    feat=v.optical_features(frames)
    flash=(feat[2]>38)
    _,valid,lags,triggers=v.estimate(feat,wave,2)
    choices=np.flatnonzero(valid[0]&valid[3]&(flash[0].argmax(-1)==flash[3].argmax(-1)))
    sample=choices[0]
    flash_ix=flash[0,sample].argmax()
s=json.loads((root/'score61.json').read_text())
cells=s['cells']['independent']
fig,(a,b)=plt.subplots(1,2,figsize=(10.1,4.15),layout='constrained')
colors=['#2a6a77','#bb542e','#644e88']
for w,desc,color,line in [(wave[0,sample,0],'Contact switch',colors[0],'-'),
                         (wave[0,sample,1],'Photodiode · (0, 0)',colors[1],'-'),
                         (wave[3,sample,1],'Photodiode · (2, 2)',colors[2],'--')]:
    a.plot(v.ADC,w,lw=1.6,color=color,ls=line,label=desc)
a.set_title('A   One paired impact',loc='left',weight='bold')
a.set(xlim=(-1.4,4.2),xlabel='Acquisition time on the ADC clock (ms)',ylabel='Measured signal (8-bit counts)')
a.legend(loc='center right',fontsize=8,frameon=True,facecolor='white',edgecolor='none',framealpha=.96)
a.text(.03,.04,f'Both impacts: LED in camera frame {flash_ix} (0-based)',transform=a.transAxes,fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
x=np.arange(4)
for idx,label,color,off,mrk in [(2,'LED-only reset',colors[1],-.07,'s'),(0,'Switch + photodiode reset',colors[0],.07,'o')]:
    yy=np.array([(k['mean_Brier'][idx]-k['mean_Brier'][6])*1e6 for k in cells])
    b.scatter(x+off,yy,label=label,c=color,marker=mrk,s=42,zorder=3)
b.axhline(0,ls=':',lw=1,color='black')
b.set_title('B   Change in deployment loss',loc='left',weight='bold')
b.set(xticks=x,xticklabels=['(0, 0)','(0, 2)','(2, 0)','(2, 2)'],xlim=(-.4,3.4),xlabel='Programmed (LED lag, camera trigger lag) in ms',ylabel='Mean Brier minus always v54 gate (×10⁻⁶)')
b.legend(frameon=False,fontsize=8,loc='upper left')
for ax in [a,b]:
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',color='#dedede',lw=.5)
    ax.set_axisbelow(True)
fig.savefig(root/'switch_diode61.png',dpi=180)
