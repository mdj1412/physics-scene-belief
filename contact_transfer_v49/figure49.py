"""Plot paired absolute risk and two prespecified Brier contrasts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parent
d=json.loads((root/'score49.json').read_text())
ks=np.array(d['kappa'])
loss=np.array(d['loss'])
effect=np.array(d['primary_local_minus_fixed_and_gate'])
control=json.loads((root/'transfer_controls49.json').read_text())
fig,axes=plt.subplots(1,3,figsize=(12,4.35),layout='constrained')
colors=['#155887','#ac5228','#14745d']
for i,(idx,name) in enumerate([(2,'Local direct'),(3,'Fixed direct'),(4,'Local gate')]):
    axes[0].plot(ks,loss[1,:,idx],marker='o',color=colors[i],label=name)
axes[0].plot(ks,control['known_orientation_brier'],marker='x',color='#55555f',ls=':',label='Known sign law')
axes[0].set_title('A   Absolute risk at ρ = 0.5',loc='left')
axes[0].set_ylabel('Held-out Brier loss')
for i,r in enumerate(d['rho']):
    y,lo,hi=effect[i,:,:,0].T[0],effect[i,:,:,1].T[0],effect[i,:,:,2].T[0]
    axes[1].plot(ks,y,marker='o',color=colors[i],label=f'ρ = {r:g}')
    axes[1].fill_between(ks,lo,hi,color=colors[i],alpha=.12)
axes[1].set_title('B   Local direct − fixed direct',loc='left')
axes[1].set_ylabel('Paired Brier difference')
y,lo,hi=effect[0,:,1].T
axes[2].plot(ks,y,marker='o',color='#155887',label='All spatial regimes')
axes[2].fill_between(ks,lo,hi,color='#155887',alpha=.18)
axes[2].set_title('C   Local direct − local gate',loc='left')
for j,ax in enumerate(axes):
    if j: ax.axhline(0,color='#444',lw=.9,ls='--')
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlabel('Weak-to-full sign retention (κ)')
    ax.set_xticks(ks)
    ax.grid(axis='y',alpha=.2)
    ax.legend(frameon=False,fontsize=8)
fig.suptitle('One weak preload in a second contact solver',fontsize=15)
fig.savefig(root/'contact_transfer_v49.png',dpi=180)
