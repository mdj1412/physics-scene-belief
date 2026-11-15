"""Scientific figures from saved validation posteriors; no new inference."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/analysis');OUT.mkdir(exist_ok=True,parents=True)
fig,axs=plt.subplots(2,3,figsize=(12,7))
for j,case in enumerate([1,25,49]):
    d=json.loads((Path('results/main')/f'case_{case:03d}.json').read_text())
    p=np.load(Path('results/main')/f'posterior_{case:03d}.npz');t=p['theta'];b=p['rounding']
    sc=axs[0,j].scatter(t[:,0]*1000,t[:,1],c=b*1000,s=6,cmap='viridis',vmin=0,vmax=3,alpha=.55)
    axs[0,j].plot(d['theta'][0]*1000,d['theta'][1],'r*',markersize=11,label='Truth (evaluation only)')
    axs[0,j].set_xlabel('Geometry offset g [mm]');axs[0,j].set_ylabel('Restitution e');axs[0,j].set_title(f"{d['family']} | validation case {case}")
    gains=np.array(d['gains']);im=axs[1,j].imshow(gains,aspect='auto',cmap='Blues')
    for q in range(6):axs[1,j].plot(int(gains[q].argmax()),q,'x',color='#b72e28')
    axs[1,j].set_xticks(range(4),['view','left','right','center']);axs[1,j].set_yticks(range(6),[-30,-17,-7,4,15,28]);axs[1,j].set_ylabel('Query source offset [mm]');axs[1,j].set_title('Estimated EVSI (red x = selected)')
    fig.colorbar(im,ax=axs[1,j],shrink=.8)
axs[0,0].legend(fontsize=7)
fig.subplots_adjust(wspace=.55,hspace=.45,top=.92,bottom=.1,right=.86)
cax=fig.add_axes([.9,.59,.015,.28]);fig.colorbar(sc,cax=cax,label='Collision rounding [mm]')
fig.savefig(OUT/'posterior_and_selection.png',dpi=180);plt.close(fig)

rows=json.loads(Path('results/audit/cross_engine_matched.json').read_text())
fig,ax=plt.subplots(figsize=(6,4))
for family in ['corner','gate','deflection']:
    means=[np.mean([r['error'] for r in rows if r['family']==family and r['dt']==dt]) for dt in sorted({r['dt'] for r in rows})]
    ax.plot([1/15360,1/7680],means,'o-',label=family)
ax.set_xlabel('PyBullet timestep [s]');ax.set_ylabel('Mean velocity vector error [m/s]');ax.set_title('Matched solver configuration, 12 worlds / family');ax.legend();fig.tight_layout();fig.savefig(OUT/'cross_engine_convergence.png',dpi=180);plt.close(fig)
