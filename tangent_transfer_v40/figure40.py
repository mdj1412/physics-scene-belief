"""Plot target effects against forecasts and the matched contour changes."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

root=Path(__file__).resolve().parent
s=json.loads((root/'stats40.json').read_text())
f=json.loads((root/'forecast40.json').read_text())['cells']
fonts={x.name for x in font_manager.fontManager.ttflist}
kfont=next((x for x in ('Apple SD Gothic Neo','Noto Sans CJK KR','NanumGothic') if x in fonts),None)
if kfont: plt.rcParams['font.family']=[kfont,'DejaVu Sans']
fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained',sharey=True)
variants=('base','hollow','bulge')
laws=('saturating','crossover')
colors={'saturating':'#1c686b','crossover':'#a94929'}
for ax,fam in zip(axes,('shelf','ridge')):
    ax.axvline(0,color='#686867',linestyle='--',lw=1)
    for i,var in enumerate(variants):
        for j,law in enumerate(laws):
            key=f'{fam}_{var}_{law}'
            y=i*2+j*.63
            c=s['cells'][key]
            lo,hi=np.array(c['ci'])*1000
            x=c['mean']*1000
            ax.hlines(y,lo,hi,lw=2,color=colors[law])
            ax.scatter(x,y,marker='o',s=48,c=colors[law],zorder=3)
            ax.scatter(f[key]['delta']*1000,y-.19,marker='D',s=33,facecolors='white',edgecolors=colors[law],zorder=4)
    ax.set_title(fam,loc='left',fontweight='bold')
    ax.set_yticks([0,.63,2,2.63,4,4.63],
          [f'{v} / {law}' for v in variants for law in laws])
    ax.grid(axis='x',color='#e0e0e0',lw=.6)
    ax.spines[['left','right','top']].set_visible(False)
    ax.tick_params(axis='y',length=0)
    ax.set_xlabel('Δ Brier × 1,000')
low=min(ax.get_xlim()[0] for ax in axes);high=max(ax.get_xlim()[1] for ax in axes)
for ax in axes:
    ax.set_xlim(low,high)
axes[0].set_ylim(5.1,-.65)
axes[1].legend(handles=[
    Line2D([],[],ls='none',marker='D',markerfacecolor='white',markeredgecolor='#333333',
           label='사전 예측' if kfont else 'Frozen forecast'),
    Line2D([],[],color='#333333',marker='o',lw=2,label='실측 평균과 동시 95% 구간' if kfont else 'Held-out mean with simultaneous 95% CI')],
    loc='lower right',frameon=False,fontsize=9)
fig.suptitle('새 contact law에서 contour별 보정 효과  ·  음수는 optimizer 우세' if kfont
             else 'Correction effects under unseen contact laws (negative favors optimizer)',
             fontweight='bold')
fig.savefig(root/'v40.png',dpi=180,bbox_inches='tight')
