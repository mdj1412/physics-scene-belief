"""Plot pre-outcome forecasts, held-out effects, and individual catalog estimates."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager

korean_fonts = ('Apple SD Gothic Neo', 'Noto Sans CJK KR', 'NanumGothic')
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((family for family in korean_fonts if family in available), None)
if font:
    plt.rcParams['font.family'] = [font, 'DejaVu Sans']

root=Path(__file__).resolve().parent
data=json.loads((root/'stats39.json').read_text())
fig,ax=plt.subplots(figsize=(8.7,4.1),layout='constrained')
ax.axvline(0,ls='--',lw=1,color='#656565',zorder=0)
for row,(key,stats) in enumerate(data['by_cell'].items()):
    x=np.asarray(stats['catalog_means'])*1000
    offsets=np.random.default_rng(39035+row).permutation(np.linspace(-.16,.16,len(x)))
    ax.scatter(x,row+offsets,s=14,facecolors='#aaa6a1',edgecolors='none',alpha=.85,zorder=2)
    lo,hi=np.array(stats['sim95'])*1000
    ax.hlines(row,lo,hi,color='#216870',lw=2.6,zorder=3)
    ax.scatter(1000*stats['mean'],row,s=61,marker='o',c='#216870',zorder=4)
    ax.scatter(1000*stats['forecast'],row-.20,marker='D',s=35,facecolors='white',edgecolors='#a14c24',linewidths=1.7,zorder=5)
ax.set_yticks(range(4),[k.replace('_',' / ').capitalize() for k in data['cells']])
ax.set_ylim(3.4,-.55)
ax.set_xlabel('Brier 차이 × 1,000     (음수일수록 optimizer 우세)' if font else 'Optimizer minus fixed Brier × 1,000     (negative favors optimizer)')
ax.set_title('새 catalog에서 사전 예측과 실제 Brier 차이' if font else 'Did the v38 forecast predict the direction on new catalogs?',loc='left',fontweight='bold')
ax.spines[['left','top','right']].set_visible(False)
ax.tick_params(axis='y',length=0)
ax.grid(axis='x',color='#e5e5e5',lw=.6,zorder=0)
ax.legend(handles=[Line2D([],[],color='#a14c24',marker='D',markerfacecolor='white',linestyle='none',label='사전 예측' if font else 'Frozen forecast'),
                   Line2D([],[],color='#216870',lw=2.5,marker='o',label='실측 평균과 동시 95% 구간' if font else 'Held-out mean and simultaneous 95% interval'),
                   Line2D([],[],color='#aaa6a1',marker='o',ms=4,linestyle='none',label='Catalog')],loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=8)
fig.savefig(root/'v39.png',dpi=175,bbox_inches='tight')
