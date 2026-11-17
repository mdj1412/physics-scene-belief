"""Forecast and observed value of spending the last slot on impact instead of reference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
p=Path(__file__).resolve().parent
q=json.loads((p/'score54.json').read_text())['quartiles']
model=np.array([v['predicted_delta'] for v in q])*1e5
obs=-np.array([v['observed_delta'] for v in q])*1e5
xs=np.arange(1,5)
fig,ax=plt.subplots(figsize=(7.4,3.75))
ax.plot(xs-.04,model,'o--',color='#5b677a',label='Gain predicted before purchase')
ax.errorbar(xs+.04,obs[:,0],yerr=np.stack((obs[:,0]-obs[:,2],obs[:,1]-obs[:,0])),fmt='o-',color='#007974',capsize=3,label='Held-out gain after purchase')
ax.axhline(0,color='#444',lw=.85)
ax.set(xticks=xs,xlabel='Quartile of predicted impact advantage (low to high)',ylabel='Brier gain of impact over reference (×10⁻⁵)')
ax.grid(axis='y',alpha=.2)
ax.spines[['right','top']].set_visible(False)
ax.legend(frameon=False,loc='upper left')
fig.tight_layout()
fig.savefig(p/'prefix_value54.png',dpi=180)
