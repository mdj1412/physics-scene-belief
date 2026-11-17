"""Plot camera readout and paired decision cost using the recorded v60 results."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
optical = np.asarray(json.loads((root / 'vision60.json').read_text())['total_active_read_wrong_invalid'])
cross = json.loads((root / 'transfer60.json').read_text())['crossings']
colors = ['#25747a', '#b55832']
fig, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4), layout='constrained')
x = np.arange(3)
for i, label in enumerate(['Fixed frame window', 'LED-aligned window']):
    denominator = optical[:, i, 1]
    pct = 100 * optical[:, i, 2] / denominator
    dx = (-.12, .12)[i]
    left.scatter(x + dx, pct, s=56, color=colors[i], label=label, zorder=3)
    for j, y in enumerate(pct):
        left.annotate(f'{y:.1f}%', (x[j] + dx, y), ha='center',
                      xytext=(0, 9 if i==0 else -14), textcoords='offset points', fontsize=8)
left.set(xticks=x, xticklabels=['0%', '20%', '100%'], xlim=(-.42,2.45), ylim=(-12,113),
         ylabel='Mismatch error among readable pairs (%)', xlabel='Share of impact clips delayed by 2 ms',
         title='A   Optical readout')
left.legend(frameon=False, fontsize=8, loc='center left')
alpha=np.linspace(0.,1.,101)
for i, (key,label) in enumerate([('gate','Reset − always v54 gate'),('sham','Reset − sham, both with camera')]):
    arm=cross[key]
    risk=1e6*((1-alpha)*arm['reversed']+alpha*arm['matched'])
    right.plot(alpha, risk, color=colors[i], lw=2.4, label=label)
    right.plot(arm['cross'],0,'o',color=colors[i],ms=5)
    right.annotate(f"{arm['cross']:.2f}",(arm['cross'],0),xytext=(3,8 if i==0 else -14),textcoords='offset points',fontsize=8)
right.axhline(0,color='#424242',lw=.7)
right.axvline(.5,color='#646464',ls=':',lw=1)
right.set(xlim=(-.02,1.02),ylim=(-.4,1.23),xlabel='Probability of source-matched fault assignment (α)',
          ylabel='Mean held-out Brier difference (×10⁻⁶)',title='B   Policy transfer after LED alignment')
right.legend(frameon=False,fontsize=8,loc='upper right')
for ax in [left,right]:
    title = ax.get_title()
    ax.set_title('')
    ax.set_title(title,loc='left',weight='bold',fontsize=12,pad=10)
    ax.grid(axis='y',color='#dddddd',lw=.55)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
fig.savefig(root/'trigger_fault60.png',dpi=180)
