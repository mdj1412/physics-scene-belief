"""Describe support and direction from the figure independent of model predictions."""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

folder=Path(__file__).resolve().parents[1]
with (folder/'data'/'feint_fig12_paired.csv').open() as f:
    records=list(csv.DictReader(f))
n=np.array([float(r['n_in_rps']) for r in records])
vx=np.array([float(r['u_out_mps']) for r in records])
nout=np.array([float(r['n_out_rps']) for r in records])
rim=.019*2*np.pi*nout
s=vx-rim
paths=sorted((folder.parent/'racket_transfer_v70'/'data'/'upstream'/'train').glob('racket_*.csv'))
source=np.concatenate([np.genfromtxt(p,delimiter=',',names=True) for p in paths])
x=np.stack([source['v_in_x'],source['v_in_z'],.02*source['spin_before_y']],axis=1)
target=np.stack([np.zeros(len(n)),np.full(len(n),-30/3.6),.019*2*np.pi*n],axis=1)
sd=x.std(0); tree=cKDTree(x/sd)
dsrc=tree.query(x/sd,k=2)[0][:,1]
dtgt=tree.query(target/sd)[0]
q95=np.quantile(dsrc,.95)
out={'source_n':len(x),'source_min':x.min(0).tolist(),'source_max':x.max(0).tolist(),
     'source_abs_spin_max_rps':float(np.max(np.abs(source['spin_before_y']))/(2*np.pi)),
     'source_pair_distance_q95':float(q95),'nearest_neighbor_note':'Euclidean distance after scaling by source column SD; source vs other nearest source impact, target vs any source impact',
     'outside_source_range_per_axis':((target<x.min(0))|(target>x.max(0))).sum(0).tolist(),
     'target_count_distance_above_q95':int(sum(dtgt>q95)),
     'points':[{'id':r['id'],'n_in_rps':float(a),'release_mps':float(b),'nearest_source':float(c)} for r,a,b,c in zip(records,n,s,dtgt)]}
(folder/'analysis'/'external_support.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10.2,3.9))
axes[0].scatter(n,vx,label="Tangential speed $u'$",color="#09796d",marker='o')
axes[0].scatter(n,rim,label="Spin at ball surface $R\\omega'$",color="#965629",marker='s')
axes[0].set(ylabel="Measured output (m/s)",title="Feint OX · matched figure points")
axes[0].legend(frameon=False,fontsize=9)
axes[1].axhspan(-.20,.20,color="#d6dbe2",alpha=.65,label="±0.20 m/s sensitivity band")
axes[1].axhline(0,color="#555a60",lw=.8)
axes[1].scatter(n,s,c=np.where(s>.20,'#305e9c',np.where(s<-.20,'#ad4e30','#707780')),zorder=3)
axes[1].set(ylabel="Signed release $u'-R\\omega'$ (m/s)",title="Release changes sign within one product")
axes[1].legend(frameon=False,fontsize=9,loc="upper left")
for ax in axes:
    ax.set_xlabel("Incident spin (rev/s)")
    ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='y',alpha=.18)
fig.tight_layout()
fig.savefig(folder/'analysis'/'feint_release.png',dpi=175)
