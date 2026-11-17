"""Diagnose sensitivity to numerical integration against a second archived contact solution."""
import json
import os
import numpy as np
from score53 import ROOT,V50,N,G,pair_ci,quantiles,bootstrap

assert os.getenv('SLURM_JOB_ID')
records=[]; before=[]; after=[]; margins=[]
for i in range(N):
    file=f'panel_{i:02d}.npz'
    with np.load(V50/'labels50'/file) as p, np.load(V50/'labels50_ultra'/file) as q:
        y=p['labels']; z=q['labels']
        assert y.shape==z.shape and np.array_equal(p['object_index'],q['object_index'])
        records.append(y!=z);before.append(y);after.append(z);margins.append(np.abs(p['slip_speed'][y!=z]-.020))
        dt=[float(p['dt_ms']),float(q['dt_ms'])];edge=[float(p['edge_mm']),float(q['edge_mm'])]
        sign=p['state'].astype(int).tolist();family=p['families'].astype(str).tolist()
        radius=p['radii_mm'].tolist()
x=np.stack(records); y=np.stack(before); z=np.stack(after)
with np.load(ROOT/'catalog53.npz') as f:
    names=list(f['methods'].astype(str)); exact=f['exact']
def idx(s):return names.index(s)
w=((1-G.mean())*exact[:,:,:,0]+G.mean()*exact[:,:,:,1]).mean((2,3,4))
delta = w[:,:,idx('impact_joint'),None]-w[:,:,[idx(k) for k in ['impact_selector','impact_vote_selector']]]
with np.load(ROOT/'session53.npz') as f:
    same_prior=f['realized'][...,[idx('impact_joint_uniform'),idx('reference4_uniform')]]
q=quantiles(same_prior,G.mean()); qs=bootstrap(same_prior)
ciq=np.quantile(qs[:,0]-qs[:,1],[.025,.975])
out=dict(total=int(x.size), changed=int(x.sum()), rate=x.mean(),primary_stick=y.mean(),ultra_stick=z.mean(),
         primary_stick_to_slip=int((y&~z).sum()), primary_slip_to_stick=int((~y&z).sum()),
         flip_by_state=dict(zip(sign,x.mean((0,1,2,4,5,6,7)))), flip_by_family=dict(zip(family,x.mean((0,2,3,4,5,6,7)))),
         flip_by_radius=dict(zip(radius,x.mean((0,1,2,3,5,6,7)))),changed_abs_margin=np.quantile(np.concatenate(margins),[.1,.5,.9]),
         steps_ms=dt, mesh_mm=edge, iterations=[80,160],contrast_names=['impact_joint minus impact_selector','impact_joint minus impact_vote_selector'],
         gains_catalog_count=(delta<0).sum(0),ultra_minus_primary_delta_ci=pair_ci(delta[:,1]-delta[:,0],2),
         count_catalogs_where_contrast_shifts_toward_joint=((delta[:,1]-delta[:,0])<0).sum(0),
         reference_substitution_under_uniform=dict(mean=pair_ci(w[:,0,idx('impact_joint_uniform')]-w[:,0,idx('reference4_uniform')],1),
                                                  p90=[q[0]-q[1],*ciq]))
(ROOT/'resolution53.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
print(out)
