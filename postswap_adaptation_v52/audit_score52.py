"""Practical and alternative-explanation audits at four post-swap witnesses."""
import json
import os
import numpy as np
from score52 import ROOT, BUDGET, DRIFT, EPS, N, NREP, METHODS, P, GMEAN, ci,bootstrap_primary_p90

assert os.getenv('SLURM_JOB_ID')
with np.load(ROOT/'catalog52.npz') as f:
    exact=f['exact'][:,0,:,:,:,:,P]; surfaces=f['selected_surface']
with np.load(ROOT/'session52.npz') as f: real=f['realized']
mix=(1-GMEAN)*exact[:,:,0]+GMEAN*exact[:,:,1]
avg=mix.mean((1,2,3))
d=avg[:,0,None]-avg[:,[5,6]]
extras=ci(d,4)
q=json.loads((ROOT/'score52.json').read_text())['absolute_p90'][P]
qchk=bootstrap_primary_p90(real[..., [0,5,6]])
quant=np.column_stack((q[0]-np.array(q)[[5,6]],qchk))
surface={}
for sf in np.unique(surfaces):
    az=np.stack([mix[i,:,:,surfaces[i]==sf].mean((0,1,2)) for i in range(N)])
    surface[str(sf)]=dict(means=az.mean(0),mean_diff=ci(az[:,0,None]-az[:,[1,2]],4))
# A realized independent deployment of 48 sessions. One state set/outcome per object,
# simultaneous operating conditions balanced as far as possible for κ, exactly for (e,g).
fleet=np.empty((N,NREP,len(METHODS))); sam=np.empty_like(fleet)
rng=np.random.default_rng(520331)
for r in range(NREP):
    ix=rng.permutation(np.tile(np.arange(12),4)).reshape(12,4)
    e=ix//4;g=np.asarray(DRIFT)[ix%4]
    ev=(rng.random((12,4))<g).astype(int)
    k=rng.permutation(np.r_[np.tile(np.arange(5),9),rng.choice(5,3,replace=False)]).reshape(12,4)
    batch=real[np.arange(12)[:,None],e,ev,k,np.arange(4)[None,:],r,P,:]
    fleet[:,r]=batch.mean(1)
    sam[:,r]=np.quantile(batch,.9,axis=1)
    if r==0:
        first_q=np.quantile(batch.reshape(-1,len(METHODS)),.9,axis=0)
    # Store each run's 48-session p90 rather than resampling individual objects.
    if r==0: q_runs=np.empty((NREP,len(METHODS)))
    q_runs[r]=np.quantile(batch.reshape(-1,len(METHODS)),.9,axis=0)
fm=fleet.mean(0)
de=fm[:,0,None]-fm[:,[1,2,5,6]]
dq=q_runs[:,0,None]-q_runs[:,[1,2,5,6]]
out=dict(methods=METHODS,holdout_surface=surface,comparator_extra=['unsigned','continuous_joint'],
         exact_diff_to_extra=extras,p90_diff_to_extra=quant,fleet_comparators=[METHODS[i] for i in (1,2,5,6)],
         fleets=NREP,sessions_per_fleet=48,mean_delta=fm.mean(0),fleet_mean_worse=(de>0).mean(0),
         fleet_p90_worse=(dq>0).mean(0),fleet_first_mean=fm[0],fleet_first_p90=first_q,
         scope='Secondary finite-fleet check conditional on the same fixed 48 environments. No new material/session populations. t-CI paired by 12 catalogs for mean, percentile/5000 for pooled p90.')
(ROOT/'audit_score52.json').write_text(json.dumps(out,default=lambda x:np.asarray(x).tolist(),indent=1))
