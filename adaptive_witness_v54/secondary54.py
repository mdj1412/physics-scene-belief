"""Secondary paired fixed-comparator intervals for frozen non-primary sensor choosers."""
import json
import os
import numpy as np
from score54 import ROOT,V53,N,R,KPOST,GM,METHODS,pair_ci,quantiles,bootstrap,catalog,leave_out_bins

def main():
    assert os.getenv('SLURM_JOB_ID')
    panels=[catalog(i) for i in range(N)]
    exact=np.stack([x[0][0] for x in panels])
    def expand(x):
        return np.broadcast_to(x[:,None,KPOST],exact[...,0].shape)
    b=expand(np.stack([x[3] for x in panels]))
    lookup=leave_out_bins(exact,b)
    robust,uniform=[expand(np.stack([x[2][k] for x in panels])) for k in ('robust','uniform')]
    with np.load(V53/'session53.npz') as f:
        names=list(f['methods'].astype(str)); idx=[names.index(n) for n in ('reference4_joint','impact_joint','reference4_uniform','impact_joint_uniform','impact_joint_noisy')]
        r,i,ur,ui,ni=np.moveaxis(f['realized'][...,idx],-1,0)
    cube=np.stack((r,i,ur,ui,np.where(robust,i,r),np.where(uniform,ui,ur),np.where(lookup,i,r),np.where(uniform,i,r)),axis=-1)
    q=quantiles(cube,GM); bs=bootstrap(cube)
    with np.load(ROOT/'catalog54.npz') as f: mu=f['means'][:,0]
    keys=['fixed4R','fixed3RI','uniform4R','uniform3RI','robust_gate','adaptive_uniform_gate','sign_lookup_loo','uniform_gate_nominal_decoder']
    means=mu[:,[METHODS.index(k) for k in keys]]
    out={}
    for m,a,bb in ((4,0,1),(5,2,3),(6,0,1),(7,0,1)):
        contrast={}
        for j in (a,bb):
            contrast[keys[j]]=dict(mean=pair_ci(means[:,m]-means[:,j],4).tolist(),
                p90=[float(q[m]-q[j]),*np.quantile(bs[:,m]-bs[:,j],[.025/4,1-.025/4]).tolist()])
        out[keys[m]]=contrast
    # Freeze routes while changing only the purchased impact sensor to the 10%-error calibrated reader.
    ex=np.stack([np.where(s,exact[...,2],exact[...,0]) for s in (robust,uniform,lookup)],-1)
    noisy_q=quantiles(np.stack([np.where(s,ni,r) for s in (robust,uniform,lookup)],-1),GM)
    noisy_m=((1-GM)*ex[:,:,0]+GM*ex[:,:,1]).mean((0,1,2,3,4))
    out['noisy_routes']=dict(methods=['robust','uniform','lookup'],mean=noisy_m.tolist(),p90=noisy_q.tolist())
    (ROOT/'secondary54.json').write_text(json.dumps(out,indent=1));print(out)
if __name__=='__main__':main()
