"""Measure sign-agreement covariance across distinct active objects as a diagnostic of common session drift."""
import json
import os
from pathlib import Path

import numpy as np
from world51 import KAPPA, full_states, marginal_retention

ROOT=Path(__file__).resolve().parent

def main():
    assert os.getenv('SLURM_JOB_ID')
    rng=np.random.default_rng(51037019)
    out=[]
    for q in KAPPA:
        n_session,n_active=8192,64
        size=(n_session,n_active)
        weak=rng.choice([1,2],size=size)
        ur,ue=rng.random((2,*size))
        ud=rng.random(n_session)
        for e,g in ((.25,0.),(0.,.25),(.5,0.),(0.,.5),(.25,.25)):
            _,impact,_=full_states(weak,np.full(n_session,q),ur,ud,ue,e,g)
            a=(weak==impact).sum(1)
            mean=marginal_retention(q,e,g)
            # Distinct objects counted with the U-statistic, centered at the known marginal.
            cov=(a*(a-1)/(n_active*(n_active-1))-2*mean*a/n_active+mean**2)
            se=cov.std(ddof=1)/np.sqrt(n_session)
            out.append(dict(kappa=q,epsilon=e,drift=g,n_session=n_session,n_active=n_active,
                            mean=cov.mean(),mc_se=se,analytic=g*(1-g)*(1-2*e)**2*(2*q-1)**2))
    (ROOT/'covariance51.json').write_text(json.dumps(out,indent=1))
if __name__=='__main__':main()
