"""Label-free switch sensitivity to the mean error prior of the prospective planning model."""
import json
import os
import numpy as np
from policy54 import ROOT,tables,likelihood_grid,PRIORS,RETENTIONS
from routes54 import raw_prefix

MEANS=np.arange(10,26)/100

def main():
    assert os.getenv('SLURM_JOB_ID')
    grid,ir,ii,hist,scale,_,_,_=tables()
    counts=[]
    for panel in range(12):
        weak,ref=raw_prefix(panel)
        a=likelihood_grid(weak,ref,'likelihood')*PRIORS
        a/=a.sum(-1,keepdims=True)
        p=a@hist.T
        m=(a*RETENTIONS)@hist.T/p
        s=(a*RETENTIONS**2)@hist.T/p
        vk=s-m**2
        br=vk**2*np.interp(m,grid,ir)
        row=[]
        for mu in MEANS:
            m2=2*mu**2/(1+2*mu)
            vq=(1-2*mu)**2*vk+(m2-mu**2)*(1-4*m+4*s)
            qi=mu+(1-2*mu)*m
            dr=(p*((1-2*mu)**2*br-vq**2*np.interp(qi,grid,ii))).sum(-1)
            row.append(np.mean(dr<0))
        counts.append(row)
    result=dict(prior_mean=MEANS.tolist(),post_impact_rate=np.mean(counts,axis=0).tolist(),
                prior='e/.5 ~ Beta(1, .5/mu-1); terminal and assumed truth change together in prospective model')
    print(result)
    (ROOT/'priors54.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
