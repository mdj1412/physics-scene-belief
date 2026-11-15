import json,platform,time
from pathlib import Path
import numpy as np
import torch
from engine import *
from inference import smc,predict,acquisition
from sensor import observe

def main():
    out=Path('results/validation');out.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(72);x=tensor(rng.random((512,4)));theta=decode(x);theta[:,2]=0
    result=simulate(theta,tensor(np.zeros(512)),'corner')
    # Corner front-face/quarter-circle closed form from v3.
    offset=.014+theta[:,0];u=(offset/R).clamp(0,1);a=1-u*u
    exact=torch.stack([1-(1+theta[:,1])*a,-(1+theta[:,1])*torch.sqrt(a)*u],1)
    err=float((result['velocity']-exact).abs().max());assert err<1e-8,err
    evidence={'host':platform.node(),'gpu':torch.cuda.get_device_name(),'analytic_max_error':err,'families':{}}
    for family in FAMILIES:
        th=decode(tensor(rng.random((1024,4))));b=tensor(rng.choice([0,.001,.003],1024))
        sim=batch(th,b,family,SHIFTS)
        assert not bool(sim['overflow'].any()),family
        kinetic=(sim['velocity']**2).sum(2)
        assert float(kinetic.max())<=1+1e-8
        codes,counts=torch.unique(sim['code'],return_counts=True)
        truth=th[0].cpu().numpy();rounding=float(b[0]);measurements={}
        for action in ['factual']+ACTIONS:measurements[action]=observe(truth,rounding,family,action,rng,out/family).tolist()
        start=time.perf_counter();atoms,stats=smc(family,[('factual',measurements['factual'],1.)],n=256)
        pred=predict(atoms,family);gains,eig=acquisition(atoms,family,pred,samples=64)
        evidence['families'][family]=dict(contact_codes=dict(zip(codes.cpu().tolist(),counts.cpu().tolist())),max_energy=float(kinetic.max()),measurements=measurements,smc=stats,policy=gains.argmax(1).tolist(),eigs=eig.tolist(),seconds=time.perf_counter()-start)
    (out/'validation.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence,indent=2))
if __name__=='__main__':main()
