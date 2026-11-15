"""Evaluate the replaced contact at the commanded height and tangential speed."""
from target36 import MATCH_HEIGHT, MATCH_SPEED
from detector34 import ActionConditionedDetector
from estimator28 import DT, UNIT_DIM, t, torch, parameters, change_proxy, factual_mean, impact
import numpy as np

class Detector(ActionConditionedDetector):
    def __init__(self,seed=3003913,bank_size=8192):
        super().__init__(seed,bank_size)
        half=bank_size//2
        unit=torch.quasirandom.SobolEngine(UNIT_DIM,scramble=True,seed=seed).draw(half,dtype=DT)
        unit[:,1]*=.5
        other=unit.clone()
        other[:,1]+=.5
        theta=parameters(t(torch.cat([unit,other])))
        centers=[]
        for code in range(4):
            particles=change_proxy(theta,torch.full((bank_size,),code,dtype=torch.uint8,device=theta.device))
            probe,_=impact(particles,MATCH_HEIGHT,MATCH_SPEED)
            centers.append(torch.cat((factual_mean(particles)/self.sigma,probe/self.probe_sigma),dim=1))
        self.near_centers=torch.stack(centers)
        self.near_energy=.5*self.near_centers.square().sum(-1)

    def fit_actions(self,record,near):
        x,ids,y,d=record
        ll=self.component_loglikelihood(x)
        if near:
            far=self.signed_joint(x[ids[:3]],y[:3],d[:3])
            raw=torch.cat((t(x[ids[3:]])/self.sigma,t(y[3:])/self.probe_sigma),-1)
            logp=raw@self.near_centers.transpose(1,2)-self.near_energy[:,None,:]
            fourth=(torch.logsumexp(logp,-1).T-np.log(self.bank_size)).cpu().numpy()
            joint=np.concatenate((far,fourth))
        else:
            joint=self.signed_joint(x[ids],y,d)
        return self.posterior_from_components(ll,ids,joint)
