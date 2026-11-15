"""Infer population coupling with a signed incoming velocity in the Coulomb likelihood."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'calibration_probe_v28'))
from estimator28 import ProbeShiftEstimator, DT, UNIT_DIM, t, torch, parameters, change_proxy, factual_mean, impact, PROBE_INCOMING

class ActionConditionedDetector(ProbeShiftEstimator):
    @torch.inference_mode()
    def __init__(self,seed=3003913,bank_size=8192):
        super().__init__(seed=seed,bank_size=bank_size)
        half=bank_size//2
        unit=torch.quasirandom.SobolEngine(UNIT_DIM,scramble=True,seed=self.seed).draw(half,dtype=DT)
        lower=unit.clone();lower[:,1]*=.5
        upper=lower.clone();upper[:,1]+=.5
        theta=parameters(t(torch.cat([lower,upper])))
        positive=[]
        for code in range(4):
            particles=change_proxy(theta,torch.full((bank_size,),code,dtype=torch.uint8,device=theta.device))
            fact=factual_mean(particles)/self.sigma
            neg,ns=impact(particles,0,PROBE_INCOMING)
            np.testing.assert_allclose(self.joint_centers[code].cpu().numpy(),
                                       torch.cat([fact,neg/self.probe_sigma],dim=1).cpu().numpy(),rtol=0,atol=1e-10)
            pos,ps=impact(particles,0,-PROBE_INCOMING)
            if bool(ns.any()|ps.any()):
                raise ArithmeticError("Calibration isotropic particles should slide")
            positive.append(torch.cat([fact,pos/self.probe_sigma],dim=1))
        pos=torch.stack(positive)
        self.intervention_centers={-1:self.joint_centers,1:pos}
        self.intervention_energy={-1:self.joint_energy,1:.5*pos.square().sum(-1)}
    @torch.inference_mode()
    def signed_joint(self,x,y,directions):
        signs=np.asarray(directions)
        if signs.shape!=(len(x),) or not np.isin(signs,[-1,1]).all():
            raise ValueError("Missing signed action for paid response")
        raw=torch.cat((t(x)/self.sigma,t(y)/self.probe_sigma),dim=-1)
        result=np.empty((len(x),4))
        for sign in (-1,1):
            rows=np.where(signs==sign)[0]
            if len(rows):
                logp=raw[rows]@self.intervention_centers[sign].transpose(1,2)-self.intervention_energy[sign][:,None,:]
                result[rows]=(torch.logsumexp(logp,-1).T-np.log(self.bank_size)).cpu().numpy()
        return result
