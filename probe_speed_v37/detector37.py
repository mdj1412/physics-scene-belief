"""Use the same Coulomb particles for every commanded fourth speed."""
import numpy as np
from detector36 import Detector
from estimator28 import DT, UNIT_DIM, t, torch, parameters, change_proxy, factual_mean, impact

class SpeedDetector(Detector):
    def __init__(self,speeds,height,seed=3003913,bank_size=8192):
        super().__init__(seed,bank_size)
        self.near_banks={.5:(self.near_centers,self.near_energy)}
        half=bank_size//2
        unit=torch.quasirandom.SobolEngine(UNIT_DIM,scramble=True,seed=seed).draw(half,dtype=DT)
        unit[:,1]*=.5
        upper=unit.clone()
        upper[:,1]+=.5
        theta=parameters(t(torch.cat((unit,upper))))
        for speed in speeds:
            if speed in self.near_banks: continue
            centers=[]
            for code in range(4):
                particles=change_proxy(theta,torch.full((bank_size,),code,dtype=torch.uint8,device=theta.device))
                vel,_=impact(particles,height,speed)
                centers.append(torch.cat((factual_mean(particles)/self.sigma,vel/self.probe_sigma),1))
            centers=torch.stack(centers)
            self.near_banks[speed]=(centers,.5*centers.square().sum(-1))

    def use_speed(self,speed):
        self.near_centers,self.near_energy=self.near_banks[speed]
