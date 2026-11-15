from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"factual_tail_v23"))
from physics23 import PhysicalBank, impact, noiseless, FY, t, torch, DT
from common import R, save
import numpy as np

# factual profile, factual velocity, query view, factual view, probe (standard deviations).
CAL={"dev_camera":(.0021,.12,.00040,.0010,.060), "dev_probe":(.0033,.18,.00130,.00075,.023),
     "camera":(.0018,.15,.00030,.0012,.075), "probe":(.0037,.11,.00165,.00055,.018),
     "balanced":(.0027,.21,.00076,.00090,.042)}
DEV_CALS=("dev_camera","dev_probe")
TEST_CALS=("camera","probe","balanced")

def parameters(u):
    ci=(3*u[:,0]).long().clamp_max(2)
    fi=(u[:,1]>=.5).long()
    curvature=(ci-1)*.29+.12*(u[:,2]-.5)
    friction=.12+.20*fi+.075*(u[:,3]-.5)
    return torch.stack([ci*2+fi,.008*(u[:,4]-.5),.36*(u[:,5]-.5),
                        curvature,friction,.56+.28*u[:,6]],dim=1).to(DT)

def sigmas(cal):
    p,v,q,f,r=CAL[cal]
    return [t([p]*9+[v]*2),t([q]*7),t([f]*7),t([r]*2)]

class FactorBank(PhysicalBank):
    def __init__(self,seed,cal):
        self.N=16384
        u=torch.quasirandom.SobolEngine(7,scramble=True,seed=seed).draw(self.N,dtype=DT).to(t(0).device)
        self.theta=parameters(u)
        self.sigma=sigmas(cal)
        self.mu=dict(enumerate(noiseless(self.theta)))
        self.white={a:m/self.sigma[a] for a,m in self.mu.items()}
        self.norm={a:.5*m.square().sum(-1) for a,m in self.white.items()}
        self.label=impact(self.theta,.025,.425)[1].to(DT)[:,None]*np.sqrt(2)

def generate(split,n,seed,cals):
    assert n%6==0
    rng=np.random.default_rng(seed);u=rng.random((n,7))
    cells=rng.permutation(np.tile(np.arange(6),n//6))
    u[:,0]=(cells//2+u[:,0])/3;u[:,1]=(cells%2+u[:,1])/2
    theta=parameters(t(u));means=noiseless(theta)
    assert np.array_equal(theta[:,0].cpu().numpy(),cells)
    data=dict(label=impact(theta,.025,.425)[1].cpu().numpy(),cell=cells,theta=theta.cpu().numpy())
    xs=[]
    for a,mu in enumerate(means):
        eps=t(rng.normal(size=(n,1 if a==0 else 32,mu.shape[-1])))
        obs=torch.stack([mu[:,None]+eps*sigmas(cal)[a] for cal in cals],dim=1).cpu().numpy()
        if a==0:xs=obs[:,:,0,:]
        else:data[f"obs{a}"]=obs
    save(R/split/"public.npz",x=xs);save(R/split/"private.npz",**data)
