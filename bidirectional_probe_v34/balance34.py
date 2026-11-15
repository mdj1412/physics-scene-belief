import numpy as np
from decisions34 import ROOT,OUT,sample,band,dump
from direct_head34 import FAMILIES,SCHEDULES
def main():
    v=np.empty((4,8,2,2,3))
    for f in range(4):
        for b in range(8):
            for d,o in enumerate(('plus','minus')):
                for a,s in enumerate(SCHEDULES):
                    with np.load(OUT/(FAMILIES[f]+'_'+o)/f'block_{b:03d}'/s/'metrics.npz') as z:
                        v[f,b,d,a]=z['brier'].mean(0)
    # Align d=0 to initially low friction, d=1 to initially high friction.
    z=v.copy();z[:,1::2]=z[:,1::2,::-1]
    raw=z[:,:,:,1]-z[:,:,:,0]
    def analysis(x):
        avg=x.mean(1)
        pick=np.argmin(avg[::-1],axis=2)
        routed=np.take_along_axis(x,pick[:,None,:,None,:],axis=3)[:,:,:,0]
        rel=x[:,:,:,1]-x[:,:,:,0]
        return np.r_[rel.mean((0,1)).flatten(), (routed-x[:,:,:,0]).mean((0,1,2)),(routed-x[:,:,:,1]).mean((0,1,2))]
    pt=analysis(z);idx=sample(np.random.default_rng(342303),8000)
    draws=np.array([analysis(z[np.arange(4)[:,None],i]) for i in idx]);ci=band(pt,draws)
    choice=np.argmin(z.mean(1)[::-1],axis=2)
    dump('balance_stats34.json',dict(initial_low_then_high=pt[:6].reshape(2,3),simultaneous95=ci,
          oracle_routed_minus_same=pt[6:9],oracle_routed_minus_alt=pt[9:],selected_alt_by_family_condition_method=choice))
if __name__=='__main__':main()
