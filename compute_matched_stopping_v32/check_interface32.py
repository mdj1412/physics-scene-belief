"""Check the extra target law and deployment decoder against established source APIs on CPU."""
import numpy as np
from evaluate32 import (Engine, Q, P, calibration, decode_selected, deployment_x, deployment_truth,
                        torch, allocate, decisions)
from contact30 import factual, draw_theta, means, sigmas
from target32 import signals
from quota_latency29 import decode

def main():
    assert not torch.cuda.is_available()
    torch.set_num_threads(1)
    for i,alpha in enumerate([[-.65,-.85],[.72,.92]]):
        seed=407000000+i
        x=deployment_x(seed,alpha,'directional')
        np.testing.assert_allclose(x,factual(len(x),seed,20,alpha,'coulomb'),atol=1.e-12,rtol=0)
        _,ids,yy=calibration(seed,alpha,'directional')
        assert len(set(ids))==8 and np.isfinite(yy).all()
        target,nominal,obs=deployment_truth(seed,alpha,'directional')
        theta=draw_theta(128,seed,20,alpha)
        np.testing.assert_allclose(signals(theta)[0].numpy(),means(theta,'coulomb')[0].numpy(),atol=1.e-12)
        print('directional preflight',i,'changed labels',int((target!=nominal).sum()),'of',len(x))
        a=np.where(np.arange(128)%9==0,P,Q)
        dec=Engine([407010100+i*2,407010101+i*2],*alpha)
        ys={k:obs[k][:,0] for k in (Q,P)}
        fast=decode_selected(dec,x,a,ys)
        idx=(0,1,2,9,22,127)
        scalar=np.array([decode(dec,x[j],int(a[j]),ys[a[j]][j]) for j in idx])
        np.testing.assert_allclose(fast[list(idx)],scalar,atol=1.e-9,rtol=0)
        print('selected-probability max error',np.max(np.abs(fast[list(idx)]-scalar)))
    print('Passed target and selected-observation interface checks')
if __name__=='__main__':
    main()
