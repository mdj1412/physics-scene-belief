"""Independent sensing and assignment diagnostics for v36."""
import numpy as np
from math import erf, sqrt

def normal_cdf(x):
    return .5*(1+np.vectorize(erf,otypes=[float])(x/sqrt(2.)))

CANDIDATES=[(h,u) for h in (0.,.023,.025) for u in (.425,.50,.75,1.10)]
def response(theta, obj, surface, sign, scale, height, incoming):
    """Noise-free impulse in global coordinates for the held-out contact law."""
    _,_,slope,curve,mu,e=np.moveaxis(np.asarray(theta),-1,0)
    x=height/.025
    local=slope+curve*x+(-.045*np.sin(x) if obj=='sinusoid' else .04*x**3)
    norm=np.sqrt(1+local**2)
    vt=(incoming+local)/norm
    jn=(1+e)*np.maximum((1-local*incoming)/norm,0.)
    v=np.clip(abs(height)/.023,0.,1.)
    local_scale=1+(scale-1)*v**2*(3-2*v)
    modulation=(.21*(2/np.pi)*np.arctan(vt/.4) if surface=='atan' else .16*vt/np.sqrt(.30**2+vt**2))
    eff=mu*(1+sign*local_scale*modulation)
    stick=abs(vt)<=eff*jn
    jt=np.clip(vt,-eff*jn,eff*jn)
    return np.stack([1-(jn+jt*local)/norm,incoming+(jn*local-jt)/norm],axis=-1),stick

def sensor_limits(theta,mu_archive,obj,surface,sgn):
    # theta: E,6. mu_archive: 3,E,2 in strong/weak/null order.
    pairs=((0,2),(0,1),(1,2))
    out=[]
    for height,speed in CANDIDATES:
        ticks=[response(theta,obj,surface,sgn,amp,height,speed) for amp in (1.,.3,0.)]
        mus=np.stack([p[0] for p in ticks])
        if (height,speed)==(.023,.50):
            np.testing.assert_allclose(mus,mu_archive,atol=2.e-10,rtol=0)
        measures=[]
        for i,j in pairs:
            sep=np.linalg.norm((mus[i]-mus[j])/.027,axis=-1)
            bothstick=ticks[i][1]&ticks[j][1]
            # Same Gaussian noise, paired latent and signs given to diagnostic.
            measures.append(np.stack([normal_cdf(.5*sep),np.asarray(sep<1e-10,float),bothstick.astype(float)],axis=-1))
        out.append(measures)
    return np.array(out)
