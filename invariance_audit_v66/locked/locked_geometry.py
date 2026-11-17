"""Two analytic, anchored 2-D height fields. Inputs and output heights are mm."""
import numpy as np
FAMILIES=('wendland','dispersive')
VARIANTS=('base','hollow','bulge')

def raw(x, family, s):
    x=np.asarray(x,dtype=float)
    z=np.zeros_like(x)
    g=np.zeros_like(x)
    # Asymmetric neighboring features at the historical evaluation heights.
    for center,w,amp,dc,dw,da in ((18.55,5.1,.68,.65,.16,.25),
                                  (31.65,5.8,-.82,-.8,-.18,-.20)):
        width=w*(1+s*dw)
        amplitude=amp*(1+s*da)
        t=(x-center-s*dc)/width
        if family=='wendland':
            envelope=np.maximum(1-np.abs(t),0)
            f=envelope**4*(1+4*np.abs(t))
            deriv=-20*t*envelope**3
        elif family=='dispersive':
            f=2*t/(1+t*t)
            deriv=2*(1-t*t)/(1+t*t)**2
        else:
            raise ValueError(family)
        z+=amplitude*f
        g+=amplitude*deriv/width
    return z,g

def profile(x,family,variant):
    s={'base':0,'hollow':-1,'bulge':1}[variant]
    f,g=raw(x,family,s)
    z,d=raw(0,family,s)
    return f-z-d*np.asarray(x),g-d
