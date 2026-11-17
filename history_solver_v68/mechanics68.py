"""Matched sensor velocity and slip for the prescribed and compliant contact solvers."""
import ctypes
import fcntl
import hashlib
from functools import lru_cache
from pathlib import Path
import os
import subprocess
import sys
import numpy as np

H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'invariance_audit_v66'/'locked'))
from locked_geometry import profile
A=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
RADII=(.25,2.)
OFF=(-.5,0.,.5)

@lru_cache(None)
def kernel():
    src=H/'compliant68.cpp'
    lib=H/('compliant68_'+hashlib.sha256(src.read_bytes()).hexdigest()[:12]+'.so')
    with (H/'compiler.lock').open('a') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        if not lib.exists():
            tmp=lib.with_suffix('.tmp')
            subprocess.run(['g++','-O3','-std=c++11','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
            tmp.replace(lib)
    fun=ctypes.CDLL(str(lib)).compliant68
    fun.argtypes=[ctypes.c_int]*3+[A]*7
    fun.restype=None
    return fun

@lru_cache(None)
def grid(backend,h,r,family,variant,fine):
    if backend=='compliant':
        dx=np.linspace(-r,r,17 if fine else 9)
        weights=None
    elif backend=='prescribed':
        x,wx=np.polynomial.legendre.leggauss(24 if fine else 12)
        z,wz=np.polynomial.legendre.leggauss(12 if fine else 6)
        dx=(r*np.sin(np.pi*(z+1)/4)[:,None]*x[None]).ravel()
        weights=(wz[:,None]*.5*.75*wx*(1.-x*x)).ravel()
    else: raise ValueError(backend)
    yy=h+dx
    if family=='flat':
        residual=np.zeros_like(dx); deriv=np.zeros_like(dx); center=0.
    else:
        f,deriv=profile(yy,family,variant)
        f0,center=profile(h,family,variant)
        residual=-f+f0+center*dx
    return dx,yy,weights,residual,deriv,float(center)

def as_row(value,n,dtype=float):
    return np.broadcast_to(np.asarray(value,dtype=dtype),(n,))

def resist(theta,vt,h,surface,law,sign,regime):
    if surface=='flat':scale=1.;orient=0.
    elif surface=='inlay':
        k=np.exp(-.5*((h-22.)/5.5)**2)
        scale,orient=1.+.079*k,1.-1.45*k
    elif surface=='woven':
        phase=2*np.pi*h/24.
        scale,orient=1.+.047*np.sin(phase),.38+.62*np.cos(phase)
    else:raise ValueError(surface)
    near=np.select([regime=='strong',regime=='weak'],[1.,.3],default=0.)
    if not np.isin(regime,['strong','weak','null','global_null']).all():raise ValueError('regime')
    env=near[:,None]+(1.-near[:,None])*np.cos(np.pi/2*np.minimum(abs(h)/23.,1.))**2
    env[regime=='global_null']=0.
    v=np.abs(vt)
    asym=.145*vt**3/(v**3+.61**3)
    if law in ('strengthening','weakening'):
        factor=1.+(.22 if law=='strengthening' else -.22)*np.tanh((v-.40)/.19)
    elif law=='constant':factor=1.
    elif law=='rational':factor=1.045-.046*v*v/(v*v+.58**2)
    elif law=='gaussian_rate':
        from scipy.special import erf
        band=np.exp(-.5*((v-.64)/.25)**2)
        asym=.16*erf(vt/.34)*(1.-1.55*band)
        factor=1.007+.043*band
    else:raise ValueError(law)
    return theta[:,4,None]*scale*(1.+sign[:,None]*env*orient*asym)*factor

def simulate(theta,height_mm,speed,family,variant,surface,law,sign,regime,radius_mm=.25,
             backend='compliant',fine=False,return_diagnostics=False):
    th=np.asarray(theta,dtype=float)
    assert th.ndim==2 and th.shape[1]==6
    h=float(height_mm);r=float(radius_mm);n=len(th)
    dx,yy,w,residual,add,gc=grid(backend,h,r,family,variant,bool(fine))
    g0=th[:,2]+th[:,3]*h/25.+gc
    g=th[:,2,None]+th[:,3,None]*yy/25.+add
    speed=as_row(speed,n)
    norm=np.hypot(1.,g)
    vt=(speed[:,None]+g)/norm
    mu=resist(th,vt,yy,surface,law,as_row(sign,n),as_row(regime,n,str))
    norm0=np.hypot(1.,g0)
    if backend=='prescribed':
        jn=(1.+th[:,5])*np.maximum(1.-speed*g0,0.)/norm0
        normal_t=(g-g0[:,None])/(norm*norm0[:,None])
        tangent_t=(1.+g*g0[:,None])/(norm*norm0[:,None])
        drive=(speed+g0)/norm0+jn*(normal_t*w).sum(1)
        cap=jn*(np.abs(tangent_t)*mu*w).sum(1)
        # Same terminal scalar as v45/v67. Normal component comes from the
        # sum of the local normal and signed tangential impulses, to define a sensor.
        fraction=np.minimum(np.abs(drive)/np.maximum(cap,1e-30),1.)*np.sign(drive)
        normal_dot_n=(1.+g*g0[:,None])/(norm*norm0[:,None])
        tan_dot_n=(g0[:,None]-g)/(norm*norm0[:,None])
        vn=(speed*g0-1.)/norm0+jn*((normal_dot_n-mu*fraction[:,None]*np.sign(tangent_t)*tan_dot_n)*w).sum(1)
        vs=np.sign(drive)*np.maximum(abs(drive)-cap,0.)
        vel=np.stack([-(vn-g0*vs)/norm0,(vs+g0*vn)/norm0],axis=1)
        slip=abs(vs);extra={'normal_impulse':jn}
    else:
        # A common affine shoe is initially tangent at the nominal center. Place
        # that shoe above the support with the first support exactly touching.
        protrusion=residual[None,:]-th[:,3,None]*dx**2/50.
        gaps=(protrusion.max(1,keepdims=True)-protrusion)*.001
        out=np.empty((n,6),float)
        pack=lambda x: np.ascontiguousarray(x,dtype=np.float64)
        kernel()(n,len(dx),48 if fine else 24,*map(pack,(gaps,g,mu,th[:,5],speed,g0)),out)
        if not np.all(out[:,5]==1):raise RuntimeError(f'unreleased collision: {int((out[:,5]==0).sum())}/{n}')
        vel,slip=out[:,:2],out[:,2]
        extra={'normal_impulse':out[:,3], 'max_penetration_m':out[:,4]}
    assert np.isfinite(vel).all() and np.isfinite(slip).all()
    return (vel,slip,extra) if return_diagnostics else (vel,slip)
