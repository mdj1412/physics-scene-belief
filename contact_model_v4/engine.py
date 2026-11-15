"""Continuous collision detection: sphere vs static rounded rectangles in 2D.

SI units, solid-sphere inertia, Coulomb instantaneous impulse. No gravity.
This is a deliberately bounded rigid-contact diagnostic, not a full simulator.
"""
import torch
import numpy as np

DTYPE=torch.float64
R=.02
FAMILIES=['corner','gate','deflection']
BOUNDS=np.array([[-.008,.008],[.15,.85],[0.,.6],[.022,.034]])

def tensor(x,device='cuda'):
    return torch.as_tensor(x,dtype=DTYPE,device=device)

def decode(x):
    b=tensor(BOUNDS,x.device)
    return b[:,0]+x*(b[:,1]-b[:,0])

def boxes(theta,family):
    g,e,mu,d=theta.T; z=torch.zeros_like(g)
    if family=='corner':
        cx=torch.stack([z+.12],1);cy=torch.stack([.014+d+g],1)
        hx=torch.stack([z+.015],1);hy=torch.stack([d],1)
    elif family=='gate':
        cx=torch.stack([z+.12,z+.12],1);cy=torch.stack([g+d+.04,g-d-.04],1)
        hx=torch.stack([z+.015,z+.015],1);hy=torch.stack([z+.04,z+.04],1)
    elif family=='deflection':
        cx=torch.stack([z+.08,z+.22],1);cy=torch.stack([.014+d+g,-.058+g],1)
        hx=torch.stack([z+.015,z+.015],1);hy=torch.stack([d,z+.035],1)
    else: raise ValueError(family)
    return torch.stack([cx,cy],2),torch.stack([hx,hy],2)

def simulate(theta,rounding,family,shift=0.,horizon=.8,max_events=32):
    """Return position, velocity, contact sequence code, last event, overflow.

    theta [N,4]=(g,e,mu,d); rounding [N]. Initial source y is intervention.
    Rounded shape keeps outer extents fixed. Sphere radius dilates its core.
    """
    n=len(theta);dev=theta.device
    pos=torch.zeros((n,2),device=dev,dtype=DTYPE);pos[:,0]=-.18
    pos[:,1]=torch.as_tensor(shift,device=dev,dtype=DTYPE)
    vel=torch.zeros_like(pos);vel[:,0]=1.
    omega=torch.zeros(n,device=dev,dtype=DTYPE)
    time=torch.zeros_like(omega);last=time.clone();code=torch.zeros(n,device=dev,dtype=torch.long)
    center,half=boxes(theta,family);core=half-rounding[:,None,None];rad=R+rounding
    inf=torch.full_like(time,float('inf'));hit=torch.zeros(n,device=dev,dtype=torch.bool)
    for iteration in range(max_events+1):
        p=pos[:,None,:]-center;v=vel[:,None,:]
        candidates=[];normals=[];labels=[]
        for axis in [0,1]:
            other=1-axis
            for sign in [-1.,1.]:
                vd=v[:,:,axis].expand_as(core[:,:,axis]); safe=torch.where(vd.abs()>1e-14,vd,torch.ones_like(vd))
                t=(sign*(core[:,:,axis]+rad[:,None])-p[:,:,axis])/safe
                crossing=p[:,:,other]+t*v[:,:,other]
                ok=(vd*sign < -1e-12)&(t>1e-9)&(crossing.abs()<=core[:,:,other]+1e-10)
                candidates.append(torch.where(ok,t,torch.full_like(t,float('inf'))))
                normal=torch.zeros_like(p);normal[:,:,axis]=sign;normals.append(normal)
        for sx,sy in [(-1.,-1.),(-1.,1.),(1.,-1.),(1.,1.)]:
            s=tensor([sx,sy],dev);offset=p-core*s
            aa=(v*v).sum(2);bb=(offset*v).sum(2);cc=(offset*offset).sum(2)-rad[:,None]**2
            disc=bb*bb-aa*cc
            t=(-bb-torch.sqrt(disc.clamp_min(0)))/aa.clamp_min(1e-20)
            at=offset+t[:,:,None]*v;normal=at/rad[:,None,None]
            ok=(disc>=0)&(t>1e-9)&(at[:,:,0]*sx>=-1e-10)&(at[:,:,1]*sy>=-1e-10)&((normal*v).sum(2)<-1e-12)
            candidates.append(torch.where(ok,t,torch.full_like(t,float('inf'))));normals.append(normal)
        ts=torch.stack(candidates,2).reshape(n,-1)
        ns=torch.stack(normals,2).reshape(n,-1,2)
        t,idx=ts.min(1);normal=ns[torch.arange(n,device=dev),idx]
        hit=(t < horizon-time)&(time<horizon-1e-12)
        if iteration==max_events: break
        step=torch.where(hit,t,(horizon-time).clamp_min(0))
        pos=pos+vel*step[:,None];time=time+step
        normal=torch.where(hit[:,None],normal,torch.zeros_like(normal))
        tangent=torch.stack([-normal[:,1],normal[:,0]],1)
        vn=(vel*normal).sum(1);vt=(vel*tangent).sum(1)-omega*R
        jn=-(1+theta[:,1])*vn
        jt=torch.clamp(-vt/3.5,min=-theta[:,2]*jn,max=theta[:,2]*jn)
        vel=vel+jn[:,None]*normal+jt[:,None]*tangent
        omega=omega-jt/(.4*R)
        last=torch.where(hit,time,last)
        # base 3 records obstacle identity (1,2); non-hit leaves prefix unchanged.
        code=torch.where(hit,code*3+(idx//8+1),code)
        if not bool(hit.any()): break
    return dict(position=pos,velocity=vel,code=code,last_event=last,overflow=hit)

def batch(theta,rounding,family,shifts):
    shifts=tensor(shifts,theta.device);n=len(theta);q=len(shifts)
    out=simulate(theta.repeat(q,1),rounding.repeat(q),family,shifts.repeat_interleave(n))
    return {k:v.reshape((q,n)+v.shape[1:]) for k,v in out.items()}

def observation_means(theta,rounding,family,action):
    if action=='view':
        # world coordinates extracted from top-view obstacle color masks
        return theta[:,[0,3]]
    shift={'factual':0.,'left':-.022,'right':.022,'center':.009}[action]
    return simulate(theta,rounding,family,shift)['velocity']

NOISE={'factual':np.array([.025,1e6]),'view':np.array([.0015,.0015]),
       'left':np.array([.025,.025]),'right':np.array([.025,.025]),'center':np.array([.025,.025])}
ACTIONS=['view','left','right','center']
SHIFTS=np.array([-.030,-.017,-.007,.004,.015,.028])
