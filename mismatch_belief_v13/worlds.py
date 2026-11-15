"""Private data generation only; selectors never import this module."""
import numpy as np
import torch
from physics import *
CONDITIONS=['matched','viscous','saturating_half','saturating_full','weakening_half','weakening_full']
def truth_velocity(th,y,vin,condition,pars):
    g,c,s,a,e=th.unbind(-1);slope=s+a*basis(t(y),g,True);norm=torch.sqrt(1+slope*slope)
    vn=(vin[0]-slope*vin[1])/norm;vt=(vin[0]*slope+vin[1])/norm;jn=(1+e)*vn
    old=torch.sign(vt)*torch.minimum(vt.abs(),.22*c*jn)
    lam,alpha,scale,beta,v0=pars
    if condition=='matched':jt=old
    else:
        if condition=='viscous':new=c*lam*vt
        elif condition.startswith('saturating'):new=c*torch.sign(vt)*alpha*scale*torch.tanh(vt.abs()/scale)
        else:new=c*vt*(.12+beta/(1+(vt.abs()/v0)**2))
        rho=.5 if condition.endswith('half') else 1.;jt=(1-rho)*old+rho*new
    return torch.stack([vin[0]-jn/norm-jt*slope/norm,vin[1]+jn*slope/norm-jt/norm],-1)
def dataset(N,seed,test=False):
    conditions=CONDITIONS if test else ['matched'];repeats=16 if test else 1
    xs=[];ts=[];ys=[];ps=[];observations={a:[] for a in [1,2,3]}
    for i in range(N):
        rng=np.random.default_rng(seed+137*i)
        theta=np.array([rng.integers(2),rng.integers(2),rng.uniform(-.4,.4),rng.uniform(-.1,.1),rng.uniform(.35,.9)])
        pars=np.array([rng.uniform(.25,.65),rng.uniform(.35,.85),rng.uniform(.25,.65),rng.uniform(.3,.6),rng.uniform(.2,.6)])
        eps0=rng.normal(size=11);eps={a:rng.normal(size=(repeats,len(noise(a)))) for a in [1,2,3]}
        pub=[];targets=[];ob={a:[] for a in observations};th=t(theta)
        for cond in conditions:
            factual=torch.cat([profile(th,FY),truth_velocity(th,0,(1,.35),cond,pars)])
            pub.append(factual.cpu().numpy()+eps0*noise(0).cpu().numpy());targets.append(truth_velocity(th,L,(1,.9),cond,pars).cpu().numpy())
            for a in [1,2,3]:
                mu=profile(th,VYS[a-1]) if a<3 else truth_velocity(th,0,(1,-.7),cond,pars)
                ob[a].append(mu.cpu().numpy()+eps[a]*noise(a).cpu().numpy())
        xs.append(pub);ts.append(theta);ps.append(pars);ys.append(targets)
        for a in observations:observations[a].append(ob[a])
    return dict(factual=np.array(xs)),dict(theta=np.array(ts),parameters=np.array(ps),target=np.array(ys),**{f'obs{a}':np.array(v) for a,v in observations.items()})
