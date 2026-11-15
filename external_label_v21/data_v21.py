"""Private physical generator; supplies no hidden quantities to the selector."""
from v21_policy import t,torch
from model import from_unit, profile, FY, VYS, context, noise, velocity, basis, measure
from common import R,save
import numpy as np

LAWS=("nominal","speed_threshold","directional_damping")


def impulse(theta,y,vin,law):
    nominal,tangent,vn=velocity(theta,y,vin,details=True)
    if law=="nominal":return nominal,tangent
    g,c,d,s,curvature,e,b0,b1=theta.unbind(-1)
    slope=s+curvature*basis(t(y),g,True)
    norm=(1+slope.square()).sqrt()
    vt=(vin[0]*slope+vin[1])/norm
    jn=(1+e)*vn
    if law=="speed_threshold":
        mu=.05+.4*torch.sigmoid(b0+1)/(1+vt.abs()/(.15+.5*torch.sigmoid(b1)))
        jt=vt.sign()*torch.minimum(vt.abs(),mu*jn.clamp_min(0))
    elif law=="directional_damping":
        fraction=torch.sigmoid(b0+1.2*torch.tanh(3*vt)+b1*vn/(1+vn.abs()))
        jt=vt*fraction
    else:raise ValueError(law)
    out=torch.stack([vin[0]-jn/norm-jt*slope/norm,vin[1]+jn*slope/norm-jt/norm],-1)
    return torch.where((c==2)[...,None],out,nominal),torch.where(c==2,vt-jt,tangent)


def generate(split,count,origin,laws):
    facts=[];contexts=[];truth=[];actual=[[],[],[]];classes=[]
    for i in range(count):
        rng=np.random.default_rng(origin+137*i)
        theta=from_unit(t(rng.uniform(size=(1,7))))[0]
        classes.append(int(theta[1]))
        eps=t(rng.normal(size=11))*noise(0)
        eta=[t(rng.normal(size=(16,len(noise(a)))))*noise(a) for a in (1,2,3)]
        xs=[];cs=[];ys=[[],[],[]];zs=[]
        for law in laws:
            v,_=impulse(theta,0,(1,.35),law)
            x=torch.cat((profile(theta,FY),v))+eps
            ctx=context(x)
            _,tan=impulse(theta,.025,(1,ctx[1]),law)
            probe,_=impulse(theta,0,(1,-.7),law)
            xs.append(x.cpu().numpy());cs.append(ctx.cpu().numpy());zs.append(float(tan.abs()<=1e-8))
            for j,mu in enumerate([profile(theta,VYS[0]),profile(theta,VYS[1]),probe]):
                ys[j].append((mu+eta[j]).cpu().numpy())
        facts.append(xs);contexts.append(cs);truth.append(zs)
        for j in range(3):actual[j].append(ys[j])
    save(R/split/"public.npz",x=facts,ctx=contexts)
    save(R/split/"private.npz",label=truth,cls=classes,**{f"obs{j+1}":data for j,data in enumerate(actual)})
