import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from model import *
from mixture import *
from common import *
class Fake:pass
def exact_risk(mu,w,label):
    x,q=hermgauss(100);q=q/np.sqrt(np.pi);risk=0.
    for i in range(len(w)):
        y=mu[i]+np.sqrt(2)*x;ll=-.5*(y[:,None]-mu[None,:])**2+np.log(w)
        p=np.exp(ll-ll.max(1,keepdims=True));p/=p.sum(1,keepdims=True);pred=p@label
        risk+=w[i]*np.sum(q*(pred-label[i])**2)
    return float(risk)
def run():
    require();torch.set_num_threads(2);eps=normal_plan();out={}
    examples=[('v15_aliasing',np.array([0.,0.,1.,1.]),np.full(4,.25),np.array([-.5,-.5,.5,.5]),np.array([-4.,4.,-4.,4.])),
              ('higher_order',np.array([0.,0.,1.,1.,1.]),np.array([.25,.25,.0625,.375,.0625]),np.array([-5.,5.,-10.,0.,10.]),np.zeros(5))]
    for name,label,w,good,bad in examples:
        b=Fake();b.white={1:t(good[:,None]),2:t(bad[:,None]),3:t(np.zeros((len(w),1)))};z=t(label[:,None]*2**.5)
        comp=compress(b,t(w),z,True);r=risks(comp,eps);base=float(r[0]);assert int(r[1:].argmin())==0
        assert abs(float(r[2])-base)<1e-10,(name,r)
        cov=float(np.sum(w*(label-np.sum(w*label))*(good-np.sum(w*good))))
        oldpair=float(np.sum(w[:,None]*w[None,:]*(label[:,None]-label[None,:])**2*(-np.expm1(-(bad[:,None]-bad[None,:])**2/8))))
        out[name]=dict(region_risks=r.cpu().tolist(),exact_risk_good=exact_risk(good,w,label),exact_risk_bad=exact_risk(bad,w,label),old_pair_bad_score=oldpair,LMMSE_covariance_good=cov,components=comp['components'])
        if name=='higher_order':
            # Means0 and variances25 for both classes; Gaussian moment reduction loses label information.
            for c in [0,1]:
                wc=w[label==c]/w[label==c].sum();m=good[label==c];assert abs(np.sum(wc*m))<1e-12 and abs(np.sum(wc*m*m)-25)<1e-12
    b=Bank();counts=[int(((b.theta[:,0]==g)&(b.theta[:,1]==c)).sum()) for g,c in [(0,0),(0,1),(1,0),(1,1),(0,2),(1,2)]];assert counts==[2048]*4+[4096]*2
    from baselines import prepare,native,nested
    from acquisition import lmmse,nested as old_nested,plan
    pub=np.load(R/'data/pilot/public.npz');x=t(pub['factual'][0,0]);ctx=t(pub['context'][0,0]);w=b.fit(x);z=targets(b.theta,ctx);prepare(b)
    assert torch.allclose(native(b,w,z),lmmse(b,w,z),atol=1e-10,rtol=1e-8)
    o=Bank(seed=2026111202);ow=o.fit(x);oz=targets(o.theta,ctx);p=plan(128)
    nr,_=nested(b,w,z,o,ow,oz,p);old=old_nested(b,w,z,o,ow,oz,M=128,preplan=p)['risk'];assert torch.allclose(nr,old,atol=1e-10,rtol=1e-8)
    for k in [4,7]:
        c=compress(b,w,z[:,k:k+1],True);assert torch.allclose(c['w']@c['z'],w@z[:,k:k+1],atol=1e-9,rtol=0)
    out['prior_counts']=counts;out['baseline_equivalence_and_event_mass_verified']=True;write(R/'validation.json',out);print(out,flush=True)
if __name__=='__main__':run()
