from model import *
from budget import *
from common import *
from mixture import posterior
from baselines import nested
def run():
    require();pub=np.load(R/'data/pilot/public.npz');b=Bank();o=Bank(seed=2026111202);x=t(pub['factual'][0,0]);ctx=t(pub['context'][0,0]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx)[:,[4,7]];oz=targets(o.theta,ctx)[:,[4,7]];pp=plans();rp=rep_plan();err=0.
    for k in range(2):
        for glob in [False,True]:
            e=Engine(b,w,z[:,k:k+1],o,ow,oz[:,k:k+1],pp,rp,global_mode=glob)
            for K,m in [(16,128),(32,128),(32,256),(64,512),(128,1024)]:
                e.ensure(K,m);r=e.metrics(K,m)[0];loss=[];n=K*e.R
                for c in [0,1]:
                    cw=e.base[c,:n];cw=cw/cw.sum();zz=e.labels[c,:n].to(DT)[:,None]*2**.5;comp=dict(w=cw,z=zz,mu={a:e.mu[c,a-1,:n] for a in [1,2,3]})
                    rs=[]
                    for a in [1,2,3]:
                        y=e.y[a-1,:,:m].reshape(-1,7);pr=posterior(comp,a,y)@zz;rs.append((.5*(pr[:,0]-e.truth[:,:m].flatten())**2).mean())
                    loss.append(torch.stack(rs))
                dense=torch.stack(loss).mean(0);err=max(err,float((dense-r).abs().max()));assert torch.allclose(dense,r,atol=1e-11,rtol=1e-10)
                if not glob:
                    prob=(w*(z[:,k]>0)).sum();assert abs(float((e.base[0,:n]*e.labels[0,:n]).sum()/e.base[0,:n].sum()-prob))<2e-10
    psmall=plans(mmax=128);rr=full_prefix(b,w,z,o,ow,oz,psmall,[128])[0]
    concat=(torch.cat([p[0] for p in psmall]),torch.cat([p[1] for p in psmall]));ref,_=nested(b,w,z,o,ow,oz,concat,[1,1]);assert torch.allclose(rr,ref[1:],atol=1e-11,rtol=1e-10)
    assert route('adaptive',16,128,.1,.01)==(32,128,0);assert route('adaptive',16,128,.01,.1)==(16,256,1);assert route('adaptive',128,128,.1,.01)==(128,256,1)
    write(R/'validation.json',dict(cached_dense_max_error=err,label_mass=True,full_inner_prefix=True,routing=True));print('Cached/dense validation passed',flush=True)
if __name__=='__main__':run()
