from common import *
from model import *
from budget import plans,rep_plan
from run_policy import CELLS,EIDX,MIX,ADAPT
from runtime_fast import engine_for,execute_fast
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');pp=plans();rp=rep_plan();count=0;err=0.
    for cell,cond,law in CELLS:
        p=np.load(R/'policies'/f'{cell}.npz');b=Bank(law=law);o=Bank(seed=2026111202,law=law)
        for i in range(512):
            x=t(pub['factual'][i,cond]);ctx=t(pub['context'][i,cond]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law)[:,EIDX];oz=targets(o.theta,ctx,law)[:,EIDX]
            for k in range(2):
                for name in MIX:
                    e=engine_for(name,b,w,z[:,k:k+1],o,ow,oz[:,k:k+1],pp,rp);a,r,meta,h=execute_fast(name,e)
                    assert a==p[name][i,k],(cell,i,k,name,a,p[name][i,k])
                    if name in ADAPT:
                        assert np.array_equal(meta.cpu().numpy(),p[name+'_meta'][i,k])
                        assert np.allclose(h.cpu().numpy(),p[name+'_history'][i,k],atol=1e-10,rtol=0,equal_nan=True)
                    err=max(err,float(abs(r.cpu().numpy()-p[name+'_risk'][i,k]).max()));count+=1
        print('Equivalent fast kernels',cell,flush=True)
    assert err<1e-10;write(R/'fast_equivalence.json',dict(cases=count,max_risk_error=err,source=sha('budget_fast.py'),time=time.time()))
if __name__=='__main__':run()
