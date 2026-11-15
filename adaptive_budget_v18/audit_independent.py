from common import *
from model import *
from budget import plans,rep_plan,full_prefix,MS
from run_policy import MIX,NEST,EIDX
from runtime_fast import engine_for,execute_fast
from budget_fast import Engine
def run():
    require();torch.set_num_threads(2);pub=np.load(R/'data/test/public.npz');primary=np.load(R/'policies/matched_nominal.npz');b=Bank();o=Bank(seed=2026111202);pp=plans(101531);rp=rep_plan(101531);primaryrep=rep_plan();out={}
    def add(key,v):out.setdefault(key,[]).append(v.detach().cpu().numpy() if torch.is_tensor(v) else v)
    for i in range(512):
        x=t(pub['factual'][i,0]);ctx=t(pub['context'][i,0]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx)[:,EIDX];oz=targets(o.theta,ctx)[:,EIDX];rows={}
        for k in range(2):
            for name in MIX:
                e=engine_for(name,b,w,z[:,k:k+1],o,ow,oz[:,k:k+1],pp,rp);a,r,meta,h=execute_fast(name,e);rows.setdefault(name+'_seed2',[]).append(a)
            if i<64:
                K,m=primary['adaptive_meta'][i,k,:2].astype(int);e=Engine(b,w,z[:,k:k+1],o,ow,oz[:,k:k+1],pp,primaryrep,int(K),int(m));e.ensure(int(K),int(m));rows.setdefault('isolated_outer_risk',[]).append(e.metrics(int(K),int(m))[0])
        for key,v in rows.items():add(key,torch.stack(v) if torch.is_tensor(v[0]) else v)
        nr=full_prefix(b,w,z,o,ow,oz,pp)
        for name,m in NEST.items():add(name+'_seed2',nr[MS.index(m)].argmin(0)+1)
        if (i+1)%64==0:print('Independent audit',i+1,flush=True)
    save(R/'independent_audit.npz',**out);freeze(R/'independent_audit_freeze.json',[R/'independent_audit.npz','audit_independent.py','budget_fast.py','runtime_fast.py'],private_outcomes_read=False,master_seed=101531)
if __name__=='__main__':run()
