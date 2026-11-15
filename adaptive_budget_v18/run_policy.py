import time
import numpy as np
import torch
from model import *
from baselines import prepare,native
from budget import *
from common import *
CELLS=[('matched_nominal',0,'nominal'),('shifted_nominal',1,'nominal'),('shifted_correct',1,'shifted')]
ETASKS=['contact_event','slip_event'];EIDX=[4,7]
ADAPT=['adaptive','balanced','component_only','outer_only']
FIXED={'fixed32_512':(32,128,False),'fixed128_512':(128,128,False),'fixed32_4096':(32,1024,False),'fixed128_4096':(128,1024,False),'global32_512':(32,128,True),'global128_4096':(128,1024,True)}
MIX=ADAPT+list(FIXED);NEST={'nested512':128,'nested2048':512,'nested4096':1024}
def make_engine(name,b,w,z,o,ow,oz,pp,rp):
    K,m,g=FIXED[name] if name in FIXED else (128,128 if name=='component_only' else 1024,False)
    return Engine(b,w,z,o,ow,oz,pp,rp,K,m,g)
def execute(name,eng):return select(eng,mode=name if name in ADAPT else 'adaptive',fixed=FIXED[name][:2] if name in FIXED else None)
def run():
    require();torch.set_num_threads(2);source=sources();write(R/'source_policy.json',source);start=time.time();pub=np.load(R/'data/test/public.npz')
    pp=plans();rp=rep_plan();pa=plans(92531);ra=rep_plan(92531);pref=plans(93531,2048)
    for cell,cond,law in CELLS:
        b=prepare(Bank(law=law));o=Bank(seed=2026111202,law=law);big=prepare(Bank(65536,2026111203,law));rec={};audit={}
        def add(dest,key,val):dest.setdefault(key,[]).append(val.detach().cpu().numpy() if torch.is_tensor(val) else val)
        for i in range(512):
            x=t(pub['factual'][i,cond]);ctx=t(pub['context'][i,cond]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law)[:,EIDX];oz=targets(o.theta,ctx,law)[:,EIDX]
            add(rec,'native',native(b,w,z,[1,1])[1:].argmax(0)+1);bw=big.fit(x);bz=targets(big.theta,ctx,law)[:,EIDX];add(rec,'qmc64_native',native(big,bw,bz,[1,1])[1:].argmax(0)+1)
            nr=full_prefix(b,w,z,o,ow,oz,pp);add(rec,'full_prefix_risk',nr)
            for m,mm in NEST.items():add(rec,m,nr[MS.index(mm)].argmin(0)+1)
            rows={};arows={}
            for k in range(2):
                zz=z[:,k:k+1];ooz=oz[:,k:k+1]
                for name in MIX:
                    eng=make_engine(name,b,w,zz,o,ow,ooz,pp,rp);a,r,meta,h=execute(name,eng)
                    for key,v in [(name,a),(name+'_risk',r),(name+'_meta',meta),(name+'_history',h),(name+'_work',eng.work)]:rows.setdefault(key,[]).append(v)
                    if name=='adaptive' and cond==0 and i<64:
                        K,m=int(meta[0]),int(meta[1]);e2=Engine(b,w,zz,o,ow,ooz,pa,rp,K,m);e2.ensure(K,m);rr=e2.metrics(K,m)[0];arows.setdefault('isolated_outer_risk',[]).append(rr)
                if cond==0:
                    for name in MIX:
                        eng=make_engine(name,b,w,zz,o,ow,ooz,pa,ra);a,r,meta,h=execute(name,eng);arows.setdefault(name+'_seed2',[]).append(a)
            for key,v in rows.items():add(rec,key,torch.stack(v) if torch.is_tensor(v[0]) else v)
            for key,v in arows.items():add(audit,key,torch.stack(v) if torch.is_tensor(v[0]) else v)
            add(rec,'ess',1/(w*w).sum())
            if cond==0:
                nr2=full_prefix(b,w,z,o,ow,oz,pa)
                for name,m in NEST.items():add(audit,name+'_seed2',nr2[MS.index(m)].argmin(0)+1)
            if i<64:
                rr=full_prefix(b,w,z,o,ow,oz,pref,[2048])[0];add(audit,'reference_risk',rr);add(audit,'reference',rr.argmin(0)+1)
            if (i+1)%32==0:save(R/'checkpoints'/f'{cell}.npz',**rec);print(cell,i+1,round(time.time()-start,1),flush=True)
        save(R/'policies'/f'{cell}.npz',**rec);save(R/'audits'/f'{cell}.npz',**audit)
    assert all(sha(p)==h for p,h in source.items());freeze(R/'policy_freeze.json',list((R/'policies').glob('*.npz'))+list((R/'audits').glob('*.npz'))+[R/'data/test/public.npz'],private_outcomes_read=False)
    write(R/'policy_done.json',dict(time=time.time(),elapsed=time.time()-start));print('Policies frozen',flush=True)
if __name__=='__main__':run()
