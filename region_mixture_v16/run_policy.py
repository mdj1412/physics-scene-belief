import time
import numpy as np
import torch
from model import *
from acquisition import pair,plan
from mixture import compress,risks,normal_plan,payload
from baselines import prepare,native,nested
from common import *
CELLS=[('matched_nominal',0,'nominal'),('shifted_nominal',1,'nominal'),('shifted_correct',1,'shifted')]
def run():
    require();torch.set_num_threads(2);source=sources();write(R/'source_policy.json',source);start=time.time()
    pub=np.load(R/'data/test/public.npz');n=len(pub['factual']);starts=np.cumsum([0]+SIZES);eps=normal_plan();eps2=normal_plan(seed=91632);p512=plan(512);p2048=plan(2048,81532)
    large=prepare(Bank(65536,2026111203))
    for cell,condition,law in CELLS:
        b=prepare(Bank(law=law));o=Bank(seed=2026111202,law=law);records={};audits={}
        def rec(key,val):records.setdefault(key,[]).append(val.detach().cpu().numpy() if torch.is_tensor(val) else val)
        def aud(key,val):audits.setdefault(key,[]).append(val.detach().cpu().numpy())
        for i in range(n):
            x=t(pub['factual'][i,condition]);ctx=t(pub['context'][i,condition]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law);oz=targets(o.theta,ctx,law)
            ns=native(b,w,z);rec('native',ns[1:].argmax(0)+1);rec('native_scores',ns)
            ps,_=pair(b,w,z,x);rec('pair',ps[1:].argmax(0)+1)
            nr,nr128=nested(b,w,z,o,ow,oz,p512)
            for name,v in [('nested512',nr),('nested128',nr128)]:rec(name,v[1:].argmin(0)+1);rec(name+'_risk',v)
            if law=='nominal':
                lw=large.fit(x);lz=targets(large.theta,ctx);rec('qmc64_native',native(large,lw,lz)[1:].argmax(0)+1)
            ms={m:[] for m in ['region32','global32','region64','region_cf32']};audit_scores={'region32_Q8':[],'region32_seed2':[]};meta=[]
            for k in range(5):
                sl=slice(starts[k],starts[k+1]);zz=z[:,sl];ooz=oz[:,sl];event=k in [2,4]
                c32=compress(b,w,zz,event);c64=compress(b,w,zz,event,64);cg=compress(b,w,zz,event,global_mode=True);co=compress(o,ow,ooz,event)
                for name,c,outer in [('region32',c32,None),('global32',cg,None),('region64',c64,None),('region_cf32',c32,co)]:ms[name].append(risks(c,eps,outer=outer))
                meta.append([c32['components'],c32['omitted_mass'],payload(c32)])
                if i<32:
                    audit_scores['region32_Q8'].append(risks(c32,eps,Q=8));audit_scores['region32_seed2'].append(risks(c32,eps2))
            for name,v in ms.items():
                v=torch.stack(v,1);rec(name,v[1:].argmin(0)+1);rec(name+'_risk',v)
            rec('mixture_meta',meta);rec('ess',1/(w*w).sum())
            if i<32:
                for name,v in audit_scores.items():v=torch.stack(v,1);aud(name,v[1:].argmin(0)+1);aud(name+'_risk',v)
                r,_=nested(b,w,z,o,ow,oz,p2048);aud('nested2048',r[1:].argmin(0)+1);aud('nested2048_risk',r)
            if (i+1)%32==0:
                save(R/'checkpoints'/f'{cell}.npz',**records);print(cell,i+1,round(time.time()-start,1),flush=True)
        save(R/'policies'/f'{cell}.npz',**records);save(R/'audits'/f'{cell}.npz',**audits)
    assert all(sha(p)==h for p,h in source.items())
    freeze(R/'policy_freeze.json',list((R/'policies').glob('*.npz'))+list((R/'audits').glob('*.npz'))+[R/'data/test/public.npz'],private_outcomes_read=False)
    write(R/'policy_done.json',dict(time=time.time(),elapsed=time.time()-start));print('Policies frozen',flush=True)
if __name__=='__main__':run()
