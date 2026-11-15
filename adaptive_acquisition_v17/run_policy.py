import time
import torch
import numpy as np
from model import *
from baselines import prepare,native,nested
from mixture import compress,risks,normal_plan
from adaptive import select,phases,outer_samples
from acquisition import plan
from common import *
CELLS=[('matched_nominal',0,'nominal'),('shifted_nominal',1,'nominal'),('shifted_correct',1,'shifted')]
ETASKS=['contact_event','slip_event'];EIDX=[4,7]
ADAPT=['adaptive','adaptive_global','fixed32','fixed128','global128']
def config(name):return dict(global_mode=name in ['adaptive_global','global128'],fixed=32 if name=='fixed32' else (128 if name in ['fixed128','global128'] else None))
def run():
    require();torch.set_num_threads(2);source=sources();write(R/'source_policy.json',source);start=time.time()
    pub=np.load(R/'data/test/public.npz');n=len(pub['factual']);pre=plan(512);alt=plan(512,81532);ref=plan(2048,81533);ph=phases(81531);ph2=phases(81532);eps=normal_plan()
    for cell,cond,law in CELLS:
        b=prepare(Bank(law=law));o=Bank(seed=2026111202,law=law);large=prepare(Bank(65536,2026111203,law));records={};audits={}
        if cond==0:ab=prepare(Bank(seed=2026111301));ao=Bank(seed=2026111302)
        def rec(key,val):records.setdefault(key,[]).append(val.detach().cpu().numpy() if torch.is_tensor(val) else val)
        def aud(key,val):audits.setdefault(key,[]).append(val.detach().cpu().numpy() if torch.is_tensor(val) else val)
        for i in range(n):
            x=t(pub['factual'][i,cond]);ctx=t(pub['context'][i,cond]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law)[:,EIDX];oz=targets(o.theta,ctx,law)[:,EIDX]
            ns=native(b,w,z,[1,1]);rec('native',ns[1:].argmax(0)+1)
            lw=large.fit(x);lz=targets(large.theta,ctx,law)[:,EIDX];rec('qmc64_native',native(large,lw,lz,[1,1])[1:].argmax(0)+1)
            nr,n128=nested(b,w,z,o,ow,oz,pre,[1,1]);rec('nested512',nr[1:].argmin(0)+1);rec('nested128',n128[1:].argmin(0)+1);rec('nested512_risk',nr[1:])
            rows={};arows={}
            for k in range(2):
                zz=z[:,k:k+1];ooz=oz[:,k:k+1];sample=outer_samples(o,ow,ooz,pre)
                for name in ADAPT:
                    a,r,meta,h=select(b,w,zz,sample,ph,**config(name))
                    for key,val in [(name,a),(name+'_risk',r),(name+'_meta',meta),(name+'_history',h)]:rows.setdefault(key,[]).append(val)
                for name in ['region32','global32']:
                    r=risks(compress(b,w,zz,True,global_mode=name=='global32'),eps)
                    rows.setdefault(name,[]).append(int(r[1:].argmin())+1)
                if cond==0:
                    sample2=outer_samples(o,ow,ooz,alt)
                    for name in ADAPT:
                        a,r,meta,h=select(b,w,zz,sample2,ph2,**config(name));arows.setdefault(name+'_seed2',[]).append(a)
            for key,vals in rows.items():rec(key,torch.stack(vals) if torch.is_tensor(vals[0]) else vals)
            for key,vals in arows.items():aud(key,vals)
            rec('ess',1/(w*w).sum())
            if cond==0:
                rr,_=nested(b,w,z,o,ow,oz,alt,[1,1]);aud('nested512_seed2',rr[1:].argmin(0)+1)
            if i<64:
                rr,_=nested(b,w,z,o,ow,oz,ref,[1,1]);aud('nested2048',rr[1:].argmin(0)+1);aud('nested2048_risk',rr[1:])
                if cond==0:
                    aw=ab.fit(x);aow=ao.fit(x);az=targets(ab.theta,ctx)[:,EIDX];aoz=targets(ao.theta,ctx)[:,EIDX];br={}
                    for k in range(2):
                        samp=outer_samples(ao,aow,aoz[:,k:k+1],pre)
                        for name in ['adaptive','adaptive_global','fixed128','global128']:
                            a,_,_,_=select(ab,aw,az[:,k:k+1],samp,ph,**config(name));br.setdefault(name+'_bank2',[]).append(a)
                    for key,val in br.items():aud(key,val)
                    aud('native_bank2',native(ab,aw,az,[1,1])[1:].argmax(0)+1)
                    rr,_=nested(ab,aw,az,ao,aow,aoz,pre,[1,1]);aud('nested512_bank2',rr[1:].argmin(0)+1)
            if (i+1)%32==0:save(R/'checkpoints'/f'{cell}.npz',**records);print(cell,i+1,round(time.time()-start,1),flush=True)
        save(R/'policies'/f'{cell}.npz',**records);save(R/'audits'/f'{cell}.npz',**audits)
    assert all(sha(p)==h for p,h in source.items())
    freeze(R/'policy_freeze.json',list((R/'policies').glob('*.npz'))+list((R/'audits').glob('*.npz'))+[R/'data/test/public.npz'],private_outcomes_read=False)
    write(R/'policy_done.json',dict(time=time.time(),elapsed=time.time()-start));print('Policies frozen',flush=True)
if __name__=='__main__':run()
