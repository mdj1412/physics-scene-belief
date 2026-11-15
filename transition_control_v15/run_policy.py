"""Only public.npz is read here; outcomes never enter selectors."""
import time
import numpy as np
import torch
from model import *
from acquisition import *
from common import *

CELLS=[('matched_nominal',0,'nominal'),('shifted_nominal',1,'nominal'),('shifted_correct',1,'shifted')]

def run():
    require();source=sources();start=time.time()
    if (R/'source_policy.json').exists():
        old=read(R/'source_policy.json');assert all(sha(p)==h for p,h in old.items()),'Resume source mismatch'
        source=old
    else:write(R/'source_policy.json',source)
    pub=np.load(R/'data/test/public.npz');n=len(pub['factual']);banks={law:(Bank(law=law),Bank(seed=2026111202,law=law)) for law in ['nominal','shifted']}
    large=Bank(65536,2026111203);largeouter=Bank(65536,2026111204)
    plans={512:plan(512),2048:plan(2048)};p2048b=plan(2048,81532)
    for cell,condition,law in CELLS:
        b,o=banks[law];records={};audits={};checkpoint=R/'checkpoints'/f'{cell}.npz'
        if checkpoint.exists():
            saved=np.load(checkpoint)
            records={k:list(v) for k,v in saved.items() if not k.startswith('audit__')}
            audits={k[7:]:list(v) for k,v in saved.items() if k.startswith('audit__')}
        def record(key,value):records.setdefault(key,[]).append(value.detach().cpu().numpy() if torch.is_tensor(value) else value)
        def audit(key,value):audits.setdefault(key,[]).append(value.detach().cpu().numpy())
        first=len(records.get('native',[]))
        print(cell,'resume at',first,flush=True)
        for i in range(first,n):
            x=t(pub['factual'][i,condition]);ctx=t(pub['context'][i,condition]);w=b.fit(x);ow=o.fit(x);z=targets(b.theta,ctx,law);oz=targets(o.theta,ctx,law)
            native=lmmse(b,w,z);record('native',choose(native));record('native_scores',native);record('ess',1/(w*w).sum())
            for mode in ['pair','unfiltered','agnostic']:
                score,diag=pair(b,w,z,x,mode=mode);record(mode,choose(score));record(mode+'_scores',score)
                if mode=='pair':record('eligible',diag['eligible']);record('fallback',diag['fallback'])
            nr=nested(b,w,z,o,ow,oz,preplan=plans[512],return128=True)
            for name,key in [('nested512','risk'),('nested128','risk128'),('variance512','variance')]:record(name,choose_risk(nr[key]));record(name+'_risk',nr[key])
            if law=='nominal':
                lw=large.fit(x);lz=targets(large.theta,ctx);record('qmc64_native',choose(lmmse(large,lw,lz)));record('qmc64_pair',choose(pair(large,lw,lz,x)[0]));record('ess64',1/(lw*lw).sum())
            if i<32:
                audit('pair512',choose(pair(b,w,z,x,K=512)[0]));na=nested(b,w,z,o,ow,oz,M=2048,preplan=plans[2048]);nb=nested(b,w,z,o,ow,oz,M=2048,preplan=p2048b)
                for name,v in [('nested2048',na),('nested2048_b',nb)]:audit(name,choose_risk(v['risk']));audit(name+'_risk',v['risk'])
                if law=='nominal':
                    low=largeouter.fit(x);loz=targets(largeouter.theta,ctx);la=nested(large,lw,lz,largeouter,low,loz,preplan=plans[512]);audit('qmc64_nested512',choose_risk(la['risk']));audit('qmc64_nested512_risk',la['risk'])
            if (i+1)%32==0:
                temporary=checkpoint.with_name(checkpoint.stem+'.tmp.npz')
                save(temporary,**records,**{'audit__'+k:v for k,v in audits.items()});temporary.replace(checkpoint)
                print(cell,i+1,'seconds',round(time.time()-start,1),flush=True)
        save(R/'policies'/f'{cell}.npz',**records);save(R/'audits'/f'{cell}.npz',**audits)
    assert all(sha(p)==h for p,h in source.items()),'Source changed during policy selection'
    freeze(R/'policy_freeze.json',list((R/'policies').glob('*.npz'))+list((R/'audits').glob('*.npz'))+[R/'data/test/public.npz'],private_outcomes_read=False)
    write(R/'policy_done.json',dict(time=time.time(),elapsed=time.time()-start));print('All policies frozen',flush=True)
if __name__=='__main__':run()
