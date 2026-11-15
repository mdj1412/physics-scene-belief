import os,time
import numpy as np
import torch
from common import *
from acquisition import pair_select,nested_scores,nested_plan
from run_compact import evaluate
def main():
    assert os.environ.get('SLURM_JOB_ID') and (R/'compact_done.json').exists();torch.set_num_threads(2)
    from bank import Bank
    write(R/'bank_import.json',dict(time=time.time(),compact=read(R/'compact_done.json')));snapshot('source_banks.json')
    x=public('test');setup={}
    for nm,N in [('qmc16',16384),('qmc64',65536)]:
        m=Bank(True,N);outer=Bank(True,16384,20261113) if N==16384 else None;plan=nested_plan()
        modes=['native','pair','unfiltered','agnostic'] if N==16384 else ['native','pair']
        g={k:[] for k in modes};diags={k:[] for k in modes if k!='native'}
        for i in range(0,len(x),16):
            xx=x[i:i+16];st=m.fit(xx);g['native'].append(m.gains(st).cpu().numpy())
            for k in diags:
                v,d=pair_select(m,st,xx,k);g[k].append(v.cpu().numpy());diags[k].append({a:b.cpu().numpy() for a,b in d.items()})
        if outer:
            nested=[]
            for i in range(0,len(x),4):
                xx=x[i:i+4];nested.append(nested_scores(m,m.fit(xx),outer,outer.fit(xx),plan=plan).cpu().numpy())
                if i%256==0:print('nested',i,flush=True)
            g['nested128']=nested
        for k,parts in g.items():
            v=np.concatenate(parts);out=dict(actions=v.argmax(1),gains=v)
            if k in diags:out.update({a:np.concatenate([d[a] for d in diags[k]]) for a in diags[k][0]})
            np.savez_compressed(R/'policies'/f'{nm}_{k}.npz',**out);freeze(nm+'_'+k)
        pred=evaluate(m,x,obs('test'));np.savez_compressed(R/'evaluations'/f'{nm}.npz',predictive=pred)
        setup[nm]=dict(cache_bytes=m.cache_bytes(),state_bytes=m.state_bytes());write(R/'bank_memory.json',setup)
        if outer:
            # Fixed audit subset: first16 independent worlds, six conditions.
            audit={};xx=x[:96]
            for label,M,seed in [('nested512',512,7919),('nested512_seed2',512,7920)]:
                pp=nested_plan(M,seed);vals=[]
                for i in range(0,len(xx),2):vals.append(nested_scores(m,m.fit(xx[i:i+2]),outer,outer.fit(xx[i:i+2]),plan=pp).cpu().numpy())
                audit[label]=np.concatenate(vals)
            pairs=[]
            for i in range(0,len(xx),8):pairs.append(pair_select(m,m.fit(xx[i:i+8]),xx[i:i+8],K=512)[0].cpu().numpy())
            audit['pair512']=np.concatenate(pairs);np.savez_compressed(R/'ranking_audit.npz',**audit)
        del m,outer;torch.cuda.empty_cache();print('done',nm,flush=True)
    write(R/'banks_done.json',dict(time=time.time()))
if __name__=='__main__':main()
