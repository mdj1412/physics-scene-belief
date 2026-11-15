import os,time
import numpy as np
import torch
from physics import noise
from common import *
def main():
    assert os.environ.get('SLURM_JOB_ID') and (R/'compact_done.json').exists();torch.set_num_threads(2)
    from bank import Bank
    write(R/'bank_import.json',dict(time=time.time(),compact=read(R/'compact_done.json')));x=public('test');setup={}
    for name,expanded,N,seed in [('qmc16',False,16384,20261112),('expanded_qmc16',True,16384,20261112),('expanded_qmc64',True,65536,20261112),('expanded_qmc64_seed2',True,65536,20261113)]:
        torch.cuda.synchronize();t0=time.perf_counter();m=Bank(expanded,N,seed);torch.cuda.synchronize()
        setup[name]=dict(setup_s=time.perf_counter()-t0,state_bytes=m.state_bytes(),cache_bytes=m.cache_bytes())
        gains=[]
        for i in range(0,len(x),32):gains.append(m.gains(m.fit(x[i:i+32])).cpu().numpy())
        g=np.concatenate(gains);np.savez_compressed(R/'policies'/f'{name}.npz',actions=g.argmax(1),gains=g);freeze(name)
        ob=observations('test');preds=[];esses=[]
        for i in range(0,len(x),16):
            end=min(i+16,len(x));s=m.fit(x[i:end]);pp=[];ee=[]
            for a in range(4):
                st=s if a==0 else m.update(s.repeat_interleave(16,0),a,ob[a][i:end].reshape(-1,len(noise(a))))
                p=m.predict(st);ess=1/(st*st).sum(1)
                pp.append((p[:,None,:].expand(-1,16,-1) if a==0 else p.reshape(end-i,16,2)).cpu().numpy())
                ee.append((ess[:,None].expand(-1,16) if a==0 else ess.reshape(end-i,16)).cpu().numpy())
            preds.append(np.stack(pp,1));esses.append(np.stack(ee,1))
        np.savez_compressed(R/'evaluations'/f'{name}.npz',predictive=np.concatenate(preds),ess=np.concatenate(esses));print('done',name,flush=True)
        del m;torch.cuda.empty_cache()
    write(R/'bank_setup.json',setup);write(R/'banks_done.json',dict(time=time.time()))
if __name__=='__main__':main()
