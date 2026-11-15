import os,time,json
from pathlib import Path
import numpy as np
import torch
from physics import *
from common import R,write,public,private,freeze,check,tree_choice,sha

def evaluate_bank(bank,x,truth,batch=8,mixture=False):
    preds=[];mix=[];ess=[]
    for start in range(0,len(x),batch):
        end=min(start+batch,len(x));w=bank.fit(x[start:end]);pp=[];mm=[];ee=[]
        for a in range(4):
            updated=w if a==0 else bank.update(w.repeat_interleave(16,0),a,truth[f'obs{a}'][start:end].reshape(-1,len(noise(a))))
            pred=bank.predict(updated)
            pp.append((pred[:,None,:].expand(-1,16,-1) if a==0 else pred.reshape(end-start,16,2)).cpu().numpy())
            es=1/(updated*updated).sum(1);ee.append((es[:,None].expand(-1,16) if a==0 else es.reshape(end-start,16)).cpu().numpy())
            if mixture:
                p=bank.mixture(updated);mm.append((p[:,None,:].expand(-1,16,-1) if a==0 else p.reshape(end-start,16,2)).cpu().numpy())
        preds.append(np.stack(pp,1));ess.append(np.stack(ee,1))
        if mixture:mix.append(np.stack(mm,1))
        if start%128==0:print('eval',bank.kind,bank.size,start,flush=True)
    out=dict(predictive=np.concatenate(preds),ess=np.concatenate(ess))
    if mixture:out['mixture']=np.concatenate(mix)
    return out

def main():
    assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(2)
    stage=json.loads((R/'compact_stage_done.json').read_text());assert stage['no_bank_import']
    # Deliberately imported only after the standalone compact phase is frozen/completed.
    from bank import Bank
    write(R/'reference_import.json',dict(time=time.time(),compact_stage_done=stage))
    write(R/'source_baselines.json',{p:sha(p) for p in ['physics.py','bank.py','common.py','run_baselines.py','frozen_rule/rules.npz','frozen_rule/tree.json']})
    x=public('test');metadata={};reference=None
    for kind,size,nm in [('qmc',4096,'qmc4096'),('qmc',16384,'qmc16384'),('qmc',65536,'qmc65536'),('quad',61,'reference61')]:
        torch.cuda.synchronize();started=time.perf_counter();bank=Bank(kind,size);torch.cuda.synchronize()
        metadata[nm]=dict(setup_seconds=time.perf_counter()-started,cache_bytes=bank.cache_bytes(),state_bytes=bank.state_bytes(),nodes=bank.N)
        gain=[]
        for start in range(0,len(x),32):gain.append(bank.gains(bank.fit(x[start:start+32])).cpu().numpy())
        gain=np.concatenate(gain);np.savez_compressed(R/'policies'/f'{nm}.npz',actions=gain.argmax(1),gains=gain);freeze(nm)
        if nm in ['qmc16384','reference61']:
            np.savez_compressed(R/'policies'/f'{nm}_tree.npz',actions=tree_choice(x));freeze(nm+'_tree')
        check(nm);truth=private('test')
        out=evaluate_bank(bank,x,truth,batch=8 if kind=='quad' else 32,mixture=nm in ['qmc16384','reference61'])
        np.savez_compressed(R/'evaluations'/f'{nm}.npz',**out);write(R/'bank_setup.json',metadata)
        print('bank done',nm,metadata[nm],flush=True)
        del bank;torch.cuda.empty_cache()
    # Fixed, non-selected first16 base worlds, all3 conditions.
    bank=Bank('quad',81,moments=False);truth=private('test');sub={k:v[:48] for k,v in truth.items() if k!='lam'}
    out=evaluate_bank(bank,x[:48],sub,batch=8)
    np.savez_compressed(R/'reference81_audit.npz',**out)
    low=np.load(R/'evaluations/reference61.npz')['predictive'][:48]
    delta=abs(low-out['predictive']);lossdiff=.5*((out['predictive']-sub['target'][:,None,None,:])**2-(low-sub['target'][:,None,None,:])**2).sum(-1)
    write(R/'reference_audit.json',dict(base_worlds=16,cases=48,max_prediction_difference=float(delta.max()),mean_prediction_difference=float(delta.mean()),mean_loss_change=float(lossdiff.mean()),max_single_loss_change=float(abs(lossdiff).max())))
    write(R/'baselines_done.json',dict(time=time.time()));print('BASELINES DONE',flush=True)
if __name__=='__main__':main()
