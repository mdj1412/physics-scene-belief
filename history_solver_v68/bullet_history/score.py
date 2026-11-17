"""Separate evaluator for sealed full-history Bullet probabilities and existing external labels."""
from pathlib import Path
import hashlib,json,os,sys
import numpy as np
from scipy.stats import t
B=Path(__file__).resolve().parent;H=B.parent
sys.path.insert(0,str(H))
from crossover68 import routed,FAMS,LAWS,ROUTES

def read(path):
    with np.load(path) as f:return f['p'],f['shift'],list(f['cells']),f['actions']
def ci(x,m=1):
    avg=x.mean(0);half=x.std(0,ddof=1)*t.isf(.025/m,len(x)-1)/np.sqrt(len(x))
    return np.stack((avg,avg-half,avg+half),-1).tolist()
def main():
    assert os.getenv('SLURM_JOB_ID') and os.environ.get('CUDA_VISIBLE_DEVICES')==''
    seal=json.loads((B/'forecasts_sealed.json').read_text())
    risk=np.zeros((12,3,2,2,4,5))
    ns=np.zeros((12,2,2));label_mean=ns.copy();route_changed=np.zeros(12)
    for k in range(12):
        sub=f'panel_{k:02d}';pfile=B/'forecasts'/(sub+'.npz')
        assert hashlib.sha256(pfile.read_bytes()).hexdigest()==seal['forecasts'][pfile.name]
        pb,sh,names,act=read(pfile);pc,sc,nmc,actc=read(H/'forecasts'/'compliant'/(sub+'.npz'))
        for i,name in enumerate(names):
            fam,s,law,v,h=name[:-4].split('_');f=FAMS.index(fam);l=LAWS.index(law)
            ix=nmc.index(name)
            route_changed[k]+=np.mean(act[i]!=actc[ix])/len(names)
            with np.load(ROUTES/sub/f'{fam}_{v}_{h}.npz') as z:g=z['route'][...,:8]
            with np.load(H/'bullet_labels'/sub/name) as z:valid=z['valid'];y=z['speed']<=.02
            assert valid.shape==(2,3,4,8) and valid.any()
            ns[k,f,l]+=valid.sum();label_mean[k,f,l]+=(valid&y).sum()
            for m,base,shift in [(0,pb[i],sh[i]),(1,pb[i],None),(2,pc[ix][...,:8,:],sc[ix])]:
                p=routed(base,shift,g)
                # [seed,stage,area,offset,env,obj,rep]
                loss=(p[:,:,None]-y[None,None,...,None])**2
                risk[k,m,f,l]+=(loss*valid[None,None,...,None]).mean(-1).sum((2,3,4,5))
    risk/=ns[:,None,:,:,None,None]
    avg=risk.mean(4);soft=avg[...,1]-avg[...,0]; rescue=avg[...,4]-avg[...,3]
    results=dict(order=['Bullet full history and 4cal','Bullet full history without calibration correction','compliant history and 4cal'],
        catalog_mean=avg.tolist(),abs_brier=avg.mean(0).tolist(),cells='history x family x law x stage',
        soft_full=[ci(soft[:,m],6) for m in range(3)],restored_dropped=[ci(rescue[:,m].mean(1),6) for m in range(3)],
        family_rescue=ci(rescue[:,0]),seed_soft=(risk[:,0,...,1]-risk[:,0,...,0]).mean(0).tolist(),
        seed_rescue=(risk[:,0,...,4]-risk[:,0,...,3]).mean(0).tolist(),label_prevalence=ci(label_mean/ns),
        valid=int(ns.sum()),total=12*48*2*3*4*8,
        matched_history_soft_difference=ci(soft[:,0]-soft[:,2]),matched_history_rescue_difference=ci((rescue[:,0]-rescue[:,2]).mean(1)),
        route_change_compliant_to_Bullet=ci(route_changed),job=os.environ['SLURM_JOB_ID'])
    (B/'external_history_result.json').write_text(json.dumps(results,indent=2))
if __name__=='__main__':main()
