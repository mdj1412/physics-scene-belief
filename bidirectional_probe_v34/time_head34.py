"""Measure the same complete path for the externally fit direction-aware gain head and both baselines."""
from pathlib import Path
import json,os,time
import numpy as np

from run34 import (ROOT,BASE,OUT,LOCK,STAGES,METHODS as PRIMARY,SCHEDULES,LAWS,ActionConditionedDetector,
    History,allocate,setup,paired_timing,Engine,deployment_truth,atomic_json,atomic_npz,jsonable,digest,torch)
from direct_head34 import load_artifact,score_direct34

METHODS=('signed','fixed4','uncertainty')
TIMING=ROOT/'results_head_time'
def callback(schedule,family,artifact):
    def head(h,k):
        return score_direct34(h.feature(k),h.x[h.ids[:k]],h.y[:k],h.d[:k],schedule,k,family,artifact)
    return head

def main():
    cfg=json.loads((ROOT/'policy_design34.json').read_text())
    assert torch.cuda.device_count()==1 and torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
    assert (OUT/'completion.json').is_file()
    assert cfg['timed_blocks_per_cell']==2
    torch.set_num_threads(2)
    artifact=load_artifact()
    model=json.loads(LOCK.read_text())['models']['broad']
    files=[ROOT/'direct_head34.py',ROOT/'direct_models34.json',ROOT/'diag_models.json',ROOT/'diag34_model.py',Path(__file__),ROOT/'run34.py',ROOT/'target34.py',ROOT/'detector34.py',OUT/'completion.json']
    fingerprint='|'.join(digest(p) for p in files)
    with torch.inference_mode():
        det=ActionConditionedDetector()
        for b in range(2):
            for i,law in enumerate(LAWS):
                tic=time.perf_counter();dst=TIMING/law/f'block_{b:03d}'
                dst.mkdir(exist_ok=True,parents=True)
                if (dst/'meta.json').exists():
                    m=json.loads((dst/'meta.json').read_text())
                    assert m['fingerprint']==fingerprint and m['data_sha256']==digest(dst/'timing.npz')
                    continue
                base,alpha,pair,recs,xx=setup(cfg,i,b)
                decs=[Engine(pair,*al) for al in alpha]
                truth=[deployment_truth(base+200+e,al,law) for e,al in enumerate(alpha)]
                references=[];heads=[];risks=[]
                for s,(schedule,records) in enumerate(zip(SCHEDULES,recs)):
                    origin=OUT/law/f'block_{b:03d}'/schedule
                    with np.load(origin/'metrics.npz') as z,np.load(origin/'choices.npz') as ch:
                        hs=[History(det,r) for r in records];f=callback(schedule,i//2,artifact);heads.append(f)
                        counts=np.stack([allocate(hs,m,model,f) for m in METHODS],axis=1)
                        # Check raw, precommitted active-stage computation using archived observations.
                        for ee,h in enumerate(hs):
                            for st,k in enumerate((2,4)):
                                from_archive=score_direct34(ch['features'][ee,st],ch['paid_x'][ee,:k],
                                  ch['paid_y'][ee,:k],ch['paid_direction'][ee,:k],schedule,k,i//2,artifact)
                                assert abs(f(h,k)-from_archive)<1e-12
                        np.testing.assert_array_equal(counts[:,1:],z['counts'][:,1:])
                        acts=np.ones(ch['gaps'].shape,dtype=np.int8)
                        order=np.argsort(-ch['gaps'],axis=-1,kind='stable')[...,:13]
                        np.put_along_axis(acts,order,3,axis=-1)
                        chosen=acts[np.arange(8)[:,None],np.searchsorted(STAGES,counts)].transpose(0,2,1)
                        references.append(dict(counts=counts,actions=chosen,probabilities=z['probabilities']))
                        risks.append(z['oracle_stage_risk'][np.arange(8)[:,None],np.searchsorted(STAGES,counts)])
                durations=paired_timing(det,recs,METHODS,model,pair,decs,xx,truth,references,base+711+i%2,heads)
                fn=dst/'timing.npz'
                atomic_npz(fn,methods=METHODS,full_ms=durations[:,:,:,0],warm_ms=durations[:,:,:,1],
                           counts=[r['counts'] for r in references],actions=[r['actions'] for r in references],brier=risks)
                atomic_json(dst/'meta.json',dict(law=law,block=b,fingerprint=fingerprint,data_sha256=digest(fn)))
                print(law,b,round(time.perf_counter()-tic,2),flush=True)
        atomic_json(TIMING/'completion.json',dict(job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name(),
                      fingerprint=fingerprint,timed_signed_blocks=16,calls_per_method_per_schedule=48))
if __name__=='__main__':main()
