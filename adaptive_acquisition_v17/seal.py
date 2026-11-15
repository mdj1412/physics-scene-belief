import os,time
os.environ['BELIEF_DEVICE']='cpu'
from common import *
from run_policy import CELLS,ETASKS,EIDX,ADAPT
from analyze import METHODS,choose
require();checks={}
for p,h in read(R/'source_policy.json').items():assert sha(p)==h,p
checks['policy_sources_unchanged']=True
freeze=read(R/'policy_freeze.json');assert not freeze['private_outcomes_read'] and freeze['time']<read(R/'evaluation_done.json')['time']
for p,h in freeze['files'].items():assert sha(p)==h,p
assert 'private.npz' not in Path('run_policy.py').read_text();checks['public_policy_freeze_verified']=True
truth=np.load(R/'data/test/private.npz');pub=np.load(R/'data/test/public.npz');L=np.load(R/'world_losses.npz');A=np.load(R/'decision_actions.npz');mask=truth['theta'][:,1]!=2
assert len(truth['theta'])==512 and np.array_equal(pub['context'][:,0],pub['context'][:,1])
assert np.array_equal(pub['factual'][mask,0],pub['factual'][mask,1])
for key in ['target','obs1','obs2','obs3']:assert np.array_equal(truth[key][mask,0],truth[key][mask,1])
for cell,cond,law in CELLS:
    pol=np.load(R/'policies'/f'{cell}.npz');tt=truth['target'][:,cond][:,EIDX]
    for N in [16384,65536]:
        pred=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];assert pred.shape==(512,4,16,2) and np.isfinite(pred).all()
        assert pred.min()>-1e-12 and pred.max()<2**.5+1e-12
        la=(.5*(pred-tt[:,None,None,:])**2).mean(2)
        for m in METHODS:
            if (m=='qmc64_native')!=(N==65536):continue
            assert np.allclose(choose(la,A[cell+'__'+m]),L[cell+'__'+m],atol=1e-14,rtol=0),m
    for m in ADAPT:
        meta=pol[m+'_meta'];hist=pol[m+'_history'];risk=pol[m+'_risk'];acts=pol[m].astype(int)
        assert ((acts>=1)&(acts<=3)).all()
        for i in range(512):
            for k in range(2):
                mm=meta[i,k];last=hist[i,k,int(mm[1])-1];reason=int(mm[2]);assert np.allclose(last[:3],risk[i,k])
                expected=1 if np.ptp(risk[i,k])<=1e-12 else int(risk[i,k].argmin())+1
                assert acts[i,k]==expected
                if reason==0:assert mm[1]>=2 and last[8]>0
                elif reason==1:assert mm[0]==128 and last[8]<=0
                elif reason==2:assert np.ptp(risk[i,k])<=1e-12
                else:assert reason==3 and mm[1]==1
for m in METHODS:assert np.array_equal(L['matched_nominal__'+m][mask],L['shifted_nominal__'+m][mask]),m
checks['losses_probabilities_stopping_rules_verified']=True;checks['negative_controls']=int(mask.sum())
assert read(R/'validation.json')['label_mass_ties_optimized_native_verified'];checks['mechanism_validation']=True
for dev in ['gpu','cpu']:
    done=read(R/f'timing_{dev}_done.json');assert done['source']==sha('timing.py') and done['runtime']==sha('runtime.py') and done['adaptive']==sha('adaptive.py')
    for m,v in read(R/f'timing_{dev}.json').items():
        assert v['n']==120 and v['max_prediction_error']<1e-9
        if dev=='gpu':assert v['action_mismatches']==[]
for dev in ['gpu','cpu']:
    done=read(R/f'timing_fast_{dev}_done.json');assert done['source']==sha('timing_fast.py') and done['runtime']==sha('runtime_fast.py') and done['adaptive']==sha('adaptive_fast.py')
    for m,v in read(R/f'timing_fast_{dev}.json').items():
        assert v['n']==120 and v['max_prediction_error']<1e-9
        if dev=='gpu':assert v['action_mismatches']==[]
fast=read(R/'fast_equivalence.json');assert fast['cases']==15360 and fast['max_risk_error']<1e-14 and fast['source']==sha('adaptive_fast.py')
exit_fast=read(R/'fast_worker_exit.json');assert exit_fast['code']==0 and exit_fast['foreign']==[]
checks['all_policy_fast_path_equivalent']=15360
checks['online_paths_verified']=True
e=read(R/'gpu_worker_exit.json');assert e['code']==0 and e['foreign']==[];checks['gpu_worker_success']=True
old=Path('../region_mixture_v16');manifest=read(old/'MANIFEST.json')
for p,h in manifest.items():assert sha(old/p)==h,('v16 changed',p)
checks['v16_files_unchanged']=len(manifest);checks['completed']=time.time();write('INTEGRITY.json',checks)
files=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
write('MANIFEST.json',{str(p):sha(p) for p in sorted(files)});print(dict(checks=checks,files=len(files)),flush=True)
