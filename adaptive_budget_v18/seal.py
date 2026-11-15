import os,time
os.environ['BELIEF_DEVICE']='cpu'
from common import *
from run_policy import CELLS,EIDX,ADAPT,MIX,NEST,FIXED
from analyze import METHODS,choose
require();checks={}
for p,h in read(R/'source_policy.json').items():assert sha(p)==h,p
checks['policy_sources_unchanged']=True
for name in ['policy_freeze','independent_audit_freeze']:
    f=read(R/(name+'.json'));assert not f['private_outcomes_read']
    for p,h in f['files'].items():assert sha(p)==h,p
assert 'private.npz' not in Path('run_policy.py').read_text() and 'private.npz' not in Path('audit_independent.py').read_text()
assert read(R/'policy_freeze.json')['time']<read(R/'evaluation_done.json')['time'];checks['policy_freezes_verified']=True
pr=np.load(R/'data/test/private.npz');pu=np.load(R/'data/test/public.npz');L=np.load(R/'world_losses.npz');A=np.load(R/'decision_actions.npz');mask=pr['theta'][:,1]!=2
assert np.array_equal(pu['context'][:,0],pu['context'][:,1]);assert np.array_equal(pu['factual'][mask,0],pu['factual'][mask,1])
for key in ['target','obs1','obs2','obs3']:assert np.array_equal(pr[key][mask,0],pr[key][mask,1])
for cell,cond,law in CELLS:
    p=np.load(R/'policies'/f'{cell}.npz');target=pr['target'][:,cond][:,EIDX]
    for N in [16384,65536]:
        pred=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];assert pred.shape==(512,4,16,2) and np.isfinite(pred).all() and pred.min()>-1e-12 and pred.max()<2**.5+1e-12
        la=(.5*(pred-target[:,None,None,:])**2).mean(2)
        for m in METHODS:
            if (m=='qmc64_native')!=(N==65536):continue
            assert np.allclose(choose(la,A[cell+'__'+m]),L[cell+'__'+m],rtol=0,atol=1e-14)
    for name in MIX:
        for i in range(512):
            for k in range(2):
                meta=p[name+'_meta'][i,k];h=p[name+'_history'][i,k,int(meta[2])-1];r=p[name+'_risk'][i,k];reason=int(meta[3]);K,m=int(meta[0]),int(meta[1]);act=int(p[name][i,k])
                assert act==(1 if np.ptp(r)<=1e-12 else int(r.argmin())+1)
                assert np.allclose(h[:3],r,rtol=0,atol=1e-14) and abs(h[8]-(h[5]-h[6]-h[7]))<1e-12
                assert p[name+'_work'][i,k]==2*3*4*m*K*meta[6]
                if reason==0:assert h[8]>0
                elif reason==1:assert (name=='component_only' and K==128) or (K==128 and m==1024)
                elif reason==2:assert np.ptp(r)<=1e-12
                else:assert reason==3 and name in FIXED
for name in METHODS:assert np.array_equal(L['matched_nominal__'+name][mask],L['shifted_nominal__'+name][mask]),name
checks['losses_probabilities_work_and_stopping_verified']=True;checks['negative_controls']=int(mask.sum())
v=read(R/'validation.json');assert v['cached_dense_max_error']<1e-10 and v['label_mass'] and v['full_inner_prefix'] and v['routing'];checks['correctness_verified']=True
for dev in ['gpu','cpu']:
    f=read(R/f'timing_fast_{dev}_done.json');assert f['source']==sha('timing_fast.py') and f['runtime']==sha('runtime_fast.py') and f['budget']==sha('budget_fast.py')
    for name,x in read(R/f'timing_fast_{dev}.json').items():
        assert x['n']==120 and x['max_prediction_error']<1e-9
        if dev=='gpu':assert x['action_mismatches']==[]
eq=read(R/'fast_equivalence.json');assert eq['cases']==30720 and eq['max_risk_error']<1e-10 and eq['source']==sha('budget_fast.py');checks['equivalent_fast_cases']=eq['cases']
for name in ['gpu_worker_exit','fast_worker_exit']:
    e=read(R/(name+'.json'));assert e['code']==0 and e['foreign']==[]
checks['workers_success']=True
old=Path('../adaptive_acquisition_v17');manifest=read(old/'MANIFEST.json')
for p,h in manifest.items():assert sha(old/p)==h,('v17 changed',p)
checks['v17_files_unchanged']=len(manifest);checks['completed']=time.time();write('INTEGRITY.json',checks)
files=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json'];write('MANIFEST.json',{str(p):sha(p) for p in sorted(files)});print(dict(checks=checks,files=len(files)))
