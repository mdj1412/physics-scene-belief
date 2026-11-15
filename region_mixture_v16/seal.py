import os,time
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
from common import *
from model import TASKS
from analyze import group_loss,choose_loss,METHODS
from run_policy import CELLS
require();checks={}
for p,h in read(R/'source_policy.json').items():assert sha(p)==h,p
checks['policy_sources_unchanged']=True
f=read(R/'policy_freeze.json');assert not f['private_outcomes_read'] and f['time']<read(R/'evaluation_done.json')['time']
for p,h in f['files'].items():assert sha(p)==h,p
assert 'private.npz' not in Path('run_policy.py').read_text();checks['public_only_policy_freeze_verified']=True
truth=np.load(R/'data/test/private.npz');pub=np.load(R/'data/test/public.npz');mask=truth['theta'][:,1]!=2
assert np.array_equal(pub['context'][:,0],pub['context'][:,1]);assert np.array_equal(pub['factual'][mask,0],pub['factual'][mask,1])
for key in ['target','obs1','obs2','obs3']:assert np.array_equal(truth[key][mask,0],truth[key][mask,1])
L=np.load(R/'world_losses.npz');A=np.load(R/'decision_actions.npz')
for cell,cond,law in CELLS:
    for N in [16384]+([65536] if law=='nominal' else []):
        pred=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];assert pred.shape==(512,4,16,8) and np.isfinite(pred).all()
        assert pred[...,[4,7]].min()>-1e-12 and pred[...,[4,7]].max()<2**.5+1e-12
        la=group_loss(pred,truth['target'][:,cond,None,None,:]).mean(2)
        for m in METHODS:
            if m=='region32_compact' or (m=='qmc64_native')!=(N==65536):continue
            assert np.allclose(choose_loss(la,A[cell+'__'+m]),L[cell+'__'+m],atol=1e-14,rtol=0)
    cp=np.load(R/'evaluations'/f'{cell}_compact.npz')['pred'];assert np.isfinite(cp).all() and cp.shape==(512,16,8)
    assert np.allclose(group_loss(cp,truth['target'][:,cond,None,:]).mean(1),L[cell+'__region32_compact'],atol=1e-14,rtol=0)
for m in ['region32','global32','region64','region_cf32','native','pair','nested512']:assert np.array_equal(L['matched_nominal__'+m][mask],L['shifted_nominal__'+m][mask])
checks['losses_probabilities_and_negative_controls_verified']=int(mask.sum())
v=read(R/'validation.json');assert v['baseline_equivalence_and_event_mass_verified']
for name in ['v15_aliasing','higher_order']:
    r=v[name]['region_risks'];assert r[1]<r[2] and abs(r[0]-r[2])<1e-10
checks['counterexamples_and_baselines_verified']=True
for dev in ['gpu','cpu']:
    done=read(R/f'timing_{dev}_done.json');assert done['source']==sha('timing.py' if dev=='gpu' else 'timing_cpu_audit.py') and done['runtime']==sha('runtime.py') and done['mixture']==sha('mixture.py')
    for name,v in read(R/f'timing_{dev}.json').items():assert v['n']==300 and v['max_prediction_error']<1e-9
checks['online_gpu_cpu_paths_verified']=True
cm=read(R/'timing_cpu.json');assert sum(len(v['cross_device_action_mismatches']) for v in cm.values())==1
c=cm['region64']['cross_device_action_mismatches'][0];assert c['world']==11 and c['task']=='contact_velocity' and max(c['cpu_risks'])<1e-14 and max(c['gpu_risks'])<1e-14
checks['cpu_cross_device_near_zero_risk_tie_documented']=True
for name,v in read(R/'counterexample_global.json').items():
    r=v['global_risks'];assert r[1]<r[2] and abs(r[0]-r[2])<1e-10
checks['global_mixture_specificity_verified']=True
e=read(R/'gpu_worker_exit.json');assert e['code']==0 and e['foreign']==[];checks['gpu_worker_success']=True
old=Path('../transition_control_v15');m=read(old/'MANIFEST.json')
for p,h in m.items():assert sha(old/p)==h,('v15 modified',p)
checks['v15_files_unchanged']=len(m);checks['completed']=time.time();write('INTEGRITY.json',checks)
paths=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
write('MANIFEST.json',{str(p):sha(p) for p in sorted(paths)});print(dict(checks=checks,files=len(paths)),flush=True)
