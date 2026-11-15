import os,time,ast
os.environ['BELIEF_DEVICE']='cpu'
import numpy as np
from common import *
from model import TASKS,SIZES
from run_policy import CELLS
from analyze import by_action_loss,select_loss,action_for,METHODS
require();checks={}
for p,h in read(R/'source_policy.json').items():assert sha(p)==h,p
checks['policy_source_unchanged']=True
f=read(R/'policy_freeze.json')
for p,h in f['files'].items():assert sha(p)==h,p
assert not f['private_outcomes_read'];assert f['time']<read(R/'evaluation_done.json')['time']
assert 'private.npz' not in Path('run_policy.py').read_text()
checks['public_only_policy_and_freeze_verified']=True
pub=np.load(R/'data/test/public.npz');priv=np.load(R/'data/test/private.npz');control=priv['theta'][:,1]!=2
assert np.array_equal(pub['context'][:,0],pub['context'][:,1])
for name in ['target','obs1','obs2','obs3']:assert np.array_equal(priv[name][control,0],priv[name][control,1]),name
assert np.array_equal(pub['factual'][control,0],pub['factual'][control,1])
checks['paired_context_and_negative_controls_verified']=int(control.sum())
saved=np.load(R/'world_losses.npz');actions=np.load(R/'decision_actions.npz')
for cell,cond,law in CELLS:
    p=np.load(R/'policies'/f'{cell}.npz')
    for N in [16384]+([65536] if law=='nominal' else []):
        pred=np.load(R/'evaluations'/f'{cell}_{N}.npz')['pred'];assert pred.shape==(512,4,16,8) and np.isfinite(pred).all()
        assert pred[...,[4,7]].min()>-1e-12 and pred[...,[4,7]].max()<2**.5+1e-12
        la=by_action_loss(pred,priv['target'][:,cond])
        for name in METHODS:
            if name.startswith('qmc64')!=(N==65536):continue
            assert np.allclose(select_loss(la,actions[cell+'__'+name]),saved[cell+'__'+name],atol=1e-14,rtol=0),name
checks['all_predictions_losses_probabilities_verified']=True
for name in ['native','pair','probe','nested512']:
    assert np.array_equal(saved['matched_nominal__'+name][control],saved['shifted_nominal__'+name][control])
for device in ['gpu','cpu']:
    done=read(R/f'timing_{device}_done.json');assert done['source']==sha('benchmark.py')
    for name,value in read(R/f'timing_{device}.json').items():assert value['n']==300 and value['max_prediction_error']<1e-9
checks['online_gpu_cpu_timing_verified']=True
exitinfo=read(R/'gpu_worker_exit.json');assert exitinfo['code']==0 and exitinfo['foreign']==[]
checks['final_gpu_worker_success']=True
old=Path('../disagreement_design_v14');manifest=read(old/'MANIFEST.json')
for p,h in manifest.items():assert sha(old/p)==h,('v14 changed',p)
checks['v14_files_unchanged']=len(manifest);checks['sealed_at']=time.time();write('INTEGRITY.json',checks)
files=[p for p in Path('.').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='MANIFEST.json']
write('MANIFEST.json',{str(p):sha(p) for p in sorted(files)});print(dict(checks=checks,files=len(files)),flush=True)
