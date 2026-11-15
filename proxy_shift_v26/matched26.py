"""Matched-prior secondary experiment. All choices are sealed before the evaluator loads outcomes."""
from pathlib import Path
import os
import numpy as np
from physics26 import Engine, variants, t, torch
from run26 import BANKS, CONDITIONS, frozen, timed
from common import freeze, require, save, sha, write

ROOT = Path(__file__).resolve().parent
R = Path('results')
M = R / 'matched'
LANDMARKS = [(1.,1.), (-1.,1.), (1.,-1.), (-1.,-1.), (0.,0.)]


def run():
    os.chdir(ROOT)
    require()
    torch.set_num_threads(2)
    assert torch.cuda.is_available()
    M.mkdir(exist_ok=True)
    conds = [tuple(c) for c in CONDITIONS]
    assert len(conds) == 36 and len(set(conds)) == 36 and (1.,1.) in conds
    timed_cond = np.asarray([conds.index(land) for land in LANDMARKS])
    model, cut = frozen()
    with np.load(R/'test/assignment.npz') as assignment:
        all_cell = assignment['cell']
        ids = np.sort(np.concatenate([np.flatnonzero(all_cell == c)[:64] for c in range(4)]))
        uniforms = assignment['u_flip'][ids]
        cell = all_cell[ids]
    with np.load(R/'test/public.npz') as d:
        x = d['x'][ids]
    n = len(ids)
    assert n == 256 and np.array_equal(np.bincount(cell), [64]*4)
    loc = np.arange(n)
    time_ids = np.sort(np.concatenate([np.flatnonzero(cell==c)[:4] for c in range(4)]))
    selection = np.stack([variants(uniforms,*c) for c in conds],axis=1)
    val = np.array([[float(model.score(row)) for row in world] for world in x])
    route = np.abs(val) >= cut
    direct = np.where(val >= 0, 1, 3)
    save(M/'cohort.npz',ids=ids,cell=cell,time_ids=time_ids,variant=selection,conditions=conds,
         timed_cond=timed_cond,confident=route[loc[:,None],selection])
    write(M/'sources.json',{str(p):sha(p) for p in (Path('matched26.py'),Path('physics26.py'),Path('run26.py'),R/'plan.json',R/'test/public.npz',R/'test/assignment.npz')})
    write(M/'device.json',dict(host=os.environ.get('SLURMD_NODENAME'),job=os.environ['SLURM_JOB_ID'],gpu=torch.cuda.get_device_name()))
    paths = [M/'cohort.npz', M/'sources.json']
    for b,pair in enumerate(BANKS[:3]):
        acts = np.empty((n,len(conds),2),dtype=np.int8)
        for j,condition in enumerate(conds):
            eng = Engine(pair,*condition)
            for i,v in enumerate(selection[:,j]):
                st = eng.fit(t(x[i,v]))
                acts[i,j,1] = eng.choose_pair(st,256)
                acts[i,j,0] = direct[i,v] if route[i,v] else eng.choose_pair(st,1024)
            del eng
        f = M/f'choices_{b}.npz'
        save(f,acts=acts)
        paths.append(f)
        print('Matched choices bank',b+1,flush=True)
    freeze(M/'choices_freeze.json',paths)
    with np.load(R/'test/private.npz') as data:
        labels = data['label'][ids]
        q = data['obs1'][ids]
        p = data['obs3'][ids]
    # Different conditions may share the same observed raw input through common flips. Their
    # posterior predictions still differ because the actual known prior changes per condition.
    for b,pair in enumerate(BANKS[:3]):
        acts = np.load(M/f'choices_{b}.npz')['acts']
        losses = np.empty(acts.shape)
        timing = np.empty((len(time_ids),2,len(timed_cond),2))
        for j,condition in enumerate(conds):
            eng = Engine(pair,*condition)
            vv = selection[:,j]
            xx = x[loc,vv]
            for i,v in enumerate(vv):
                w = eng.inner.fit(t(xx[i])); z = eng.inner.label[:,0]
                for action in np.unique(acts[i,j]):
                    yy = q[i,v] if action == 1 else p[i,v]
                    prob = eng.inner.update(w,int(action),t(yy)) @ z / np.sqrt(2.)
                    loss = float(((prob-float(labels[i,v]))**2).mean())
                    losses[i,j,acts[i,j]==action] = loss
            if j in timed_cond:
                k = int(np.flatnonzero(timed_cond == j)[0])
                timing[:,:,k] = timed(eng,model,cut,xx,q[loc,vv,0],p[loc,vv,0],time_ids,
                                      acts[:,j].astype(int),2,260330+b*100+j)
            del eng
        assert np.isfinite(losses).all() and np.isfinite(timing).all()
        save(M/f'selected_loss_{b}.npz',loss=losses)
        save(M/f'times_{b}.npz',times=timing,timed_cond=timed_cond,time_ids=time_ids)
        print('Matched loss bank',b+1,flush=True)
if __name__ == '__main__':
    run()
