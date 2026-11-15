from experiment import Factorial, public_rows, actual_loss, generate, R, write, save, require, freeze, sha, t, TASK
from policy import design, predict, action
from common import read
from model import targets
import numpy as np
import torch
import time
from pathlib import Path

TEACHERS=["E-C-mean","fixed_global"]
POLICIES=["probe","fixed_global","E-C-mean","original_routed","aligned_E-C-mean","aligned_fixed_global"]

def teacher_risk(engine,state,name):
    if name=="fixed_global":
        return engine.baseline(state,"v18_global")
    return engine.risks(state,[name])[name]

def prep(engine,rows):
    features=[]; actions=[]
    for x,ctx in rows:
        state=engine.fit(x,ctx,TASK,True)
        features.append(engine.features(x,ctx,*state[:2]))
        actions.append([action(teacher_risk(engine,state,n)) for n in TEACHERS])
    return np.array(features),np.array(actions)

def fit_aligned(x,y):
    model={"mean":x.mean(0),"scale":np.maximum(x.std(0),1e-9)}
    mat=design(x,model)
    model.update(center=mat.mean(0),intercept=y.mean(0))
    f=mat-model["center"]
    model["beta"]=np.linalg.solve(f.T@f+.05*len(f)*np.eye(f.shape[1]),f.T@(y-model["intercept"]))
    return model

def limit(g):
    return {"all":-np.inf,"none":np.inf}.get(g["setting"],g["threshold"])

def optimized(g,a,loss):
    opt=[]
    for value in np.r_[-np.inf,0,np.quantile(g,[.25,.5,.75,.9]),np.inf]:
        active=g>value
        selected=np.where(active,a,3)
        opt.append((float(loss[np.arange(len(g)),selected-1].mean()),int(active.sum()),float(value)))
    val,n,cut=min(opt)
    return dict(setting="all" if cut==-np.inf else "none" if cut==np.inf else "threshold",threshold=cut if np.isfinite(cut) else None,
                active=n,val_loss=val,cuts=np.quantile(g,[.25,.5,.75]).tolist())

def measure(engine,rows,actions,models,cfg,original,orig_cfg):
    with np.load(R/"data/fresh/private.npz") as data:
        observations={a:t(data[f"obs{a}"][:32,0,0]) for a in (1,2,3)}
    elapsed={k:[] for k in POLICIES}
    rng=np.random.default_rng(984118)
    for rep in range(4):
        for i,(x,ctx) in enumerate(rows[:32 if rep else 1]):
            for name in rng.permutation(POLICIES):
                torch.cuda.synchronize()
                start=time.perf_counter()
                w,z=engine.fit(x,ctx,TASK,False)[:2]
                run=name
                if name=="original_routed" or name.startswith("aligned_"):
                    features=engine.features(x,ctx,w,z)
                    if name=="original_routed":
                        run=TEACHERS[0] if predict(features,original).max()>orig_cfg["threshold"] else "probe"
                    else:
                        teacher=name.replace("aligned_","")
                        value=predict(features,models)[TEACHERS.index(teacher)]
                        run=teacher if value>limit(cfg[teacher]) else "probe"
                a=3
                if run!="probe":
                    state=(w,z,engine.outer.fit(x),targets(engine.outer.theta,ctx,task=TASK))
                    a=action(teacher_risk(engine,state,run))
                estimate=engine.inner.update(w,a,observations[a][i])@z
                torch.cuda.synchronize()
                time_ms=(time.perf_counter()-start)*1000
                assert a==actions[i,POLICIES.index(name)]
                if rep:elapsed[name].append(time_ms)
    return {k:[float(np.median(v)),float(np.quantile(v,.9))] for k,v in elapsed.items()}

def run():
    require()
    torch.set_num_threads(2)
    samp=R/"aligned"
    samp.mkdir(exist_ok=True)
    freeze(samp/"source.json",[Path("aligned.py"),Path("ALIGNED_PROTOCOL.md"),Path("aligned_analysis.py")])
    engine=Factorial()
    orig=dict(np.load(R/"ridge.npz"))
    og=read(R/"gate.json")["routed"]
    assert og["method"]==TEACHERS[0]
    train=public_rows("train",0)
    x,a=prep(engine,train)
    train_loss=actual_loss(engine,"train",0,train)
    model=fit_aligned(x,train_loss[:,2,None]-np.take_along_axis(train_loss,a-1,axis=1))
    valid=np.load(R/"validation.npz")
    from policy import CELLS
    va=valid["choices"][:,[CELLS.index(TEACHERS[0]),8]]
    vg=predict(valid["features"],model)
    cfg={n:optimized(vg[:,j],va[:,j],valid["loss"]) for j,n in enumerate(TEACHERS)}
    save(samp/"ridge.npz",**model); write(samp/"gate.json",cfg)
    freeze(samp/"gate_freeze.json",[samp/"ridge.npz",samp/"gate.json",samp/"source.json"])
    print("Additional gates frozen",cfg,flush=True)
    generate(1024,15000011,"fresh")
    fresh=public_rows("fresh",0)
    x,teachers=prep(engine,fresh)
    g=predict(x,model)
    actions=np.column_stack([np.full(len(x),3),teachers[:,1],teachers[:,0],
                        np.where(predict(x,orig).max(1)>og["threshold"],teachers[:,0],3)]+
                       [np.where(g[:,j]>limit(cfg[n]),teachers[:,j],3) for j,n in enumerate(TEACHERS)])
    save(samp/"actions.npz",gain=g,choices=actions,teachers=teachers)
    freeze(samp/"choices_freeze.json",[samp/"actions.npz",samp/"gate_freeze.json",R/"data/fresh/public.npz"])
    save(samp/"loss.npz",loss=actual_loss(engine,"fresh",0,fresh))
    write(samp/"timing.json",measure(engine,fresh,actions,model,cfg,orig,og))
    print("Additional fresh-world test complete",flush=True)

if __name__=="__main__":
    run()
