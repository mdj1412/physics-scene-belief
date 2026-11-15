from policy import Factorial, CELLS, TASK, fit_gate, predict, action
from common import require, R, save, write, freeze, sha
from model import t
from generate import generate
import numpy as np
import torch
from pathlib import Path
import time
import os

NAMES = CELLS + ["fixed_global","probe","direct","routed"]
SEEDS = {"train":(1024,12000011),"validation":(384,13000011),"test":(1024,14000011)}


def public_rows(split, cond):
    with np.load(R/"data"/split/"public.npz") as data:
        return list(zip(t(data["factual"][:,cond]),t(data["context"][:,cond])))


def collect(engine, rows, full):
    features, choices, risks = [], [], []
    for i,(x,ctx) in enumerate(rows):
        state = engine.fit(x, ctx, TASK, full)
        w,z = state[:2]
        features.append(engine.features(x,ctx,w,z))
        if full:
            values = engine.risks(state,CELLS)
            values["fixed_global"] = engine.baseline(state,"v18_global")
            rs = torch.stack(list(values[n] for n in CELLS+["fixed_global"])).cpu().numpy()
            assert np.isfinite(rs).all() and rs.min() >= -1e-12 and rs.max() <= 1+1e-12
            choices.append(np.where(np.ptp(rs,axis=1)<=1e-12,1,rs.argmin(1)+1))
            risks.append(rs)
        if (i+1)%256 == 0:
            print("public",i+1,flush=True)
    return np.array(features),np.array(choices),np.array(risks)


def actual_loss(engine,split,cond,rows):
    with np.load(R/"data"/split/"private.npz") as private:
        ys = {a:t(private[f"obs{a}"][:,cond]) for a in (1,2,3)}
        truth = t(private["target"][:,cond,7]) / np.sqrt(2)
    loss = []
    for i,(x,ctx) in enumerate(rows):
        w,z = engine.fit(x,ctx,TASK,False)[:2]
        prob = torch.stack([engine.inner.update(w,a,ys[a][i])@z[:,0] for a in (1,2,3)])/np.sqrt(2)
        loss.append((prob-truth[i]).square().mean(1).cpu().numpy())
    out = np.array(loss)
    assert out.shape == (len(rows),3) and np.isfinite(out).all() and np.all((out>=0)&(out<=1+1e-12))
    return out


def apply_gate(gain, candidates, threshold):
    return np.where(gain.max(1)>threshold,candidates,3)


def select_gate(gain, choices, loss):
    thresholds = np.r_[0,np.quantile(gain.max(1),[.25,.5,.75,.9]),np.inf]
    fitted = {}
    for strategy in ["direct","routed"]:
        options = []
        candidates = [("direct",gain.argmax(1)+1)] if strategy=="direct" else list(zip(CELLS,choices.T))
        for order,(name,actions) in enumerate(candidates):
            for threshold in thresholds:
                aa = apply_gate(gain,actions,threshold)
                decision_loss = loss[np.arange(len(loss)),aa-1].mean()
                options.append((float(decision_loss),int((gain.max(1)>threshold).sum()),order,float(threshold),name))
        value, calls, _, threshold, name = min(options)
        fitted[strategy] = dict(method=name, threshold=None if np.isinf(threshold) else threshold,
                                validation_loss=value,active=calls)
    fitted["cuts"] = np.quantile(gain.max(1),[.25,.5,.75]).tolist()
    return fitted


def threshold(cfg):
    return np.inf if cfg["threshold"] is None else cfg["threshold"]


def test_actions(features, base, model, cfg):
    gain = predict(features,model)
    direct = apply_gate(gain,gain.argmax(1)+1,threshold(cfg["direct"]))
    routed = apply_gate(gain,base[:,CELLS.index(cfg["routed"]["method"])],threshold(cfg["routed"]))
    return gain,np.column_stack((base,np.full(len(base),3),direct,routed))


def online(engine,x,ctx,name,model,cfg):
    need_outer = name=="fixed_global" or name in CELLS and "E" in name.split("-")[:2]
    state = engine.fit(x,ctx,TASK,need_outer)
    w,z = state[:2]
    if name in ("direct","routed"):
        gain = predict(engine.features(x,ctx,w,z),model)
        if gain.max() <= threshold(cfg[name]):
            return 3,w,z
        if name=="direct":
            return int(gain.argmax())+1,w,z
        name=cfg["routed"]["method"]
        if "E" in name.split("-")[:2]:
            from model import targets
            state = (w,z,engine.outer.fit(x),targets(engine.outer.theta,ctx,task=TASK))
    if name=="probe":
        a=3
    elif name=="fixed_global":
        a=action(engine.baseline(state,"v18_global"))
    else:
        a=action(engine.risks(state,[name])[name])
    return a,w,z


def timing(engine,rows,model,cfg,expected):
    with np.load(R/"data/test/private.npz") as private:
        obs={a:t(private[f"obs{a}"][:32,0,0]) for a in (1,2,3)}
    output={n:[] for n in NAMES}
    rng=np.random.default_rng(12012)
    for rep in range(4):
        for i,(x,ctx) in enumerate(rows[:1 if rep==0 else 32]):
            for name in rng.permutation(NAMES):
                torch.cuda.synchronize()
                start=time.perf_counter()
                a,w,z=online(engine,x,ctx,name,model,cfg)
                estimate=engine.inner.update(w,a,obs[a][i])@z
                torch.cuda.synchronize()
                elapsed=(time.perf_counter()-start)*1000
                assert a==expected[i,NAMES.index(name)],(name,i,a)
                if rep:
                    output[name].append(elapsed)
    return {n:{"p50":float(np.median(v)),"p90":float(np.quantile(v,.9)),"mean":float(np.mean(v)),"n":len(v)} for n,v in output.items()}


def main():
    require()
    assert torch.cuda.is_available()
    torch.set_num_threads(2)
    write(R/"device.json",dict(gpu=torch.cuda.get_device_name(),node=os.environ["SLURMD_NODENAME"],job=os.environ["SLURM_JOB_ID"]))
    files = [Path(p) for p in ("policy.py","experiment.py","analyze.py","PROTOCOL.md","../conditional_risk_v19/selector.py")]
    files += [Path("../adaptive_budget_v18")/p for p in ("model.py","budget.py","budget_fast.py","generate.py","common.py","guard.py","acquisition.py")]
    write(R/"sources.json",{str(p):sha(p) for p in files})
    seeds=[s+137*i for n,s in SEEDS.values() for i in range(n)]
    assert len(set(seeds))==len(seeds)
    for split,(n,seed) in SEEDS.items():
        generate(n,seed,split)
    engine=Factorial()
    train=public_rows("train",0)
    x,_,_=collect(engine,train,False)
    loss=actual_loss(engine,"train",0,train)
    model=fit_gate(x,loss)
    save(R/"ridge.npz",**model)
    valid=public_rows("validation",0)
    xf,choices,risk=collect(engine,valid,True)
    vl=actual_loss(engine,"validation",0,valid)
    vg=predict(xf,model)
    cfg=select_gate(vg,choices,vl)
    write(R/"gate.json",cfg)
    save(R/"validation.npz",features=xf,choices=choices,risks=risk,loss=vl,gain=vg)
    freeze(R/"gate_freeze.json",[R/"ridge.npz",R/"gate.json",R/"sources.json"])
    print("Gate frozen:",cfg,flush=True)
    files=[]
    for cond in (0,1):
        rows=public_rows("test",cond)
        features,base,risk=collect(engine,rows,True)
        gain,choices=test_actions(features,base,model,cfg)
        path=R/f"choices_{cond}.npz"
        save(path,features=features,gain=gain,actions=choices,risk=risk)
        files.append(path)
    freeze(R/"choices_freeze.json",files+[R/"gate_freeze.json",R/"data/test/public.npz"])
    print("Test choices frozen",flush=True)
    for cond in (0,1):
        rows=public_rows("test",cond)
        save(R/f"loss_{cond}.npz",loss=actual_loss(engine,"test",cond,rows))
    exp=np.load(files[0])["actions"]
    write(R/"timing.json",timing(engine,public_rows("test",0),model,cfg,exp))
    print("Timing complete",flush=True)

if __name__=="__main__":
    main()
