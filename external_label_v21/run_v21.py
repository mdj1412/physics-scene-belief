import os,time
from pathlib import Path
import numpy as np
from v21_policy import StudyEngine,PAIRS,PILOT,CANDIDATES,CELLS,TASK,t,torch,action
from data_v21 import generate,LAWS
from common import R,save,write,read,freeze,sha,require


def public(split):
    with np.load(R/split/"public.npz") as data:return t(data["x"]),t(data["ctx"])


def private(split):
    with np.load(R/split/"private.npz") as data:return {n:t(data[n]) for n in data.files}


def world_loss(engine,x,ctx,data,i,l):
    w,z=engine.fit(x,ctx,TASK,False)[:2]
    pred=torch.stack([engine.inner.update(w,a,data[f"obs{a}"][i,l])@z[:,0] for a in (1,2,3)])/np.sqrt(2.)
    assert torch.isfinite(pred).all() and pred.min()>-1e-12 and pred.max()<1+1e-12
    return (pred-data["label"][i,l]).square().mean(1).cpu().numpy()


def online_times(engine,inputs,obs,names,indices,repeats,checks=None):
    x,ctx=inputs
    out=np.empty((len(indices),repeats,len(names)))
    rng=np.random.default_rng(383113)
    for rep in range(-1,repeats):
        for j,(i,l) in enumerate(indices[:1] if rep==-1 else indices):
            y=[obs[f"obs{a}"][i,l,0] for a in (1,2,3)]
            factual=x[i,l];context=ctx[i,l]
            for k in rng.permutation(len(names)):
                torch.cuda.synchronize()
                start=time.perf_counter()
                a,w,z=engine.online(factual,context,names[k])
                pred=engine.inner.update(w,a,y[a-1])@z
                torch.cuda.synchronize()
                ms=(time.perf_counter()-start)*1000
                if checks is not None:assert a==checks[i,l,k],(names[k],a,checks[i,l,k])
                if rep>=0:out[j,rep,k]=ms
    return out


def development():
    generate("development",128,18100111,["nominal"])
    xy=public("development");x,ctx=xy
    data=private("development")
    eng=StudyEngine(PILOT)
    names=CANDIDATES+["fixed"]
    times=online_times(eng,xy,data,names,[(i,0) for i in range(24)],3)
    losses=[]
    for i in range(len(x)):
        state=eng.fit(x[i,0],ctx[i,0],TASK,True)
        scores=world_loss(eng,x[i,0],ctx[i,0],data,i,0)
        losses.append([scores[action(eng.risk(state,n))-1] for n in names])
    full=eng.risks(state,["E-E-mean"])["E-E-mean"]
    dense,_=eng.compressed(state,eng.outer.N,512)
    err=float((full-dense).abs().max());assert err<1e-10,err
    averages=np.mean(losses,axis=0);p90=np.quantile(times.reshape(-1,len(names)),.9,axis=0)
    eligible=[i for i in range(len(CANDIDATES)) if p90[i]<=p90[-1]]
    idx=min(eligible,key=lambda i:(averages[i],p90[i])) if eligible else int(p90[:-1].argmin())
    choice=CANDIDATES[idx]
    write(R/"design.json",dict(selected=choice,eligible=[names[i] for i in eligible],max_error=err,
                               methods=names,development_loss=averages.tolist(),p90=p90.tolist()))
    freeze(R/"design_freeze.json",[R/"design.json",R/"sources.json"])
    print("FROZEN",choice,p90.tolist(),flush=True)
    return choice


def collect(engine,xy,names,selected):
    x,ctx=xy
    ks,ms=map(int,selected[3:].split("m"))
    choices=np.empty((len(x),len(LAWS),len(names)),int)
    omitted=np.empty((len(x),len(LAWS)))
    risks=np.empty((len(x),len(LAWS),len(names)-1,3))
    for i in range(len(x)):
        for l in range(len(LAWS)):
            state=engine.fit(x[i,l],ctx[i,l],TASK,True)
            scores=engine.risks(state,CELLS)
            scores["fixed"]=engine.risk(state,"fixed")
            scores[selected],om=engine.compressed(state,ks,ms)
            control=f"top{ks}m512"
            if control!=selected:scores[control]=engine.risk(state,control)
            rs=torch.stack([scores[n] for n in names[:-1]]).cpu().numpy()
            assert np.isfinite(rs).all()
            risks[i,l]=rs
            choices[i,l]=[*np.where(np.ptp(rs,axis=1)<=1e-12,1,rs.argmin(1)+1),3]
            omitted[i,l]=float(om)
        if (i+1)%128==0:print("world",i+1,flush=True)
    return choices,omitted,risks


def main():
    require();torch.set_num_threads(2)
    assert torch.cuda.is_available()
    paths=[Path(p) for p in ("v21_policy.py","data_v21.py","run_v21.py","analyze_v21.py","PROTOCOL.md",
                              "../factual_switch_v20/policy.py","../conditional_risk_v19/selector.py")]
    paths += [Path("../adaptive_budget_v18")/n for n in ["model.py","budget.py","budget_fast.py","guard.py","common.py","acquisition.py"]]
    write(R/"sources.json",{str(p):sha(p) for p in paths})
    write(R/"device.json",dict(node=os.environ["SLURMD_NODENAME"],gpu=torch.cuda.get_device_name(),job=os.environ["SLURM_JOB_ID"]))
    selected=development()
    generate("test",512,19100217,LAWS)
    xy=public("test")
    control=selected.split("m")[0]+"m512"
    names=CELLS+["fixed",selected]+([control] if control!=selected else [])+["probe"]
    write(R/"methods.json",names)
    paths=[R/"methods.json",R/"design_freeze.json",R/"test/public.npz"]
    for b,pair in enumerate(PAIRS):
        engine=StudyEngine(pair)
        acts,om,risks=collect(engine,xy,names,selected)
        path=R/f"policy_{b}.npz";save(path,acts=acts,omitted=om,risks=risks)
        del engine
        paths.append(path);print("bank",b,"complete",flush=True)
    freeze(R/"test_freeze.json",paths)
    print("All choices frozen",flush=True)
    data=private("test");x,ctx=xy
    timed=[selected,"fixed","E-E-mean","probe"]
    write(R/"timed_names.json",timed)
    for b,pair in enumerate(PAIRS):
        engine=StudyEngine(pair)
        scores=np.empty((len(x),len(LAWS),3))
        for i in range(len(x)):
            for l in range(len(LAWS)):
                scores[i,l]=world_loss(engine,x[i,l],ctx[i,l],data,i,l)
        decisions=np.load(R/f"policy_{b}.npz")["acts"][:,:,[names.index(n) for n in timed]]
        ti=online_times(engine,xy,data,timed,[(i,l) for i in range(16) for l in range(3)],2,decisions)
        save(R/f"eval_{b}.npz",loss=scores,times=ti.reshape(16,3,2,len(timed)))
        del engine
    print("Evaluation and online timing complete",flush=True)
if __name__=="__main__":main()
