import os,time,math
from pathlib import Path
import numpy as np
from policy23 import TailGate,action
from physics23 import generate,t,torch
from common import R,save,write,read,require,freeze,sha

PILOT=(237001,238003)
PAIRS=[(23141001+i*10111,23151007+i*13217) for i in range(4)]
BUDGETS=[(k,2048) for k in (2048,2560,3072,3328,3584,4096,4608)]+[(k,4096) for k in (1024,1280,1536,1792)]+[(6144,1024),(7168,1024)]
def public(split):
    return t(np.load(R/split/"public.npz")["x"])
def private(split):
    with np.load(R/split/"private.npz") as f:return {k:t(f[k]) for k in f.files}
def loss(engine,state,data,i,setting):
    w,z=state[:2]
    ps=torch.stack([engine.inner.update(w,a,data[f"obs{a}"][i,setting])@z[:,0]/np.sqrt(2) for a in (1,2,3)])
    assert ps.min()>-1e-10 and ps.max()<1+1e-10
    return (ps-data["label"][i]).square().cpu().numpy()
def timing(engine,x,data,ids,setting,names,repeats,expected=None):
    rng=np.random.default_rng(2390161)
    result=np.empty((len(ids),repeats,len(names)))
    for repeat in range(-1,repeats):
        for j,i in enumerate(ids[:2] if repeat<0 else ids):
            yy=[data[f"obs{a}"][i,setting,0] for a in (1,2,3)]
            fact=x[i]
            for k in rng.permutation(len(names)):
                torch.cuda.synchronize()
                start=time.perf_counter()
                a,w,z=engine.online(fact,None,names[k])
                _=engine.inner.update(w,a,yy[a-1])@z
                torch.cuda.synchronize()
                duration=(time.perf_counter()-start)*1000
                if repeat>=0:result[j,repeat,k]=duration
                if expected is not None:assert a==expected[i,k],(a,expected[i,k],names[k])
    return result

def calibrate():
    generate("train",96,23110617);generate("calibration",48,23120621)
    eng=TailGate(PILOT,0)
    train=public("train");d=private("train")
    eng.cutoff=float(np.median([float(eng.score(eng.fit(x))) for x in train]))
    names=[f"{family}{k}m{m}" for family in ("top","gate") for k,m in BUDGETS]+["fixed"]
    x=public("calibration");data=private("calibration")
    def measure(ids,names,reps):
        ti=timing(eng,x,data,ids,0,names,reps)
        return dict(zip(names,np.quantile(ti.reshape(-1,len(names)),.9,axis=0)))
    first=measure(list(range(24)),names,1)
    short={f:sorted((n for n in names if n.startswith(f)),key=lambda n:abs(math.log(first[n]/first["fixed"])))[:4] for f in ("top","gate")}
    finalists=short["top"]+short["gate"]+["fixed"]
    second=measure(list(range(24,48)),finalists,3)
    averages=[]
    for i,v in enumerate(train):
        s=eng.fit(v);cost=loss(eng,s,d,i,0).mean(1)
        averages.append([cost[action(eng.risk(s,n))-1] for n in finalists])
    dev=dict(zip(finalists,np.mean(averages,axis=0)))
    triples=[]
    for b in short["top"]:
        for g in short["gate"]:
            r=np.array([second[b]/second["fixed"],second[g]/second["fixed"],second[g]/second[b]])
            near=bool(np.all((r>=.94)&(r<=1.06)));gap=float(np.max(np.abs(np.log(r))))
            triples.append((not near,(dev[b]+dev[g]) if near else gap,gap,b,g))
    _,_,_,b,g=min(triples)
    cfg=dict(top=b,gate=g,cutoff=eng.cutoff,first=first,second=second,dev=dev)
    write(R/"design.json",cfg);print("LOCKED", b,g,[(n,round(second[n],3)) for n in (b,g,"fixed")],flush=True)
    return cfg

def main():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    write(R/"device.json",dict(job=os.environ["SLURM_JOB_ID"],node=os.environ["SLURMD_NODENAME"],gpu=torch.cuda.get_device_name()))
    sources=[*Path(".").glob("*.py"),Path("PROTOCOL.md")]
    sources+=[Path("../tail_recovery_v22/policy22.py"),Path("../external_label_v21/v21_policy.py"),Path("../factual_switch_v20/policy.py"),Path("../conditional_risk_v19/selector.py")]
    sources+=[Path("../adaptive_budget_v18")/f for f in ("model.py","common.py","budget.py","budget_fast.py","acquisition.py","guard.py")]
    write(R/"sources.json",{str(f):sha(f) for f in sources})
    cfg=calibrate();g=cfg["gate"]
    names=[g,cfg["top"],"fixed","top"+g[4:],"keep"+g[4:],"top512m512","gate512m512","E-E-mean"]
    write(R/"methods.json",names)
    freeze(R/"plan.json",[R/"sources.json",R/"design.json",R/"methods.json"])
    generate("test",512,23130731);x=public("test")
    locked=[R/"plan.json",R/"test/public.npz"]
    for b,pair in enumerate(PAIRS):
        acts=np.empty((len(x),2,len(names)),int); gate_score=np.empty(len(x))
        for setting in (0,1):
            eng=TailGate(pair,setting,cfg["cutoff"])
            for i,v in enumerate(x):
                st=eng.fit(v);score=float(eng.score(st))
                if setting==0:gate_score[i]=score
                else:assert abs(score-gate_score[i])<1e-11
                cache={}
                for j,n in enumerate(names):
                    if n.startswith("gate"):n=("keep" if score>=eng.cutoff else "top")+n[4:]
                    if n not in cache:cache[n]=action(eng.risk(st,n))
                    acts[i,setting,j]=cache[n]
                assert acts[i,setting,0]==acts[i,setting,4 if score>=eng.cutoff else 3]
        path=R/f"choices_{b}.npz";save(path,acts=acts,score=gate_score);locked.append(path)
        print("choices",b+1,flush=True)
    freeze(R/"choices_freeze.json",locked)
    data=private("test")
    for b,pair in enumerate(PAIRS):
        acts=np.load(R/f"choices_{b}.npz")["acts"]
        cost=np.empty((len(x),2,3,32)); times=[]
        for setting in (0,1):
            eng=TailGate(pair,setting,cfg["cutoff"])
            for i,v in enumerate(x):cost[i,setting]=loss(eng,eng.fit(v),data,i,setting)
            times.append(timing(eng,x,data,list(range(48)),setting,names[:3],3,acts[:,setting,:3]))
        save(R/f"evaluation_{b}.npz",loss=cost,times=np.stack(times))
        print("evaluation",b+1,flush=True)
if __name__=="__main__": main()
