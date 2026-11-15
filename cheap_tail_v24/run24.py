from pathlib import Path
import os,time
from gate24 import Engine,Gate,Features,learn,loss,BUDGETS,PAIRS,REFERENCE
from physics24 import generate,DEV_CALS,TEST_CALS,t,torch,np
from policy22 import action
from common import R,require,write,read,save,freeze,sha

BANKS=[(2451001+11003*b,2462003+13001*b) for b in range(5)]
NAMES=["gate","reference","low","high","static","global"]
def public(split):
    with np.load(R/split/"public.npz") as f:return f["x"]
def private(split):
    with np.load(R/split/"private.npz") as f:return {k:t(f[k]) for k in f.files}
def develop(split,n,seed,pair):
    generate(split,n,seed,DEV_CALS)
    xs=public(split);data=private(split)
    costs=np.empty((n,2,len(BUDGETS)));features=np.empty((n,2,6))
    for j,cal in enumerate(DEV_CALS):
        eng=Engine(pair,cal);feat=Features(cal)
        for i,x in enumerate(xs[:,j]):
            st=eng.fit(t(x));out=loss(eng,st,data,i,j).mean(1)
            costs[i,j]=[out[eng.choose(st,k)-1] for k in BUDGETS]
            features[i,j]=feat(x)
    save(R/split/"development.npz",cost=costs,features=features)
    return features,costs

def train():
    x,y=develop("train",240,241011,(2401003,2402003))
    xv,yv=develop("val",192,241021,(2404003,2405003))
    choices=[]
    for lo,hi in PAIRS:
        low=BUDGETS.index(lo);high=BUDGETS.index(hi)
        model=learn(x.reshape(-1,6),(y[:,:,low]-y[:,:,high]).ravel())
        gate=Gate(model)
        scores=np.array([gate.score(a) for a in x.reshape(-1,6)])
        val=np.array([gate.score(a) for a in xv.reshape(-1,6)]).reshape(xv.shape[:2])
        for q in (.4,.6,.8):
            cut=float(np.quantile(scores,q));fire=val>=cut
            delta=(np.where(fire,yv[:,:,high],yv[:,:,low])-yv[:,:,-1]).mean(0)
            cost=lo+float(fire.mean())*(hi-lo);worst=float(delta.max())
            key=(worst>.0015,cost if worst<=.0015 else worst,worst if worst<=.0015 else cost)
            static=min(BUDGETS,key=lambda k:(abs(k-cost),k))
            choices.append((key,dict(low=lo,high=hi,static=static,quantile=q,
                  model=dict(model,cut=cut),val_diff=delta.tolist(),val_rate=fire.mean(0).tolist(),val_k=cost)))
    selected=min(choices,key=lambda row:row[0])[1]
    write(R/"development.json",[v for _,v in choices]);write(R/"selected.json",selected)
    print("Frozen budgets",selected["low"],selected["high"],selected["static"],selected["val_rate"],flush=True)
    return selected

def online_time(eng,xs,data,ids,cal,j,gate,cfg,acts):
    feat=Features(cal);rng=np.random.default_rng(239888+j)
    block=np.empty((len(ids),4,5))
    # Realized observations are simulated, staged as host sensor buffers before the timer.
    sensor=[data[f"obs{a}"][:,j,0].cpu().numpy() for a in (1,2,3)]
    for rep in range(-1,4):
        for row,i in enumerate(ids[:4] if rep<0 else ids):
            x=xs[i,j];views=[v[i] for v in sensor]
            for m in rng.permutation(5):
                torch.cuda.synchronize()
                start=time.perf_counter()
                if m==0:k=cfg["high"] if gate.high(feat(x)) else cfg["low"]
                else:k=(REFERENCE,cfg["low"],cfg["high"],cfg["static"])[m-1]
                st=eng.fit(t(x));a=eng.choose(st,k)
                _=eng.inner.update(st[0],a,t(views[a-1]))@st[1]
                torch.cuda.synchronize()
                duration=(time.perf_counter()-start)*1000
                assert a==acts[i,m]
                if rep>=0:block[row,rep,m]=duration
    return block

def main():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    write(R/"device.json",dict(gpu=torch.cuda.get_device_name(),host=os.environ["SLURMD_NODENAME"],job=os.environ["SLURM_JOB_ID"]))
    files=[Path(f) for f in ("PROTOCOL.md","run24.py","gate24.py","physics24.py")]
    files += [Path("../factual_tail_v23")/f for f in ("run23.py","physics23.py","policy23.py")]
    files += [Path(p) for p in ("../tail_recovery_v22/policy22.py","../external_label_v21/v21_policy.py","../factual_switch_v20/policy.py","../conditional_risk_v19/selector.py")]
    files += [Path("../adaptive_budget_v18")/p for p in ("model.py","common.py","budget.py","budget_fast.py","acquisition.py","guard.py")]
    write(R/"sources.json",{str(f):sha(f) for f in files})
    cfg=train();gate=Gate(cfg["model"])
    freeze(R/"plan.json",[R/"sources.json",R/"selected.json",R/"development.json"])
    generate("test",1152,241037,TEST_CALS);xs=public("test")
    paths=[R/"plan.json",R/"test/public.npz"]
    for b,pair in enumerate(BANKS):
        acts=np.empty((len(xs),3,len(NAMES)),int);fires=np.empty((len(xs),3),bool)
        for j,cal in enumerate(TEST_CALS):
            eng=Engine(pair,cal);feat=Features(cal)
            for i,x in enumerate(xs[:,j]):
                fire=gate.high(feat(x));fires[i,j]=fire
                st=eng.fit(t(x));budgets=[cfg["high"] if fire else cfg["low"],REFERENCE,cfg["low"],cfg["high"],cfg["static"]]
                picks={k:eng.choose(st,k) for k in sorted(set(budgets))}
                acts[i,j]=[picks[k] for k in budgets]+[action(eng.risk(st,"fixed"))]
        f=R/f"choices_{b}.npz";save(f,acts=acts,fire=fires);paths.append(f)
        print("Choices",b+1,flush=True)
    freeze(R/"choices_freeze.json",paths)
    data=private("test");cells=data["cell"].cpu().numpy()
    ids=np.sort(np.concatenate([np.flatnonzero(cells==c)[:12] for c in range(6)]))
    for b,pair in enumerate(BANKS):
        acts=np.load(R/f"choices_{b}.npz")["acts"];cost=np.empty((len(xs),3,3,32));tim=[]
        for j,cal in enumerate(TEST_CALS):
            eng=Engine(pair,cal)
            for i,x in enumerate(xs[:,j]):
                cost[i,j]=loss(eng,eng.fit(t(x)),data,i,j)
            tim.append(online_time(eng,xs,data,ids,cal,j,gate,cfg,acts[:,j]))
        save(R/f"eval_{b}.npz",loss=cost,times=np.stack(tim),ids=ids)
        print("Evaluation",b+1,flush=True)
if __name__=="__main__":main()
