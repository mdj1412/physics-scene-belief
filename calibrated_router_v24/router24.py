"""Prospective second gate after a zero-target development pilot."""
from pathlib import Path
import sys,os,time
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"cheap_tail_v24"))
import run24 as original
from run24 import R,require,write,save,freeze,sha,public,private,online_time,generate,DEV_CALS,TEST_CALS
from gate24 import Engine,Features,Gate,learn,REFERENCE
from physics24 import t,torch,DT,np
from policy22 import action

PAIRS=((64,768),(128,1024),(256,1536))
BANKS=[(34251001+b*13001,34362003+b*19001) for b in range(5)]
BUDGETS=(64,128,256,512,768,1024,1280,1536,1792,2048)
NAMES=original.NAMES
class RouterEngine(Engine):
    def __init__(self,pair,cal):
        super().__init__(pair,cal)
        for k in (64,128):
            n=k//8
            jitter=torch.rand(2,n,dtype=DT,generator=torch.Generator().manual_seed(798311+k))
            self.strata[k]=((torch.arange(n,dtype=DT)+jitter)/n).to(self.inner.theta.device)

def develop(split,n,seed,pair):
    generate(split,n,seed,DEV_CALS)
    xs=public(split);data=private(split)
    evaluated=sorted(set(k for pair in PAIRS for k in pair)|{REFERENCE})
    features=np.empty((n,2,6));risks=np.empty((n,2,len(evaluated),3));briers=np.empty(risks.shape[:-1])
    for j,cal in enumerate(DEV_CALS):
        eng=RouterEngine(pair,cal);feat=Features(cal)
        for i,x in enumerate(xs[:,j]):
            st=eng.fit(t(x));cost=original.loss(eng,st,data,i,j).mean(1)
            for col,k in enumerate(evaluated):
                r=eng.risk(st,f"keep{k}m2048")
                risks[i,j,col]=r.cpu().numpy()
                briers[i,j,col]=cost[action(r)-1]
            features[i,j]=feat(x)
    save(R/split/"development.npz",cost=briers,features=features,risks=risks,budgets=evaluated)
    return features,briers,risks,evaluated

def train():
    x,_,risk,ks=develop("train",288,341011,(3401003,3402003))
    xv,yv,_,_=develop("val",240,341021,(3404003,3405003))
    options=[]
    for low,high in PAIRS:
        dr=risk[:,: ,ks.index(low)]-risk[:,:,-1]
        err=np.sqrt(np.mean((dr-dr.mean(-1,keepdims=True))**2,axis=-1))
        target=np.log(err+1e-7)
        model=learn(x.reshape(-1,6),target.ravel());gate=Gate(model)
        ts=np.array([gate.score(v) for v in x.reshape(-1,6)])
        vs=np.array([gate.score(v) for v in xv.reshape(-1,6)]).reshape(xv.shape[:2])
        for q in (.4,.6,.8):
            cut=float(np.quantile(ts,q));on=vs>=cut;rate=on.mean(0);cost=low+on.mean()*(high-low)
            delta=(np.where(on,yv[:,:,ks.index(high)],yv[:,:,ks.index(low)])-yv[:,:,-1]).mean(0)
            worst=float(delta.max())
            admissible=bool(np.all((rate>=.10)&(rate<=.90)))
            key=(not admissible,worst>.0015,cost if worst<=.0015 else worst,worst if worst<=.0015 else cost)
            options.append((key,dict(low=low,high=high,static=min(BUDGETS,key=lambda k:(abs(k-cost),k)),
                    quantile=q,model=dict(model,cut=cut),val_diff=delta.tolist(),val_rate=rate.tolist(),
                    val_k=float(cost),train_target_sd=float(target.std()),valid=admissible)))
    cfg=min(options,key=lambda z:z[0])[1]
    write(R/"development.json",[v for _,v in options]);write(R/"selected.json",cfg)
    print("LOCKED",cfg["low"],cfg["high"],cfg["static"],cfg["val_rate"],cfg["val_diff"],flush=True)
    return cfg

def main():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    src=read_sources()
    write(R/"sources.json",src)
    write(R/"device.json",dict(gpu=torch.cuda.get_device_name(),host=os.environ["SLURMD_NODENAME"],job=os.environ["SLURM_JOB_ID"]))
    cfg=train();gate=Gate(cfg["model"])
    freeze(R/"plan.json",[R/"sources.json",R/"development.json",R/"selected.json"])
    generate("test",1152,341037,TEST_CALS);xs=public("test")
    paths=[R/"plan.json",R/"test/public.npz"]
    for b,pair in enumerate(BANKS):
        acts=np.empty((len(xs),3,len(NAMES)),int);fires=np.empty((len(xs),3),bool)
        for j,cal in enumerate(TEST_CALS):
            eng=RouterEngine(pair,cal);feat=Features(cal)
            for i,x in enumerate(xs[:,j]):
                on=gate.high(feat(x));fires[i,j]=on
                st=eng.fit(t(x))
                ks=[cfg["high"] if on else cfg["low"],REFERENCE,cfg["low"],cfg["high"],cfg["static"]]
                actions={k:eng.choose(st,k) for k in sorted(set(ks))}
                acts[i,j]=[actions[k] for k in ks]+[action(eng.risk(st,"fixed"))]
        f=R/f"choices_{b}.npz";save(f,acts=acts,fire=fires);paths.append(f)
        print("Choices",b+1,flush=True)
    freeze(R/"choices_freeze.json",paths)
    data=private("test");cell=data["cell"].cpu().numpy()
    ids=np.sort(np.concatenate([np.flatnonzero(cell==k)[:12] for k in range(6)]))
    for b,pair in enumerate(BANKS):
        acts=np.load(R/f"choices_{b}.npz")["acts"];cost=np.empty((len(xs),3,3,32));tim=[]
        for j,cal in enumerate(TEST_CALS):
            eng=RouterEngine(pair,cal)
            for i,x in enumerate(xs[:,j]):cost[i,j]=original.loss(eng,eng.fit(t(x)),data,i,j)
            tim.append(online_time(eng,xs,data,ids,cal,j,gate,cfg,acts[:,j]))
        save(R/f"eval_{b}.npz",loss=cost,times=np.stack(tim),ids=ids)
        print("Evaluation",b+1,flush=True)

def read_sources():
    import json
    base=Path("../cheap_tail_v24")
    files=[base/p for p in json.loads((base/"results/sources.json").read_text())]
    files.extend([Path("router24.py"),Path("PROTOCOL.md"),base/"stats24.py"])
    return {str(f):sha(f) for f in files}
if __name__=="__main__":main()
