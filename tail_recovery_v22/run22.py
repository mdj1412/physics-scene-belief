from pathlib import Path
import os
import numpy as np
from policy22 import RecoveryEngine, PAIRS,PILOT,KGRID,TASK,t,torch,action,DT
from run_v21 import public,private,online_times
from data_v21 import generate,LAWS
from model import noise
from common import R,read,write,save,require,sha,freeze

def calibration():
    generate("pilot",32,26071137,["nominal"])
    engine=RecoveryEngine(PILOT);xy=public("pilot");obs=private("pilot")
    names=[f"{family}{k}m2048" for family in ("top","keep") for k in KGRID]+["fixed"]
    def measure(names,ids,reps):
        tm=online_times(engine,xy,obs,names,[(i,0) for i in ids],reps)
        return dict(zip(names,np.quantile(tm.reshape(-1,len(names)),.9,axis=0)))
    first=measure(names,range(16),2)
    short={f:sorted([n for n in names if n.startswith(f)],key=lambda n:abs(np.log(first[n]/first["fixed"])))[:4] for f in ("top","keep")}
    second=measure(short["top"]+short["keep"]+["fixed"],range(16,32),3)
    opts=[]
    for top in short["top"]:
        for keep in short["keep"]:
            x,y=second[top]/second["fixed"],second[keep]/second["fixed"]
            ratios=np.array([x,y,y/x]);err=float(np.abs(np.log(ratios)).max())
            near=bool(np.all((ratios>=.9)&(ratios<=1.1)))
            opts.append((not near,not(x<=1 and y<=1),err,top,keep))
    _,_,_,top,keep=min(opts)
    cfg=dict(top=top,keep=keep,first=first,second=second)
    write(R/"design.json",cfg)
    state=engine.fit(xy[0][0,0],xy[1][0,0],TASK,True)
    for k in [512,int(keep[4:].split("m")[0])]:
        ow,oz=state[2:]
        ix,ww=engine.support(ow,oz,k)
        assert len(ww)==k and abs(float(ww.sum())-1)<1e-10
        for mask in [oz[:,0]>0,oz[:,0]==0]:
            assert abs(float(ww[mask[ix]].sum()-ow[mask].sum()))<1e-10
    print("Locked",top,keep,second,flush=True)
    return cfg

# Columns are float64 except discrete quantities cast for compact storage.
COLUMNS=["tail_before","tail_after","any_atom","max_tail_atom","winner_rank","winner_fold","p_full","p_top","p_keep","surprise","label_separation","retained_tail_mass"]

def diagnostic(engine,state,y,a,k,ks):
    _,_,ow,oz=state;labels=(oz[:,0]>0).to(DT)
    bank=engine.outer
    sorted_idx=torch.argsort(ow,descending=True)
    head=sorted_idx[:k]
    tail=sorted_idx[k:]
    prior=ow.clamp_min(1e-300).log()
    logits=bank.ll(a,y)+prior
    post=logits.softmax(-1)
    removed=post[:,tail]
    tail_mass=removed.sum(1)
    orig=ow[tail]
    fold=removed/orig.clamp_min(1e-300)
    max_atom,position=removed.max(1)
    entrants=((removed>=.01)&(fold>=10)).any(1)
    plausible=logits[:,head].softmax(-1)@labels[head]
    p=post@labels
    tail_label=logits[:,tail].softmax(-1)@labels[tail]
    assert float(((p-plausible)-tail_mass*(tail_label-plausible)).abs().max())<1e-9
    idx,weight=engine.support(ow,oz,ks)
    lp=bank.ll(a,y)[:,idx]+weight.log()
    preserved=lp.softmax(-1)@labels[idx]
    surprise=.5*(y/noise(a)).square().sum(-1)-torch.logsumexp(logits,-1)
    represented=torch.zeros_like(ow,dtype=torch.bool)
    represented[idx[weight>0]]=True
    retained=removed@represented[tail].to(DT)
    fields=[orig.sum().expand_as(tail_mass),tail_mass,entrants,max_atom,position+k+1,
            fold.gather(1,position[:,None])[:,0],p,plausible,preserved,surprise,tail_label-plausible,retained]
    return torch.stack(fields,-1).cpu().numpy()

def main():
    require();torch.set_num_threads(2);assert torch.cuda.is_available()
    write(R/"device.json",dict(job=os.environ["SLURM_JOB_ID"],node=os.environ["SLURMD_NODENAME"],gpu=torch.cuda.get_device_name()))
    src=[Path(n) for n in ("policy22.py","run22.py","stats22.py","PROTOCOL.md","../external_label_v21/v21_policy.py","../external_label_v21/data_v21.py","../external_label_v21/run_v21.py","../external_label_v21/analyze_v21.py","../adaptive_budget_v18/model.py","../conditional_risk_v19/selector.py","../factual_switch_v20/policy.py")]
    write(R/"sources.json",{str(p):sha(p) for p in src})
    cfg=calibration()
    names=["top512m512","keep512m512","E-E-mean",cfg["top"],cfg["keep"],"fixed","probe"]
    write(R/"methods.json",names)
    freeze(R/"plan.json",[R/"sources.json",R/"design.json",R/"methods.json"])
    generate("test",512,27071917,LAWS)
    xy=public("test");x,ctx=xy
    paths=[R/"plan.json",R/"test/public.npz"]
    for b,pair in enumerate(PAIRS):
        eng=RecoveryEngine(pair)
        scores=np.empty((len(x),3,len(names)-1,3))
        for i in range(len(x)):
            for l in range(3):
                state=eng.fit(x[i,l],ctx[i,l],TASK,True)
                scores[i,l]=torch.stack([eng.risk(state,n) for n in names[:-1]]).cpu().numpy()
        assert np.isfinite(scores).all()
        decisions=np.where(np.ptp(scores,axis=-1)<=1e-12,1,scores.argmin(-1)+1)
        decisions=np.concatenate([decisions,np.full((*decisions.shape[:2],1),3)],axis=2)
        path=R/f"choices_{b}.npz";save(path,acts=decisions,risk=scores);paths.append(path)
        del eng;print("Policies",b+1,flush=True)
    freeze(R/"choices_freeze.json",paths)
    obs=private("test");write(R/"columns.json",COLUMNS)
    kt=int(cfg["top"][3:].split("m")[0]);kp=int(cfg["keep"][4:].split("m")[0])
    for b,pair in enumerate(PAIRS):
        eng=RecoveryEngine(pair)
        acts=np.load(R/f"choices_{b}.npz")["acts"]
        loss=np.empty((len(x),3,3,16))
        detail=np.empty((len(x),3,2,3,16,len(COLUMNS)),np.float32)
        for i in range(len(x)):
            for l in range(3):
                st=eng.fit(x[i,l],ctx[i,l],TASK,True)
                for a in (1,2,3):
                    y=obs[f"obs{a}"][i,l];w,z=st[:2]
                    p=eng.inner.update(w,a,y)@z[:,0]/np.sqrt(2)
                    loss[i,l,a-1]=(p-obs["label"][i,l]).square().cpu().numpy()
                    for g,(k,ks) in enumerate([(512,512),(kt,kp)]):
                        detail[i,l,g,a-1]=diagnostic(eng,st,y,a,k,ks)
        timed=names[3:]
        ti=online_times(eng,xy,obs,timed,[(i,l) for i in range(24) for l in range(3)],2,acts[:,:,3:])
        save(R/f"evaluation_{b}.npz",loss=loss,diagnostics=detail,times=ti.reshape(24,3,2,len(timed)))
        del eng;print("Evaluated",b+1,flush=True)
if __name__=="__main__":main()
