import numpy as np
from experiment import CELLS, NAMES, R, require, write, threshold, sha
from common import read


def ci(values, alpha=.05):
    if len(values)==0:
        return None
    rng=np.random.default_rng(712553)
    draws=[]
    for _ in range(40):
        draws.extend(values[rng.integers(len(values),size=(500,len(values)))].mean(1))
    return [float(values.mean()), *np.quantile(draws,[alpha/2,1-alpha/2]).tolist()]


def analyze(cond,cfg):
    rec=np.load(R/f"choices_{cond}.npz")
    acts=rec["actions"]
    gains=rec["gain"]
    losses=np.load(R/f"loss_{cond}.npz")["loss"]
    n=len(acts)
    assert acts.shape==(1024,len(NAMES)) and np.isin(acts,[1,2,3]).all()
    scores=np.take_along_axis(losses,acts-1,axis=1)
    probe=losses[:,2]
    summary={}
    for i,name in enumerate(NAMES):
        a=acts[:,i]
        sw=a!=3
        summary[name]=dict(loss=float(scores[:,i].mean()),switch=int(sw.sum()),actions=np.bincount(a,minlength=4)[1:].tolist(),
                           benefit_if_switched=ci((probe-scores[:,i])[sw]),delta_to_probe=ci(scores[:,i]-probe))
    routed=scores[:,NAMES.index("routed")]
    teacher_idx=CELLS.index(cfg["routed"]["method"])
    tg=probe-scores[:,teacher_idx]
    direct_view=gains.argmax(1)
    dg=probe-losses[np.arange(n),direct_view]
    g=gains.max(1)
    cuts=np.digitize(g,cfg["cuts"])
    strata=[]
    for j in range(4):
        m=cuts==j
        strata.append(dict(n=int(m.sum()),predicted=float(g[m].mean()),direct=ci(dg[m]),teacher=ci(tg[m]),
                           gate_switch=int((acts[m,-1]!=3).sum()),
                           minority_median=float(np.median(np.minimum(rec["features"][m,13],1-rec["features"][m,13])))))
    cube=scores[:,:8].reshape(n,2,2,2)
    effects={}
    for mask in range(1,8):
        subset=[j for j in range(3) if mask & (1<<j)]
        vals=cube
        for j in reversed(range(3)):
            vals=np.diff(vals,axis=j+1).squeeze(j+1) if j in subset else vals.mean(j+1)
        effects[" x ".join(["Y law E-C","label source E-C","mean-sample"][j] for j in subset)]=ci(vals)
    contrasts={name:ci(routed-scores[:,NAMES.index(name)],.025 if cond==0 else .05) for name in ["probe","fixed_global"]}
    # Compare individual observation choices in each one-factor intervention.
    flips={}
    for i,name in enumerate(CELLS):
        for bit in (4,2,1):
            target=i^bit
            if i<target:
                flips[name+" -> "+CELLS[target]]=int(np.sum(acts[:,i]!=acts[:,target]))
    return dict(rows=summary,contrasts=contrasts,strata=strata,effects=effects,flips=flips,
                teacher_called=int((g>threshold(cfg["routed"])).sum()),top_bottom_direct=float(dg[cuts==3].mean()-dg[cuts==0].mean()),
                hindsight_lower=float(losses.min(1).mean()),mean_loss_by_action=losses.mean(0).tolist())


def main():
    require()
    for path in ("sources","gate_freeze","choices_freeze"):
        obj=read(R/(path+".json"))
        for name,h in (obj if path=="sources" else obj["files"]).items():assert sha(name)==h,name
    cfg=read(R/"gate.json")
    summary={str(c):analyze(c,cfg) for c in (0,1)}
    write(R/"summary.json",summary)
    print("Verified frozen files; held-out results ready")
if __name__=="__main__":
    main()
