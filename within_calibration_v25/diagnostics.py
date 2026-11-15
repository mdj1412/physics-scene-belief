"""Exploratory dependence on generator cues; fit without any test costs or labels."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from gate25 import Predictor, features
from physics25 import CALIBRATION
from stats25 import crossed_means, column_stats, file_checks

R = Path("results")
MASKS = {"all": list(range(6)), "mean_vx": [0,3], "delete_mean_vx": [1,2,4,5]}


def get(split):
    with np.load(R/split/"public.npz") as f:
        return f["x"][:,0,:]


def restricted_fit(f, y, columns, family, penalty):
    # Zero unselected feature and its interactions, including in RBF center distances.
    f = f.copy()
    missing = np.setdiff1d(np.arange(f.shape[1]), columns)
    f[:,missing] = 0.
    mean = f.mean(0)
    scale = np.maximum(f.std(0), 1e-8)
    rng = np.random.default_rng(2510801)
    normalized = (f-mean)/scale
    centers = normalized[rng.choice(len(f), 96, replace=False)] if family == "rbf" else np.empty((0,6))
    model = dict(family=family, mean=mean.tolist(), std=scale.tolist(),
                 centers=centers.tolist(), width=1.8, beta=[], bias=0.)
    matrix = Predictor(model).design(f)
    ctr = matrix.mean(0)
    matrix -= ctr
    beta = np.linalg.solve(matrix.T@matrix + penalty*len(y)*np.eye(matrix.shape[1]),
                           matrix.T@(y-y.mean()))
    model.update(beta=beta.tolist(), bias=float(y.mean()-ctr@beta))
    return model


def value(model, f, columns):
    f = f.copy()
    f[:,np.setdiff1d(np.arange(f.shape[1]), columns)] = 0.
    return Predictor(model).design(f)@np.asarray(model["beta"]) + model["bias"]


def train_masked(train, val, y, val_cost, fall, columns):
    candidates = []
    for family in ("quadratic","rbf"):
        for penalty in (.0001,.001,.01):
            model = restricted_fit(train,y,columns,family,penalty)
            tr, va = (value(model,f,columns) for f in (train,val))
            for q in (.2,.4,.6,.8):
                cut = float(np.quantile(np.abs(tr),q))
                confident = np.abs(va)>=cut
                choice = np.where(confident,np.where(va>=0,0,2),fall)
                loss = float(val_cost[np.arange(len(val)),choice].mean())
                key = (loss,-confident.mean(),family!="quadratic",penalty,q)
                cfg = dict(family=family,penalty=penalty,quantile=q,cut=cut,
                           validation_loss=loss,confident=float(confident.mean()))
                candidates.append((key,cfg,model))
    _,cfg,model = min(candidates,key=lambda z:z[0])
    return dict(**cfg,model=model)


def fixed_width_score(f):
    # Do not fit using any test statistics. Proxy centers come from the simulator spec.
    # Conditional spread is approximated with uniform-to-Gaussian variance matching.
    offset_sd = np.hypot(.0034/np.sqrt(12), CALIBRATION[0]/3)
    vx_sd = np.hypot(.18/np.sqrt(12), CALIBRATION[1])
    log_curve_width_odds = 2*.00115*f[:,0]/offset_sd**2
    log_friction_width_odds = -2*.095*(f[:,3]+.70)/vx_sd**2
    return log_curve_width_odds - log_friction_width_odds


def exposure(scores, actual):
    # All boundaries are descriptive score quantiles for this split, not learned cutoffs.
    bins = np.array_split(np.argsort(scores, kind="stable"), 5)
    ybar = actual.mean(0)
    sd = scores.std()
    bias = np.mean(scores-ybar)
    return {
        "n_worlds": len(scores), "predicted_probe_fraction": float((scores < 0).mean()),
        "mean_score":float(scores.mean()), "mean_actual_probe_minus_query":float(ybar.mean()),
        "mean_error":float(bias),
        "sign_agreement":float(np.mean((scores<0)==(ybar<0))),
        "spearman_unique_worlds":float(spearmanr(scores,ybar).statistic) if sd>0 and ybar.std()>0 else None,
        "rmse_to_realized_margin":float(np.sqrt(np.mean((scores-ybar)**2))),
        "quintiles":[{"n":len(ix), "score_mean":float(scores[ix].mean()),
            "realized_probe_minus_query_mean":float(ybar[ix].mean()),
            "realized_probe_win_fraction":float((ybar[ix]<0).mean())} for ix in bins],
    }


def conditional_errors(route, score, margin1, margin2):
    # This consistency subset is evaluator-only and depends on realized labels.
    q=(margin1>.02)&(margin2>.02);p=(margin1<-.02)&(margin2<-.02)
    out={}
    for name, mask in (("robust_query",q),("robust_probe",p),("ambiguous",~(q|p))):
        here=mask&route
        y=(margin1+margin2)/2
        wrong=np.where(score>=0,-y,y)
        out[name]=dict(worlds=int(mask.sum()),direct_worlds=int(here.sum()),
                       direct_errors=int(((wrong>0)&here).sum()),
                       error_rate=float((wrong[here]>0).mean()) if here.any() else None,
                       mean_regret=float(np.maximum(wrong[here],0).mean()) if here.any() else None)
    return out


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated compute node for paired diagnostics.")
    file_checks(R)
    train,val=(features(get(split)) for split in ("train","validation"))
    with np.load(R/"train/development.npz") as d:
        train_margin = d["cost"][:,:,2]-d["cost"][:,:,0]
    with np.load(R/"validation/development.npz") as d:
        val_cost=d["cost"][0];fall=d["fallback"]-1
        val_margin=(val_cost[:,2]-val_cost[:,0])[None,:]
    cfgs={"all":json.loads((R/"selected.json").read_text())}
    for n in ("mean_vx","delete_mean_vx"):
        cfgs[n]=train_masked(train,val,train_margin.mean(0),val_cost,fall,MASKS[n])

    test=features(get("test"))
    names=json.loads((R/"methods.json").read_text())["names"]
    actions=[];raw=[]
    for i in range(5):
        with np.load(R/f"choices_{i}.npz") as d:actions.append(d["acts"]-1)
        with np.load(R/f"eval_{i}.npz") as d:raw.append(d["loss"])
    a=np.stack(actions);raw=np.stack(raw);cost=raw.mean(-1)
    with np.load(R/"test/private.npz") as d:cell=d["cell"]
    margin=cost[:,:,2]-cost[:,:,0]
    h1=raw[:,:,:,:16].mean(-1);h2=raw[:,:,:,16:].mean(-1)
    m1=h1[:,:,2]-h1[:,:,0];m2=h2[:,:,2]-h2[:,:,0]
    oracle=.5*(np.where(m1>=0,h2[:,:,0],h2[:,:,2])+np.where(m2>=0,h1[:,:,0],h1[:,:,2]))
    action_by_method={};cal_rows={};robust_rows={};models={}
    for n,cfg in cfgs.items():
        cols=MASKS[n]
        tr,va,te=(value(cfg["model"],x,cols) for x in (train,val,test))
        cutoff=cfg["cut"];fire=np.abs(te)>=cutoff
        actions_n=np.where(fire,np.where(te>=0,0,2),a[:,:,names.index("k1024")])
        if n=="all":
            assert np.array_equal(actions_n, a[:,:,names.index("gate")])
        action_by_method[n]=actions_n
        cal_rows[n]={lbl:exposure(sc,act) for lbl,sc,act in
                     [("train",tr,train_margin),("validation",va,val_margin),("test",te,margin)]}
        robust_rows[n]=conditional_errors(fire,te,m1.mean(0),m2.mean(0))
        models[n]={k:cfg[k] for k in ("family","penalty","quantile","cut","validation_loss")}
        models[n]["features"]=cols;models[n]["test_direct_fraction"]=float(fire.mean())
    untrained=fixed_width_score(test)
    action_by_method["prior_width_only"]=np.broadcast_to(np.where(untrained>=0,0,2),margin.shape)
    for n in ("k256","query","direct","k1024"):
        action_by_method[n]=a[:,:,names.index(n)]
    methods=list(action_by_method)
    loss=np.stack([np.take_along_axis(cost,act[:,:,None],axis=2)[:,:,0]
                   for act in action_by_method.values()],axis=-1)
    values=np.concatenate([loss,oracle[:,:,None]],axis=-1);ms=methods+["qp_oracle"]
    avg=values.mean((0,1));boot=crossed_means(values,cell)
    def diff(x,y):
        i,j=(ms.index(n) for n in (x,y))
        return column_stats(boot[:,i]-boot[:,j],avg[i]-avg[j])
    effects={n:{ref:diff(n,ref) for ref in ("all","k256","query","qp_oracle")} for n in methods}
    risk_by_cell={str(int(c)):{n:float(loss[:,cell==c,j].mean()) for j,n in enumerate(methods)}
                  for c in np.unique(cell)}
    output=dict(
        models=models, split_calibration=cal_rows, robust_decisions=robust_rows,
        mean_loss=dict(zip(ms,avg.tolist())), paired_effects=effects, cell_mean_loss=risk_by_cell,
        action_counts={n:np.bincount(act.ravel(),minlength=3).tolist() for n,act in action_by_method.items()},
        flip_vs_full={n:int((act!=action_by_method["all"]).any(0).sum()) for n,act in action_by_method.items()},
        prior_width_proxy_rho=float(spearmanr(untrained,margin.mean(0)).statistic),
        prior_width_proxy_query_frac=float((untrained>=0).mean()),
    )
    (R/"diagnostics.json").write_text(json.dumps(output,indent=2,allow_nan=False))
    print("Exploratory cue diagnostics:", {n:round(float(x),6) for n,x in zip(ms,avg)},flush=True)

if __name__=="__main__":main()
