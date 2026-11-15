"""Paired factorial-shift evaluation with simultaneous confidence bands on Brier effects.

Only summary statistics and physical metadata are read on the allocated compute node.
All shift, method, and intervention comparisons share bank and world bootstrap draws.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

R = Path("results")


def verify_records():
    for n in ("sources", "plan", "choices_freeze"):
        rec = json.loads((R / (n + ".json")).read_text())
        for p, expected in rec.get("files", rec).items():
            if hashlib.sha256(Path(p).read_bytes()).hexdigest() != expected:
                raise ValueError(f"{n} digest mismatch: {p}")


def crossed_means(values, cells, n=10000):
    """B x W x K. Resample common world ids within fixed physical cells and banks independently."""
    b, w, k = values.shape
    a = np.ascontiguousarray(values.transpose(1, 0, 2).reshape(w, b*k))
    rng = np.random.default_rng(264103)
    groups = [np.flatnonzero(cells == c) for c in np.unique(cells)]
    boot = np.empty((n,k))
    for start in range(0,n,50):
        size = min(n-start,50)
        bank = rng.multinomial(b,np.full(b,1/b),size)/b
        weights = np.empty((size,w))
        for ids in groups:
            weights[:,ids] = rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size)/w
        means = (weights@a).reshape(size,b,k)
        boot[start:start+size] = np.einsum("tb,tbk->tk",bank,means)
    return boot


def brier_contrasts(risk, gate_col, baseline_col):
    """risk[...,condition,method]; retain vectors for all conditions on the last axis."""
    gate = risk[...,gate_col]
    baseline = risk[...,baseline_col]
    d = gate-baseline
    return {"gap":d, "excess_gap":d-d[...,0:1],
            "gate_change":gate-gate[...,0:1], "k256_change":baseline-baseline[...,0:1]}


def interval(row, point):
    lo,hi=np.quantile(row,[.025,.975])
    return {"mean":float(point), "ci95":[float(lo),float(hi)]}


def simultaneous_band(bootstrap, point):
    """Centered bootstrap max-|t| using bootstrap SD across all contrasts, excluding constants."""
    err = bootstrap-point
    scale = bootstrap.std(axis=0,ddof=1)
    eligible = scale>1e-12
    largest = np.max(np.abs(err[:,eligible]/scale[eligible]),axis=1) if eligible.any() else np.zeros(len(err))
    critical=float(np.quantile(largest,.95))
    return point-critical*scale,point+critical*scale,critical


def paired_time_quantiles(times, cells, n=5000):
    """B x Wtimed x repeat x settings x methods. Preserve all measured interleaved pairs."""
    b,w,reps,settings,methods=times.shape
    rng=np.random.default_rng(266103)
    groups=[np.flatnonzero(cells==k) for k in np.unique(cells)]
    quant=np.empty((n,settings,methods))
    for start in range(0,n,5):
        size=min(5,n-start)
        bank=rng.integers(b,size=(size,b,1,1))
        world=np.concatenate([rng.choice(g,(size,len(g)),replace=True) for g in groups],1)[:,None,:,None]
        repeat=rng.integers(reps,size=(size,1,1,reps))
        sample=times[bank,world,repeat].reshape(size,-1,settings,methods)
        quant[start:start+size]=np.quantile(sample,.90,axis=1)
    return quant


def subset_mean(x, mask):
    mask=np.broadcast_to(mask,x.shape)
    return float(x[mask].mean()) if mask.any() else None


def transitions(ordered, passes):
    yes=[n for n in ordered if passes[n]]
    sustained=next((i for j,i in enumerate(ordered) if all(passes[n] for n in ordered[j:])),None)
    return {"first":yes[0] if yes else None,"first_maintained":sustained,"detected":yes,
            "nonmonotone_pass_pattern":bool(yes and not all(passes[n] for n in ordered[ordered.index(yes[0]):]))}


def pathways(pairs):
    levels = [1.,.75,.5,0.,-.5,-1.]
    ids = {tuple(v):i for i,v in enumerate(pairs)}
    return {"profile":[ids[x,1.] for x in levels],
            "velocity":[ids[1.,x] for x in levels],
            "diagonal":[ids[x,x] for x in levels]}


def expanded(raw,variants):
    """B x W x vertex x trailing axes -> B x W x settings x trailing axes."""
    return raw[:, np.arange(raw.shape[1])[:,None], variants.T]


def flags_by_margin(band,margin):
    """Failure = absolute gap plus relative degradation; separate stricter own-drift test."""
    base=band["gap"]["sim95"][0] > margin
    if band["excess_gap"]["mean"] == 0. and band["excess_gap"]["sim95"] == [0.,0.]:
        # Unshifted point: report baseline gap; never call shift-induced failure at origin.
        return dict(relative=False,absolute=False,gap_only=bool(base))
    return dict(relative=bool(base and band["excess_gap"]["sim95"][0] > 0),
                absolute=bool(base and band["excess_gap"]["sim95"][0] > 0 and band["gate_change"]["sim95"][0]>margin and
                              band["k256_change"]["sim95"][1]<margin),
                gap_only=bool(base))


def contrast_tables(values, boot, gate=0, baseline=1):
    original=brier_contrasts(values,gate,baseline)
    sampled=brier_contrasts(boot,gate,baseline)
    keys=list(original)
    b=np.stack([sampled[k] for k in keys],axis=-1)
    p=np.stack([original[k] for k in keys],axis=-1)
    c,j=p.shape
    lo,hi,crit=simultaneous_band(b.reshape(len(b),-1),p.ravel())
    lo=lo.reshape(c,j);hi=hi.reshape(c,j)
    pp=np.quantile(b,[.025,.975],axis=0)
    rows=[{k:{"mean":float(p[i,h]),"ci95":pp[:,i,h].tolist(),
              "sim95":[float(lo[i,h]),float(hi[i,h])]} for h,k in enumerate(keys)} for i in range(c)]
    return rows,crit


def evalue(risk,cell,gate,baseline):
    c,j=risk.shape[2:]
    b=crossed_means(risk.reshape(*risk.shape[:2],-1),cell).reshape(-1,c,j)
    points=risk.mean((0,1))
    bands,critical=contrast_tables(points,b,gate,baseline)
    return points,b,bands,critical


def allocation(bank_world_flag,pred,inferred_gate,margin,choice_gate,choice_k):
    """Positive factual score and realized Lprobe-Lquery both favor Q."""
    y=margin.mean(0)
    predicted=pred.mean(0)
    # A common world counts once even if it appears in every numerical bank.
    centered_pred=predicted-predicted.mean();centered_y=y-y.mean()
    norm=np.linalg.norm(centered_pred)*np.linalg.norm(centered_y)
    same=bank_world_flag==(bank_world_flag[:1])
    assert same.all()
    return {"high_confidence_rate":float(inferred_gate.mean()),"direct_query_rate":float((pred>=0).mean()),
            "direct_sign_agreement":float(np.mean((predicted>=0)==(y>=0))),
            "margin_score_corr":float(centered_pred@centered_y/norm) if norm>0 else None,
            "gated_probe_rate":float((choice_gate==1).mean()),"k256_probe_rate":float((choice_k==1).mean())}


def main():
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Use an allocated compute node.")
    verify_records()
    spec=json.loads((R/"design.json").read_text())
    pairs=np.array(spec["conditions"],dtype=float)
    assert pairs.shape==(36,2) and np.array_equal(pairs[0], [1.,1.])
    methods=["gate","k256","direct","k1024","query"]
    with np.load(R/"test/private.npz") as f:
        cell=f["cell"];label=f["label"].astype(bool)
    with np.load(R/"test/public.npz") as f:x=f["x"]
    variants=np.load(R/"scenarios.npz")["variants"].astype(int)
    c,w=variants.shape
    assert c==len(pairs) and label.shape==(w,4) and x.shape==(w,4,11)
    assert np.all((variants>=0)&(variants<4)) and np.all(variants[0]==0)
    assert len(np.unique(cell))==4 and np.ptp(np.unique(cell,return_counts=True)[1])==0
    choice=[];on=[];scores=[];cost=[];tm=[];ids=None
    for b in range(5):
        with np.load(R/f"choices_{b}.npz") as f:
            a=f["acts"];assert np.isin(a,[1,3]).all()
            choice.append((a==3).astype(int));on.append(f["confident"].astype(bool));scores.append(f["score"])
        with np.load(R/f"eval_{b}.npz") as f:
            y=f["loss"]
            assert np.isfinite(y).all() and y.min()>=0 and y.max()<=1.+1e-8
            cost.append(y.mean(-1));tm.append(f["times"])
            assert ids is None or np.array_equal(ids,f["ids"])
            ids=f["ids"]
    a=np.stack(choice);on=np.stack(on);scores=np.stack(scores);cost=np.stack(cost)
    tm=np.stack(tm)
    assert a.shape==(5,w,4,len(methods)) and cost.shape==(5,w,4,2)
    assert np.array_equal(a[...,0],np.where(on,a[...,2],a[...,3]))
    assert (a[...,4]==0).all() and np.array_equal(a[...,2],(scores<0))
    assert np.all(on==on[:1]) and np.allclose(scores,scores[:1],rtol=0,atol=0)
    assert np.isfinite(scores).all() and np.isfinite(tm).all() and (tm>0).all()
    assert tm.shape==(5,c,len(ids),3,2) and len(np.unique(ids))==len(ids)
    # Add always-probe to compare both unconditional action prevalences without re-running GPU.
    methods.append("probe")
    a=np.concatenate([a,np.ones(a.shape[:-1]+(1,),dtype=int)],axis=-1)
    pick=np.take_along_axis(cost,a,axis=3)
    actual=expanded(pick,variants)
    all_a=expanded(a,variants);all_on=expanded(on,variants);all_scores=expanded(scores,variants)
    all_cost=expanded(cost,variants)
    origin_cost=cost[:,:,0,:]
    orig_fb=a[:,:,0,3,None]
    replay_choice=np.where(all_on,all_a[...,2],orig_fb)
    replay_cost=np.where(replay_choice,origin_cost[:,:,1,None],origin_cost[:,:,0,None])
    # Replay blocks all physical changes in query truth, added measurements, posterior, and K1024 fallback.
    all_risk=np.concatenate([actual,replay_cost[...,None]],axis=-1)
    points,b,rows,critical=evalue(all_risk,cell,0,1)
    replay_did=b[:,:,-1]-b[:,0,-1,None]
    replay_gap=b[:,:,-1]-b[:,0,1,None]
    re_dp=points[:,-1]-points[0,-1]
    re_gp=points[:,-1]-points[0,1]
    replay_ci=np.quantile(replay_gap,[.025,.975],axis=0)
    rd_ci=np.quantile(replay_did,[.025,.975],axis=0)
    raw_times=tm.transpose(0,2,3,1,4)
    timing_draws=paired_time_quantiles(raw_times,cell[ids],3000)
    quant=np.quantile(tm.transpose(1,4,0,2,3).reshape(c,2,-1),[.5,.9,.95],axis=2)
    rt=quant[1,:,0]/quant[1,:,1]
    rt_ci=np.quantile(timing_draws[:,:,0]/timing_draws[:,:,1],[.025,.975],axis=0)
    t_ci=np.quantile(timing_draws,[.025,.975],axis=0)
    index=np.arange(w)
    curve=(cell//2).astype(bool);friction=(cell%2).astype(bool)
    gap=all_risk[...,0]-all_risk[...,1]
    result=[]
    for h in range(c):
        row=rows[h];a_h=all_a[:,:,h];on_h=all_on[:,:,h];risk_h=actual[:,:,h]
        variants_here=variants[h]
        same=a_h[...,0]==a_h[...,1]
        sign=np.where(same,0,gap[:,:,h])
        assert np.allclose(sign,gap[:,:,h],atol=1e-12)
        assert np.isclose(gap[:,:,h].mean(),row["gap"]["mean"],atol=1e-12)
        profile=x[index,variants_here,:9].mean(1)
        vx=x[index,variants_here,9]
        def stats_mask(mask):
            mask=np.broadcast_to(mask,same.shape)
            nh=int(mask.sum())
            if not nh:return {"n":0}
            return {"n":nh,"unique_worlds":int(mask.any(0).sum()),
                    "gate_minus_k256":float(gap[:,:,h][mask].mean()),
                    "direct_wrong_fraction":float(np.mean((all_a[:,:,h,2][mask]==1)!=
                        ((all_cost[:,:,h,1]-all_cost[:,:,h,0])[mask]<0))),
                    "high_confidence_rate":float(on_h[mask].mean()),
                    "gate_probe":float(a_h[...,0][mask].mean()),
                    "k256_probe":float(a_h[...,1][mask].mean()),
                    "loss":risk_h[mask].mean(0).tolist()}
        facts=allocation(on_h,all_scores[:,:,h],on_h,all_cost[:,:,h,1]-all_cost[:,:,h,0],a_h[...,0],a_h[...,1])
        facts.update(profile_wide_minus_narrow=float(profile[curve].mean()-profile[~curve].mean()),
                     vx_wide_minus_narrow=float(vx[friction].mean()-vx[~friction].mean()),
                     corr_profile_curvature_spread=float(np.corrcoef(profile,curve)[0,1]),
                     corr_vx_friction_spread=float(np.corrcoef(vx,friction)[0,1]),
                     label_change=float(np.mean(label[index,variants_here]!=label[:,0])),
                     factual_profile_sd=float(profile.std()),factual_velocity_sd=float(vx.std()))
        result.append(dict(idx=h,alpha_c=float(pairs[h,0]),alpha_f=float(pairs[h,1]),loss=dict(zip(methods,points[h,:-1].tolist())),
            bands=row,flags={str(m):flags_by_margin(row,m) for m in (0.,.001,.003,.005)},
            timing={name:dict(p50_ms=float(quant[0,h,k]),p90_ms=float(quant[1,h,k]),
                        p95_ms=float(quant[2,h,k]),p90_ci95=t_ci[:,h,k].tolist()) for k,name in enumerate(methods[:2])},
            p90_ratio=float(rt[h]),p90_ratio_ci95=rt_ci[:,h].tolist(),facts=facts,
            changed_choice_bankworlds=int((~same).sum()),changed_choice_worlds=int((~same).any(0).sum()),
            gap_on_difference=subset_mean(gap[:,:,h],~same),bank_gap=gap[:,:,h].mean(1).tolist(),
            cells={str(k):stats_mask(cell==k) for k in range(4)},direct_branch=stats_mask(on_h),
            fallback_branch=stats_mask(~on_h),
            pure_cue=dict(gate_loss=float(points[h,-1]),delta_vs_origin_k256=float(re_gp[h]),
                          delta_ci95=replay_ci[:,h].tolist(),change=float(re_dp[h]),change_ci95=rd_ci[:,h].tolist(),
                          choice_flip_rate_vs_origin=float(np.mean(replay_choice[:,:,h]!=all_a[:,:,0,0])))))
    paths=pathways(pairs)
    crossing={}
    for name,path in paths.items():
        out={}
        for m in (0.,.001,.003,.005):
            bykind={}
            for measure in ("relative","absolute","gap_only"):
                passes={i:result[i]["flags"][str(m)][measure] for i in path}
                bykind[measure]=transitions(path,passes)
            out[str(m)]=bykind
        crossing[name]=dict(indices=path,thresholds=out,
            point_gap_reversals=[i for before,i in zip(path,path[1:]) if rows[before]["gap"]["mean"]>rows[i]["gap"]["mean"]])
    g=b[...,0]-b[...,1]
    p=points[:,0]-points[:,1]
    lookup={tuple(x):i for i,x in enumerate(pairs)}
    interaction_samples=[]
    interaction_means=[]
    axis_samples=[]
    axis_means=[]
    for h,(ac,af) in enumerate(pairs):
        i,j=lookup[(ac,1.)],lookup[(1.,af)]
        bs=g[:,h]-g[:,i]-g[:,j]+g[:,0]
        ip=p[h]-p[i]-p[j]+p[0]
        interaction_samples.append(bs)
        interaction_means.append(ip)
    for i,j in zip(paths["profile"],paths["velocity"]):
        axis_samples.append(g[:,i]-g[:,j])
        axis_means.append(p[i]-p[j])

    def add_bands(boot_cols,point_cols):
        boot=np.column_stack(boot_cols);point=np.array(point_cols)
        lows,highs,_=simultaneous_band(boot,point)
        return [dict(**interval(bs,m),sim95=[float(lo),float(hi)])
                for bs,m,lo,hi in zip(boot_cols,point,lows,highs)]
    interactions=add_bands(interaction_samples,interaction_means)
    axis=add_bands(axis_samples,axis_means)
    info={"methods":methods, "origin_index":0, "primary_margin":.003,
          "familywise_critical":critical,"primary_band_columns":4*c,"bootstrap_draws":10000,
          "timing_bootstrap_draws":len(timing_draws),"rows":result,"paths":crossing,
          "axis_gap_profile_minus_velocity_exploratory":axis,"factorial_interaction_exploratory":interactions,
          "bank_count":len(a),"worlds":w,"worlds_per_cell":int(w/4),"timed_worlds":len(ids)}
    temp=R/"summary.tmp";temp.write_text(json.dumps(info,indent=2,allow_nan=False));temp.replace(R/"summary.json")
    print("Simultaneous critical",critical,"first relative failures",
            {n:c["thresholds"]["0.003"]["relative"]["first"] for n,c in crossing.items()},flush=True)

if __name__=="__main__":main()
