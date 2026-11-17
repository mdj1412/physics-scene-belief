"""No-label diagnostic using archived single paid readouts. Matched twin subtraction is diagnostic only."""
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, t

ROOT=Path(__file__).resolve().parent


def auc(low, high):
    """Probability that a random high exemplar outranks a random low exemplar, with ties split."""
    n, m = len(low), len(high)
    ranks = rankdata(np.concatenate((low,high)))
    return (ranks[n:].sum()-m*(m+1)/2)/(m*n)


def paired_win(a,b):
    return ((b>a).mean()+.5*(b==a).mean())


def ci(x):
    m=np.mean(x,axis=0)
    hw=t.ppf(.975,len(x)-1)*np.std(x,axis=0,ddof=1)/np.sqrt(len(x))
    return np.stack((m,m-hw,m+hw),axis=-1).tolist()


def main():
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Use an allocated CPU worker")
    model=json.loads((ROOT/"policy47.json").read_text())
    R,U=np.array(model["residual_operator"]),np.array(model["effective_axis"])
    grid,legacy,cv,ca,cr,risk=(np.asarray(model[k]) for k in ("x_grid","legacy_risk","variance_cuts",
                                                               "alignment_cuts","rms_cuts_mm","risk"))
    raw_axes=json.loads((ROOT.parent/"image_pressure_v45"/"source_axis45.json").read_text())
    metrics=[];delta_replay=0; legacy_err=0; nf=0; alignment_error=0.; masked=0; observed=0
    for p in range(12):
        records=np.zeros((4,2,2,8)); delta_model=np.zeros((4,2,2))
        with np.load(ROOT/"forecast"/f"panel_{p:02d}.npz") as f:
            paid=f["query_paid"].astype(float); new=f["gate_new"]; old=f["gate_v46"]
            xs,vs,selected=f["x_mm"],f["var_rad2"],f["selected"]
            assert tuple(f["alignments"])==("low","high")
            assert tuple(f["variants"])==("hollow","bulge")
            for fi,h in np.ndindex(4,2):
                axis=np.asarray(raw_axes[str((19,31)[h])]["axis"])
                # Subtract the two paired variant readouts; this is never fed to a policy.
                twin=(paid[fi,:,h,1]-paid[fi,:,h,0])*500.
                idx=abs(xs[fi,h,...,None]-grid).argmin(-1)
                q=(vs[fi,h,...,None]>cv[idx]).sum(-1)
                old_replay=legacy[idx,q]<0
                legacy_err+=int((old_replay!=old[fi,h]).sum())
                for ai in (0,1):
                    ma=np.isfinite(twin[ai]).all(-1)
                    target=(.1,.9)[ai]
                    arr=twin[ai][ma]
                    c0=(arr@axis)**2 / np.square(arr).sum(-1)
                    alignment_error=max(alignment_error,float(abs(c0-target).max()))
                    rr=arr@R
                    delta_model[fi,h,ai]=np.mean((rr@U[h])**2/np.square(rr).sum(-1))
                for v in (0,1):
                    raw=paid[fi,:,h,v]
                    mask=np.isfinite(raw).all(-1)
                    assert np.array_equal(mask[0],mask[1])
                    assert np.array_equal(mask[0], np.broadcast_to((selected[fi,h,v]==f["query_id"])[...,None],mask[0].shape))
                    masked+=int((~mask).sum());observed+=int(mask.sum())
                    rr=np.nan_to_num(raw)@R*1000.
                    en=np.square(rr).sum(-1)
                    cos=np.square(rr@U[h])/np.maximum(1e-20,en);rms=np.sqrt(en/7)
                    cc=np.searchsorted(ca,cos); aa=np.searchsorted(cr,rms)
                    pred=mask[:,None]&(risk[idx[v][None,...,None],old_replay[v].astype(int)[None,...,None],
                                            cc[:,None],aa[:,None]]<0)
                    delta_replay+=int((new[fi,:,h,v]!=pred).sum());nf+=pred.size
                    cl,ch=(cos[i][mask[i]] for i in (0,1))
                    rl,rh=(rms[i][mask[i]] for i in (0,1))
                    records[fi,h,v]=auc(cl,ch),paired_win(cl,ch),auc(rl,rh),paired_win(rl,rh),cl.mean(),ch.mean(),rl.mean(),rh.mean()
        if p==0:
            true_proj=delta_model
        else:
            np.testing.assert_allclose(true_proj,delta_model,atol=1e-6)
        metrics.append(records)
    values=np.array(metrics)
    with np.load(ROOT/"forecast"/"panel_00.npz") as f:
        names=f["families"].tolist()
    out=dict(fields=["stratified_auc_cos","paired_high_gt_low_cos","stratified_auc_rms","paired_high_gt_low_rms",
                     "mean_cos_low","mean_cos_high","mean_rms_low_mm","mean_rms_high_mm"],
             scope="Only paid query arm. AUC within each catalog/family/height/variant; equal weighted. Intervals descriptive catalog t.",
             overall=ci(values.mean((1,2,3))),family_height_variant=ci(values),family_names=names,
             height_mm=[19,31],variant=["hollow","bulge"],coupling_input="One record shared by null/strong twins",
             count_paid=observed,count_masked=masked,legacy_errors=legacy_err,new_gate_replay_errors=delta_replay,
             new_gate_replay_decisions=nf,raw_twin_axis_max_abs_err=alignment_error,
             residualized_true_alignment_family_height_low_high=true_proj.tolist())
    (ROOT/"proxy_audit47.json").write_text(json.dumps(out,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":
    main()
