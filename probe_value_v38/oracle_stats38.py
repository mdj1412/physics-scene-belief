"""Paired uncertainty for the finite-population Brier benchmark and true-label check."""
import json
from pathlib import Path

import numpy as np
from stats38 import CELLS, METHODS, REGIMES, ROOT, samples, draw, limits


def values(x, idx):
    return {"mean": x.mean((0,1)), "ci": limits(draw(x,idx))}


def summarize(term, idx):
    d=term["gain"]
    actual=term["risk"][:,:,0]-term["risk"][:,:,1]
    rb=term["rb_gain"]
    oracle=term["oracle_risk"]
    identity=oracle[:,:,0]-oracle[:,:,1]-d
    dboot=draw(d,idx)
    stuff={
        "ideal":values(d,idx),
        "oracle_brier":values(oracle,idx),
        "oracle_realized_gain":values(oracle[:,:,0]-oracle[:,:,1],idx),
        "oracle_identity_residual":values(identity,idx),
        "policy_realized_gain":values(actual,idx),
        "policy_rb_gain":values(rb,idx),
        "realized_minus_rb_gain":values(actual-rb,idx),
        "rb_squared_deviation_from_p4":values(term["rb_excess"],idx),
        "oracle_mean_forecast_minus_incidence":values(term["calibration_residual"],idx),
        "conditional_ideal_Q_P":values(term["conditional_gain"],idx),
        "prototype_effective_support_Q_P":values(term["ess"],idx),
    }
    for k,g in (("realized",actual),("rb",rb)):
        if d.mean() <= 0 or np.min(dboot) <=0:
            stuff[k+"_fraction"] = None
        else:
            stuff[k+"_fraction"] = dict(mean=g.mean((0,1))/d.mean(),ci=limits(draw(g,idx)/dboot[:,None]))
    return stuff


def marginal_summary():
    f=ROOT/"marginal38.json"
    if not f.exists():
        return None
    obj=json.loads(f.read_text())
    cells={}
    allrows=[]
    for i,name in enumerate(obj["cells"]):
        a=obj["cases"][name]
        assert len(a)==48
        j=lambda k:np.asarray([r[k] for r in a])
        known=j("known_gain").mean(); hidden=j("hidden_gain").mean()
        signed_prob=j("family_paid_p")[:,:,i]
        cells[name]=dict(known_gain=known,hidden_gain=hidden,hidden_minus_known=hidden-known,
                         true_family_mean_p_from_factual=j("family_factual_p")[:,i].mean(),
                         true_family_min_p_from_factual=j("family_factual_p")[:,i].min(),
                         true_family_mean_p_from_paid=signed_prob.mean(0),
                         known_brier=j("known_risk").mean(0),hidden_brier=j("hidden_risk").mean(0),
                         mean_abs_forecast_difference=j("abs_probability_difference").mean(0),
                         known_hidden_rb_policy_gain=j("rb_policy_gain").mean(0))
        allrows.extend(a)
    return dict(cells=cells,mean_known_gain=float(np.mean([r["known_gain"] for r in allrows])),
                mean_hidden_gain=float(np.mean([r["hidden_gain"] for r in allrows])),
                mean_hidden_minus_known=float(np.mean([r["hidden_gain"]-r["known_gain"] for r in allrows])),
                mean_absolute_probability_difference=np.mean([r["abs_probability_difference"] for r in allrows],0),
                mean_known_hidden_rb_policy_gain=np.mean([r["rb_policy_gain"] for r in allrows],0),
                unique_environment_base_draws=32, true_scenarios=192, noise_repeats_per_object=1)


def main():
    arrays={}
    for c,cell in enumerate(CELLS):
        for b in range(8):
            with np.load(ROOT/"oracle_results"/cell/f"block_{b:03d}"/"oracle_summary.npz") as f:
                np.testing.assert_array_equal(f["methods"],METHODS)
                np.testing.assert_array_equal(f["regimes"],REGIMES)
                np.testing.assert_array_equal(f["signs"],[1,-1])
                for k in ("gain","oracle_risk","risk","rb_gain","rb_excess","ess",
                          "conditional_gain","calibration_residual"):
                    data=f[k]
                    assert data.shape[:3]==(2,3,8) and np.all(np.isfinite(data))
                    if k not in arrays:
                        arrays[k]=np.empty((4,8)+data.shape)
                    arrays[k][c,b]=data
    means={k:v.mean((2,3,4)) for k,v in arrays.items()}
    idx=samples()
    report=dict(methods=METHODS,primary=summarize(means,idx),
                cell_ideal_gain=means["gain"].mean(1),
                cell_realized_gain=(means["risk"][:,:,0]-means["risk"][:,:,1]).mean(1),
                cell_rb_gain=means["rb_gain"].mean(1),
                descriptive_regime_ideal={r:arrays["gain"][:,:,:,i].mean() for i,r in enumerate(REGIMES)},
                law_ideal={law:means["gain"][ii].mean() for law,ii in
                           (("weakening",[0,2]),("strengthening",[1,3]))},
                hidden_cell=marginal_summary())
    out=ROOT/"oracle_stats38.json"
    out.write_text(json.dumps(report,indent=2,default=lambda v:v.tolist(),allow_nan=False)+"\n")
    print(out)

if __name__=="__main__":
    main()
