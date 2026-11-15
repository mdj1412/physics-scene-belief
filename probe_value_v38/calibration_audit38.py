"""Secondary finite-catalog prevalence and small-sample residual diagnostics."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student
import physics38 as world
from stats38 import CELLS, REGIMES, ROOT


def studentized(x):
    """Welch CI for an equally weighted average of eight design-stratum means."""
    strata=x.reshape(4,4,2,-1).transpose(0,2,1,3).reshape(8,4,-1)
    v=strata.var(1,ddof=1)/(4*8**2)
    se=v.sum(0)**.5
    mu=strata.mean((0,1))
    df=v.sum(0)**2/(v**2/3).sum(0)
    crit=student.ppf(.975,df)
    return dict(mean=mu,se=se,df=df,p_two_sided=2*student.sf(np.abs(mu/se),df),
                lo=mu-crit*se,hi=mu+crit*se)


def main():
    prevalence=np.zeros(4)
    predicted_minus_outcome=np.zeros((4,8,2))
    observed=np.zeros((4,8))
    for c,cell in enumerate(CELLS):
        shape,surface=cell.split("_")
        for sign in (1,-1):
            law=(shape,surface,sign)
            for r in REGIMES:
                for alpha in world.ALPHAS:
                    theta=world.latent_support(alpha,law,20)
                    y=world.impact(theta,world.QUERY_Y,world.QUERY_INCOMING,law,r)[1]
                    prevalence[c]+=y.double().mean().item()/(2*3*8)
        for b in range(8):
            block=f"block_{b:03d}"
            with np.load(ROOT/"oracle_results"/cell/block/"oracle_summary.npz") as f:
                predicted_minus_outcome[c,b]=f["calibration_residual"].mean((0,1,2))
            for s in ("plus","minus"):
                for r in REGIMES:
                    with np.load(ROOT/"results"/cell/block/f"{r}_{s}.npz") as f:
                        observed[c,b]+=f["labels"].mean()/6
    actual=observed-prevalence[:,None]
    result=dict(cells=CELLS,prevalence_exact=prevalence,prior_mean=prevalence.mean(),
                observed_mean=observed.mean(),observed_minus_prior_per_cell=actual.mean(1),
                observed_minus_prior=studentized(actual[:,:,None]),
                forecast_minus_prior=studentized(predicted_minus_outcome+actual[:,:,None]),
                forecast_minus_outcome=studentized(predicted_minus_outcome),
                forecast4_minus_forecast3=studentized(np.diff(predicted_minus_outcome,axis=2)),
                per_cell_forecast_minus_outcome=predicted_minus_outcome.mean(1),
                procedures="Unadjusted secondary Welch-Satterthwaite t sensitivity, 8 cell-by-start strata, 4 independent blocks per stratum. Residuals correlated and not independent tests.")
    (ROOT/"calibration_audit38.json").write_text(json.dumps(result,indent=2,default=lambda x:x.tolist())+"\n")


if __name__=="__main__":
    main()
