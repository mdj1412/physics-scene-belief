"""Evaluate locked forecasts against jointly paired catalog panels."""
import json
import numpy as np
from scipy.stats import spearmanr
from stats40 import ROOT,CELLS,SAMPLES,interval

def main():
    score=json.loads((ROOT/'stats40.json').read_text())
    pre=json.loads((ROOT/'forecast40.json').read_text())['cells']
    risk=np.array(score['panel_risk'])
    estimates=risk[:,:,1]-risk[:,:,0]
    actual=estimates.mean(0)
    boot=estimates[SAMPLES].mean(1)
    metrics={}
    primary=np.array([pre[c]['delta'] for c in CELLS])
    primary_error=np.abs(boot-primary).mean(1)
    for field in ('delta','pooled_prior','tangent_only','law_replay_no_contour'):
        point=np.array([pre[c][field] for c in CELLS])
        errs=primary_error-np.abs(boot-point).mean(1)
        metrics[field]=dict(mae=np.abs(actual-point).mean(),mae_full_minus_this_ci=np.quantile(errs,[.025,.975]),
                     rho=float(spearmanr(point,actual).statistic) if np.ptp(point)>0 else None,
                     point_correct=int(np.sum(np.sign(point)==np.sign(actual))))
    verdicts={}
    for j,cell in enumerate(CELLS):
        lo,hi=score['cells'][cell]['ci']
        confidence=np.sign(lo) if np.sign(lo)==np.sign(hi) else 0
        verdicts[cell]='unresolved' if not confidence else 'confirmed' if confidence==np.sign(primary[j]) else 'refuted'
    picked=np.where(primary<0,risk[:,:,1],risk[:,:,0]).mean(1)
    original=risk.mean(1)
    out=dict(metrics=metrics,verdicts=verdicts,selected_loss=picked.mean(),
             chosen_vs_fixed=interval(picked-original[:,0]),chosen_vs_optimizer=interval(picked-original[:,1]),
             mean_forecast=primary.mean(),actual_mean=actual.mean(),
             equal_cell_win_rate=(estimates<0).mean(0))
    (ROOT/'forecast_eval40.json').write_text(json.dumps(out,indent=2,default=lambda x:x.tolist(),allow_nan=False)+'\n')

if __name__=='__main__':
    main()
