"""Validate frozen source-only signed forecasts and design-informed choice on 12 paired panels."""
from pathlib import Path
import itertools
import json
import numpy as np
from scipy.stats import spearmanr,t

ROOT=Path(__file__).resolve().parent

def describe(x,comparisons=1):
    x=np.asarray(x)
    m=x.mean()
    r=t.ppf(1-.025/comparisons,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return dict(mean=float(m),ci=[float(m-r),float(m+r)])

def main():
    frozen=json.loads((ROOT/'forecast40.json').read_text())['cells']
    data=json.loads((ROOT/'stats40.json').read_text())
    truth=data['cells']
    assert list(truth)==list(frozen)
    risk=np.array(data['panel_risk'])
    delta=risk[:,:,1]-risk[:,:,0]
    n,c=delta.shape
    ix=np.arange(c)
    observed=delta.mean(0)
    methods=('delta','law_replay_no_contour','tangent_only','pooled_prior')
    rng=np.random.default_rng(404041)
    boot=delta[rng.integers(n,size=(30000,n))].mean(1)
    out={}
    full=None
    for method in methods:
        q=np.array([v[method] for v in frozen.values()])
        bias=q-observed
        err=np.abs(boot-q).mean(1)
        if full is None:full=err
        take=(q<0).astype(int)
        chosen=risk[:,ix,take].mean(1)
        correct=np.sign(q)==np.sign(observed)
        confirmed=np.zeros(c,dtype=int)
        lo,hi=np.array([v['ci'] for v in truth.values()]).T
        resolved=(lo>0)|(hi<0)
        confirmed[resolved&correct]=1;confirmed[resolved&~correct]=-1
        out[method]=dict(mae=float(abs(bias).mean()),spearman=float(spearmanr(q,observed).statistic) if len(set(q))>1 else None,
              full_minus_this_mae_bootstrap95=np.quantile(full-err,[.025,.975]).tolist(),
              signs=int(correct.sum()),confirmed=int((confirmed==1).sum()),refuted=int((confirmed==-1).sum()),
              unresolved=int((confirmed==0).sum()),optimizer_cells=int(take.sum()),routed_brier=float(chosen.mean()),
              router_minus_fixed=describe(chosen-risk[:,:,0].mean(1),2),
              router_minus_optimizer=describe(chosen-risk[:,:,1].mean(1),2),
              correct_among_predicted_wins=int((correct&(q<0)).sum()),correct_among_predicted_losses=int((correct&(q>0)).sum()))
    arr=np.array([v['delta'] for v in frozen.values()]).reshape(2,3,2)
    obs=observed.reshape(2,3,2)
    oc=(arr[:,1:,:]-arr[:,:1,:]).ravel()
    tc=(obs[:,1:,:]-obs[:,:1,:]).ravel()
    lawq=np.diff(arr,axis=2).ravel();lawo=np.diff(obs,axis=2).ravel()
    save=dict(methods=out,contour_shift_forecast_correct=int(((oc*tc)>0).sum()),
              contour_shift_mae=float(np.abs(oc-tc).mean()),law_shift_forecast_correct=int(((lawq*lawo)>0).sum()),
              law_shift_mae=float(np.abs(lawq-lawo).mean()))
    (ROOT/'forecast_validation40.json').write_text(json.dumps(save,indent=2)+'\n')

if __name__=='__main__': main()
