"""Independent CPU risk audit, disjoint four-pair prefixes and random replacement law."""
import os
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parent
V50=ROOT.parent/'polarity_calibration_v50'
sys.path.insert(0,str(V50))
from score50 import PRED,components,integrate,blend,interval
from estimator50 import infer,DEAD_ZONE
from world51 import paired_calibration_rows,EPSILON,DRIFT,KAPPA


def run(i):
    name=f'panel_{i:02d}.npz'
    with np.load(V50/'forecast50'/name) as f,np.load(V50/'labels50'/name) as l:
        pred=f['p'][[list(f['methods'].astype(str)).index(m) for m in PRED]]
        kernel,gap=components(pred,l['labels'])
    with np.load(V50/'worlds'/name) as w:
        partner=paired_calibration_rows(w['surface'],w['law'])
    with np.load(V50/'cal50'/name) as c:
        xs=c['weak_mm'][:,partner]; ys=c['full_mm'][:,partner]
    retention=np.asarray(KAPPA)
    # [prefix group, drift model, epsilon, drift, method]
    out=np.zeros((2,2,6,6,3))
    for prefix in (0,1):
        x=xs[...,4*prefix:4*(prefix+1)];y=ys[...,4*prefix:4*(prefix+1)]
        good=(abs(x)>=DEAD_ZONE[0])&(abs(y)>=DEAD_ZONE[1])
        hats=(infer(x,y),(1+(good&((x>0)==(y>0))).sum(-1))/(2+good.sum(-1)))
        for ie,e in enumerate(EPSILON):
            def conditional(k):
                risk,d2=integrate(kernel,gap,0.,e+(1-2*e)*k)
                return risk[[1,2,5]],d2[0]
            banks=np.stack([conditional(k)[0] for k in retention])
            d2=conditional(.5)[1]
            for kind,alternative in enumerate((banks[::-1],banks.mean(0)[None])):
                # On drift, complement the old k or resample a new k independent of the old k.
                for j,g in enumerate(DRIFT):
                    perms=(1-g)*banks+g*alternative
                    lp,lm,uns=perms.transpose(1,0,2)
                    for m,v in enumerate(hats):
                        out[prefix,kind,ie,j,m]=blend(lp,lm,d2,v.mean(-1),(v*v).mean(-1)).mean()
                    out[prefix,kind,ie,j,2]=uns.mean()
    return out


if __name__=='__main__':
    assert os.getenv('SLURM_JOB_ID')
    risks=np.stack([run(i) for i in range(12)])
    core=np.load(ROOT/'catalog51.npz')['risk'].mean((3,4))
    np.testing.assert_allclose(core[...,:3],risks[:,0,0],rtol=0,atol=2e-12)
    contrast=risks[...,0,None]-risks[...,[1,2]]
    ci=interval(contrast,72)
    data=dict(axes='prefix group(first/second four), replacement law(complement/fresh independent), epsilon, g, method',
              loss=risks.mean(0),paired95_per_sensitivity_family=ci,win_both=(ci[...,2]<0).all(-1),
              cpu_replay_error=float(abs(core[...,:3]-risks[:,0,0]).max()),
              scope='Last four use disjoint specimens within a given eight-pair prefix, not fresh catalogs. Each calibrated method gets exactly four in both checks. At a refresh new k uniformly sampled from existing five regimes, independent of pre-replacement k.')
    (ROOT/'controls51.json').write_text(json.dumps(data,default=lambda x:np.asarray(x).tolist(),indent=1))
